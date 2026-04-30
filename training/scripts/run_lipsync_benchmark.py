#!/usr/bin/env python3
"""Canonical lip-sync benchmark runner using the external face_processing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACE_LANDMARKER_NAME = "face_landmarker_v2_with_blendshapes.task"
IMG_SIZE = 96
PREPARE_BITRATE = "4M"
LOGGER = logging.getLogger("lipsync_benchmark")

if Path("/usr/bin/ffmpeg").exists():
    os.environ["PATH"] = f"/usr/bin:{os.environ.get('PATH', '')}"


def configure_import_roots() -> Path | None:
    external_root = None
    env_root = os.environ.get("FACE_PROCESSING_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "face_processing" / "__init__.py").exists():
            external_root = candidate
    if external_root is None:
        for candidate in (REPO_ROOT.parent / "face_processing", REPO_ROOT.parents[1] / "face_processing"):
            if (candidate / "face_processing" / "__init__.py").exists():
                external_root = candidate
                break

    roots = [REPO_ROOT]
    if external_root is not None:
        roots.append(external_root)

    for root in roots:
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
    return external_root


FACE_PROCESSING_IMPORT_ROOT = configure_import_roots()

from face_framedata.cut import cut_face_video
from face_framedata.pipeline import process_video_framedata
from face_framedata.restore import restore_video
from face_processing.config import PipelineConfig
from lipsync_benchmark_common import (
    default_cache_root,
    get_mel_chunks_cached,
    load_frames_cached,
    load_model,
    prepare_custom_batch,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Lip-sync benchmark runner")
    parser.add_argument("--face", required=True, help="Video or image file with a face")
    parser.add_argument("--audio", required=True, help="Audio file")
    parser.add_argument("--checkpoint", required=True, help="Official Wav2Lip or local generator checkpoint")
    parser.add_argument("--outfile", default=None, help="Output video path")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for static image input")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--static", action="store_true", help="Use only the first frame")
    parser.add_argument("--resize_factor", "--resize-factor", type=int, default=1, help="Downscale input by this factor")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Inference device for the x96 model",
    )
    parser.add_argument(
        "--landmarker_device",
        "--landmarker-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Device for MediaPipe face landmarker",
    )
    parser.add_argument(
        "--face_landmarker_path",
        "--face-landmarker-path",
        default=None,
        help="Optional path to MediaPipe face landmarker task asset",
    )
    parser.add_argument("--roi_top", "--roi-top", type=float, default=None, help="Optional top ROI ratio")
    parser.add_argument("--roi_bottom", "--roi-bottom", type=float, default=None, help="Optional bottom ROI ratio")
    parser.add_argument("--cache_root", "--cache-root", default=None, help="Optional cache root")
    parser.add_argument("--no_cache", "--no-cache", action="store_true", help="Disable cache reuse")
    parser.add_argument(
        "--keep_intermediates",
        "--keep-intermediates",
        action="store_true",
        help="Keep prepared video, framedata, faceclips, and restored_noaudio next to the output",
    )
    parser.add_argument("--dump_face_dir", "--dump-face-dir", default=None, help="Optional PNG dump dir for x96 crops")
    return parser.parse_args()


def default_output_path(args) -> str:
    out_dir = REPO_ROOT / "training" / "output" / "benchmarks" / "lipsync"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.checkpoint).stem
    audio_name = Path(args.audio).stem
    face_name = Path(args.face).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(out_dir / f"{face_name}_{audio_name}_{ckpt_name}_{timestamp}.mp4")


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_ffmpeg_bin() -> str:
    ffmpeg_bin = os.environ.get("FFMPEG_BIN")
    if ffmpeg_bin:
        return ffmpeg_bin
    if Path("/usr/bin/ffmpeg").exists():
        return "/usr/bin/ffmpeg"
    return "ffmpeg"


def resolve_landmarker_path(user_path: str | None) -> str:
    candidates: list[Path] = []
    if user_path:
        candidates.append(Path(user_path))
    env_path = os.environ.get("FACE_LANDMARKER_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            REPO_ROOT / "models" / "face_processing" / DEFAULT_FACE_LANDMARKER_NAME,
            REPO_ROOT.parent / "face_processing" / "assets" / DEFAULT_FACE_LANDMARKER_NAME,
            REPO_ROOT.parents[1] / "face_processing" / "assets" / DEFAULT_FACE_LANDMARKER_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    checked = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find MediaPipe face landmarker asset. Checked:\n"
        f"{checked}\n"
        "Pass --face_landmarker_path explicitly if needed."
    )


def resolve_landmarker_use_gpu(requested: str) -> bool:
    if requested == "gpu":
        return True
    if requested == "cpu":
        return False
    return bool(torch.backends.mps.is_available())


def build_source_sequence(frames: list[np.ndarray], mel_count: int, static: bool) -> tuple[list[np.ndarray], str]:
    if not frames:
        raise RuntimeError("No source frames available")
    if mel_count <= 0:
        raise RuntimeError("Audio produced no mel chunks")
    if static:
        return [frames[0].copy() for _ in range(mel_count)], "static-repeat"
    if len(frames) >= mel_count:
        mode = "trimmed" if len(frames) > mel_count else "matched"
        return [frame.copy() for frame in frames[:mel_count]], mode
    return [frames[idx % len(frames)].copy() for idx in range(mel_count)], "cycled"


def write_video(path: str | Path, frames: list[np.ndarray], fps: float, bitrate: str = PREPARE_BITRATE) -> None:
    if not frames:
        raise RuntimeError(f"Cannot write empty video: {path}")
    height, width = frames[0].shape[:2]
    cmd = [
        resolve_ffmpeg_bin(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-b:v",
        bitrate,
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
    proc.wait(timeout=300)
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}:\n{stderr}")


def read_video_frames(video_path: str | Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def dump_face_crops_png(face_crops: list[np.ndarray], dump_dir: str | None) -> None:
    if not dump_dir:
        return
    out_dir = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_png in out_dir.glob("*.png"):
        old_png.unlink()
    for idx, crop in enumerate(face_crops):
        out_path = out_dir / f"{idx:06d}.png"
        if not cv2.imwrite(str(out_path), crop):
            raise RuntimeError(f"Failed to write face crop PNG: {out_path}")
    print(f"[dump] saved {len(face_crops)} face crops to {out_dir}", flush=True)


def make_official_input_batch(face_batch: list[np.ndarray], mel_batch: list[np.ndarray]):
    face_arr = np.asarray(face_batch, dtype=np.float32)
    mel_arr = np.asarray(mel_batch, dtype=np.float32)

    masked = face_arr.copy()
    masked[:, IMG_SIZE // 2 :, :, :] = 0

    img_batch = np.concatenate((masked, face_arr), axis=3) / 255.0
    mel_batch = mel_arr.reshape(len(mel_arr), mel_arr.shape[1], mel_arr.shape[2], 1)
    return img_batch, mel_batch


def iter_official_batches(face_crops: list[np.ndarray], mel_chunks, batch_size: int):
    crop_batch: list[np.ndarray] = []
    mel_batch: list[np.ndarray] = []
    for crop, mel in zip(face_crops, mel_chunks):
        crop_batch.append(crop)
        mel_batch.append(mel)
        if len(crop_batch) >= batch_size:
            yield make_official_input_batch(crop_batch, mel_batch)
            crop_batch, mel_batch = [], []
    if crop_batch:
        yield make_official_input_batch(crop_batch, mel_batch)


def validate_framedata(framedata_path: Path, expected_frames: int) -> None:
    payload = json.loads(framedata_path.read_text(encoding="utf-8"))
    total_frames = int(payload.get("total_frames", 0))
    if total_frames != expected_frames:
        raise RuntimeError(
            f"Framedata total_frames mismatch: {total_frames} vs expected {expected_frames}"
        )
    failures = [int(fr["i"]) for fr in payload.get("frames", []) if fr.get("status") == "fail"]
    if failures:
        preview = failures[:10]
        raise RuntimeError(
            "Face framedata contains fail frames. "
            f"count={len(failures)} first={preview}"
        )


def build_benchmark_config(
    landmarker_path: str,
    landmarker_use_gpu: bool,
    output_dir: Path,
    fps: float,
    roi_top_ratio: float | None = None,
    roi_bottom_ratio: float | None = None,
) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.output_dir = str(output_dir)
    cfg.keep_normalized = True
    cfg.detection.model_path = landmarker_path
    cfg.detection.use_gpu = landmarker_use_gpu
    if roi_top_ratio is not None:
        cfg.detection.roi_top_ratio = float(roi_top_ratio)
    if roi_bottom_ratio is not None:
        cfg.detection.roi_bottom_ratio = float(roi_bottom_ratio)
    cfg.normalization.ffmpeg_bin = resolve_ffmpeg_bin()
    cfg.normalization.fps = max(1, int(round(fps)))
    cfg.normalization.bitrate = PREPARE_BITRATE
    cfg.normalization.codec = "libx264"
    cfg.normalization.pixel_format = "yuv420p"
    return cfg


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    cmd = [
        resolve_ffmpeg_bin(),
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{proc.stderr}")


def copy_debug_artifacts(paths: dict[str, Path], outfile: Path) -> None:
    debug_dir = outfile.with_suffix("")
    debug_dir.mkdir(parents=True, exist_ok=True)
    for src in paths.values():
        if src.exists():
            shutil.copy2(src, debug_dir / src.name)
    LOGGER.info("Kept intermediates in %s", debug_dir)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t_start = time.time()
    device = resolve_device(args.device)
    outfile = Path(args.outfile or default_output_path(args))
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 50, flush=True)
    print("LipSync Benchmark", flush=True)
    print(f"Model device: {device}", flush=True)
    print(f"Landmarker device: {args.landmarker_device}", flush=True)
    print(f"face_processing import root: {FACE_PROCESSING_IMPORT_ROOT or 'installed package'}", flush=True)
    print("=" * 50, flush=True)

    landmarker_path = resolve_landmarker_path(args.face_landmarker_path)
    landmarker_use_gpu = resolve_landmarker_use_gpu(args.landmarker_device)
    print(f"[landmarker] asset: {landmarker_path}", flush=True)
    print(f"[landmarker] use_gpu={landmarker_use_gpu}", flush=True)

    cache_root = None if args.no_cache else (
        Path(args.cache_root).resolve() if args.cache_root else default_cache_root()
    )
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        print(f"[cache] root: {cache_root}", flush=True)

    is_image = is_image_path(args.face)
    static = bool(args.static or is_image)
    (frames, video_fps), _ = load_frames_cached(args.face, args.resize_factor, cache_root)
    fps = float(video_fps or args.fps)

    model, model_meta = load_model(args.checkpoint, device=device)
    if int(model_meta["img_size"]) != IMG_SIZE:
        raise RuntimeError(f"run_lipsync_benchmark expects img_size={IMG_SIZE}, got {model_meta['img_size']}")

    mel_chunks, _ = get_mel_chunks_cached(
        args.audio,
        fps,
        audio_cfg=model_meta["audio_cfg"],
        mel_step_size=model_meta["mel_step_size"],
        cache_root=cache_root,
    )
    prepared_frames, sequence_mode = build_source_sequence(frames, len(mel_chunks), static)
    print(
        "[sync] Prepared "
        f"{len(prepared_frames)} frames for {len(mel_chunks)} mel chunks ({sequence_mode})",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="lipsync_benchmark_") as temp_dir:
        work_dir = Path(temp_dir)
        prepared_video_path = work_dir / "prepared_source.mp4"
        framedata_root = work_dir / "framedata"
        face_video_path = work_dir / "prepared_source_face96.mp4"
        generated_face_path = work_dir / "generated_face96.mp4"
        restored_noaudio_path = work_dir / "restored_noaudio.mp4"

        write_video(prepared_video_path, prepared_frames, fps)
        cfg = build_benchmark_config(
            landmarker_path,
            landmarker_use_gpu,
            output_dir=framedata_root,
            fps=fps,
            roi_top_ratio=args.roi_top,
            roi_bottom_ratio=args.roi_bottom,
        )

        t_analysis = time.time()
        report = process_video_framedata(str(prepared_video_path), cfg)
        source_name = prepared_video_path.stem
        source_dir = framedata_root / source_name
        framedata_path = source_dir / f"{source_name}_framedata.json"
        normalized_path = source_dir / "normalized.mp4"
        validate_framedata(framedata_path, len(prepared_frames))
        print(
            f"[framedata] analyzed {report['total_frames']} frames in {time.time() - t_analysis:.1f}s",
            flush=True,
        )

        t_cut = time.time()
        cut_face_video(
            framedata_path=str(framedata_path),
            video_path=str(normalized_path),
            output_path=str(face_video_path),
            output_size=IMG_SIZE,
            fps=max(1, int(round(fps))),
            ffmpeg_bin=cfg.normalization.ffmpeg_bin,
            ffmpeg_timeout=cfg.normalization.ffmpeg_timeout,
        )
        face_crops = read_video_frames(face_video_path)
        if len(face_crops) != len(prepared_frames):
            raise RuntimeError(
                f"Extracted face frame count mismatch: {len(face_crops)} vs {len(prepared_frames)}"
            )
        dump_face_crops_png(face_crops, args.dump_face_dir)
        print(
            f"[cut] exported {len(face_crops)} x96 faceclip frames in {time.time() - t_cut:.1f}s",
            flush=True,
        )

        predicted_patches: list[np.ndarray] = []
        t_inf = time.time()
        total_batches = int(np.ceil(len(mel_chunks) / args.batch_size))

        if model_meta["kind"] == "official":
            generator = iter_official_batches(face_crops, mel_chunks, args.batch_size)
            for img_batch, mel_batch in tqdm(generator, total=total_batches, desc="[infer] Wav2Lip"):
                img_tensor = torch.from_numpy(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_tensor = torch.from_numpy(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                with torch.inference_mode():
                    pred = model(mel_tensor, img_tensor)
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                predicted_patches.extend(patch.astype(np.uint8) for patch in pred)
        else:
            syncnet_T = int(model_meta["syncnet_T"])
            for start in tqdm(
                range(0, len(mel_chunks), args.batch_size),
                total=total_batches,
                desc="[infer] LocalGen",
            ):
                indices = list(range(start, min(start + args.batch_size, len(mel_chunks))))
                face_tensor, mel_tensor = prepare_custom_batch(
                    face_crops,
                    mel_chunks,
                    indices,
                    img_size=IMG_SIZE,
                    syncnet_T=syncnet_T,
                    device=device,
                )
                with torch.inference_mode():
                    pred = model(mel_tensor, face_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                if pred.dim() == 5:
                    pred = pred[:, :, pred.shape[2] // 2]
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                predicted_patches.extend(patch.astype(np.uint8) for patch in pred)

        if len(predicted_patches) != len(mel_chunks):
            raise RuntimeError(
                f"Predicted patch count mismatch: {len(predicted_patches)} vs {len(mel_chunks)}"
            )
        inf_time = time.time() - t_inf
        print(
            f"[infer] {len(predicted_patches)} frames in {inf_time:.1f}s "
            f"({len(predicted_patches) / max(inf_time, 1e-6):.1f} FPS)",
            flush=True,
        )

        write_video(generated_face_path, predicted_patches, fps)

        print("[restore] Pasting generated face back into full frames...", flush=True)
        restore_video(
            framedata_path=str(framedata_path),
            face_video_path=str(generated_face_path),
            normalized_path=str(normalized_path),
            output_path=str(restored_noaudio_path),
            ffmpeg_bin=cfg.normalization.ffmpeg_bin,
            ffmpeg_timeout=cfg.normalization.ffmpeg_timeout,
        )

        print("[mux] Adding audio track...", flush=True)
        mux_audio(restored_noaudio_path, Path(args.audio), outfile)

        if args.keep_intermediates:
            copy_debug_artifacts(
                {
                    "prepared_video": prepared_video_path,
                    "framedata": framedata_path,
                    "normalized": normalized_path,
                    "face_video": face_video_path,
                    "generated_face": generated_face_path,
                    "restored_noaudio": restored_noaudio_path,
                },
                outfile,
            )

    total_time = time.time() - t_start
    print("\n" + "=" * 50, flush=True)
    print(f"Done! Total: {total_time:.1f}s", flush=True)
    print(f"Output: {outfile}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
