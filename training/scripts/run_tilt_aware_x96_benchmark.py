#!/usr/bin/env python3
"""
Tilt-aware x96 benchmark runner.

Inference stays on the same 96x96 Wav2Lip-like checkpoints as the reference
benchmark. What changes is the face preparation / paste-back path:

video or image
-> MediaPipe face analysis
-> roll-aware stabilized pad-to-square native faceclip
-> resize native faceclip to 96x96 for model inference
-> resize generated 96x96 patch back to native faceclip size
-> inverse affine paste-back into the full frame
-> mux audio
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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

def configure_import_roots() -> Path:
    external_root = None
    env_root = os.environ.get("FACE_PROCESSING_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "face_processing" / "__init__.py").exists():
            external_root = candidate
    if external_root is None:
        candidate = REPO_ROOT.parents[1] / "face_processing"
        if (candidate / "face_processing" / "__init__.py").exists():
            external_root = candidate

    roots = [REPO_ROOT]
    if external_root is not None:
        roots.insert(0, external_root)

    for root in reversed(roots):
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)

    return external_root or REPO_ROOT


FACE_PROCESSING_IMPORT_ROOT = configure_import_roots()

from face_processing.config import PipelineConfig
from face_processing.crop_export import compute_output_size, export_segment, prepare_segment_crop_geometry
from face_processing.face_analysis import analyze_frames
from face_processing.logging_utils import save_frame_log
from face_processing.models import Segment
from face_processing.frame_quality import smooth_pose
from face_processing.normalize import normalize_video
from face_processing.restore import restore_segment
from run_official_wav2lip_benchmark import (
    default_cache_root,
    get_mel_chunks_cached,
    load_frames_cached,
    load_model,
    prepare_custom_batch,
    resolve_device,
)

IMG_SIZE = 96
BENCH_EXPORT_MODE = "pad_to_square"
BENCH_STABILIZED = True
LOGGER = logging.getLogger("tilt_aware_x96_benchmark")

print(f"[TiltBench] face_processing import root: {FACE_PROCESSING_IMPORT_ROOT}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Tilt-aware x96 benchmark runner")
    parser.add_argument("--face", required=True, help="Video or image file with a face")
    parser.add_argument("--audio", default=None, help="Audio file (wav/mp3/...)")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Official Wav2Lip or local x96 generator checkpoint",
    )
    parser.add_argument("--outfile", default=None, help="Output video path")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for static image input")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--static", action="store_true", help="Use only the first frame")
    parser.add_argument("--resize_factor", type=int, default=1, help="Downscale input by this factor")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Inference device for the x96 model",
    )
    parser.add_argument(
        "--landmarker_device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Device for MediaPipe face landmarker",
    )
    parser.add_argument(
        "--face_landmarker_path",
        default=None,
        help="Optional path to MediaPipe face landmarker task asset",
    )
    parser.add_argument(
        "--cache_root",
        default=None,
        help="Optional cache root for decoded frames and audio features",
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable sample cache reuse")
    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        help="Keep temporary prepared video / metadata / generated crops next to the output",
    )
    parser.add_argument(
        "--dump_face_dir",
        default=None,
        help="Optional directory to save 96x96 PNG face crops used by the benchmark",
    )
    parser.add_argument(
        "--passthrough_faces",
        action="store_true",
        help="Skip generator inference and restore the extracted tilt-aware face clips directly",
    )
    parser.add_argument(
        "--mode",
        default="stretch_to_square",
        choices=("stretch_to_square", "pad_to_square"),
        help="Export mode for --passthrough_faces roundtrip; matches roundtrip_main.py",
    )
    parser.add_argument(
        "--stabilized",
        action="store_true",
        help="Enable stabilized crops for --passthrough_faces roundtrip; matches roundtrip_main.py",
    )
    return parser.parse_args()


def default_output_path(args) -> str:
    out_dir = REPO_ROOT / "training" / "output" / "benchmarks" / "tilt_aware_x96"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.checkpoint).stem if args.checkpoint else "passthrough_faces"
    audio_name = Path(args.audio).stem if args.audio else "noaudio"
    face_name = Path(args.face).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(out_dir / f"{face_name}_{audio_name}_{ckpt_name}_{timestamp}.mp4")


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


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


def write_video(path: str | Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError(f"Cannot write empty video: {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def write_face_video(path: str | Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError(f"Cannot write empty face video: {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open face video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


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
    print(f"[dump] saved {len(face_crops)} face crops to {out_dir}")


def resize_face_crops(face_crops: list[np.ndarray], size: int) -> list[np.ndarray]:
    resized: list[np.ndarray] = []
    for crop in face_crops:
        if crop.shape[0] == size and crop.shape[1] == size:
            resized.append(crop.copy())
        else:
            resized.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR))
    return resized


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


def validate_face_analysis(frame_data, expected_frames: int) -> None:
    if len(frame_data) != expected_frames:
        raise RuntimeError(
            f"Face analysis returned {len(frame_data)} frames, expected {expected_frames}"
        )

    missing_landmarks = [fd.frame_idx for fd in frame_data if fd.landmarks is None]
    if missing_landmarks:
        preview = missing_landmarks[:10]
        raise RuntimeError(
            "Face landmarks missing for prepared frames. "
            f"count={len(missing_landmarks)} first={preview}"
        )

    missing_pose = [fd.frame_idx for fd in frame_data if not fd.pose_valid]
    if missing_pose:
        preview = missing_pose[:10]
        raise RuntimeError(
            "Head pose / roll missing for prepared frames. "
            f"count={len(missing_pose)} first={preview}"
        )


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


def copy_debug_artifacts(work_dir: Path, outfile: Path) -> None:
    debug_dir = outfile.with_suffix("")
    debug_dir.mkdir(parents=True, exist_ok=True)
    for item in work_dir.iterdir():
        shutil.copy2(item, debug_dir / item.name)
    LOGGER.info("Kept intermediates in %s", debug_dir)


def build_pipeline_config(
    landmarker_path: str,
    landmarker_use_gpu: bool,
    *,
    export_mode: str,
    stabilization_enabled: bool,
    export_fps: float | None = None,
) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.detection.model_path = landmarker_path
    cfg.detection.use_gpu = landmarker_use_gpu
    cfg.stabilization.enabled = stabilization_enabled
    cfg.save_frame_log = True
    cfg.export.mode = export_mode
    if export_fps is not None:
        cfg.export.fps = max(1, int(round(export_fps)))
    return cfg


def clear_stable_crop_fields(frame_data) -> None:
    for fd in frame_data:
        fd.stable_crop_cx_rot = None
        fd.stable_crop_cy_rot = None
        fd.stable_crop_w_rot = None
        fd.stable_crop_h_rot = None
        fd.scale_deviation_ratio = None


def run_passthrough_roundtrip(
    args,
    outfile: Path,
    landmarker_path: str,
    landmarker_use_gpu: bool,
) -> None:
    cfg = build_pipeline_config(
        landmarker_path,
        landmarker_use_gpu,
        export_mode=args.mode,
        stabilization_enabled=bool(args.stabilized),
    )

    with tempfile.TemporaryDirectory(prefix="tilt_aware_roundtrip_") as temp_dir:
        work_dir = Path(temp_dir)
        normalized_path = work_dir / "normalized.mp4"
        face_video_path = work_dir / "extracted_face.mp4"
        frame_log_path = work_dir / "frame_log.csv"
        segment_json_path = work_dir / "segment.json"

        normalize_video(args.face, str(normalized_path), cfg.normalization)

        cap = cv2.VideoCapture(str(normalized_path))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            raise RuntimeError(f"No frames found after normalization: {normalized_path}")

        t_analysis = time.time()
        frame_data = analyze_frames(str(normalized_path), cfg.detection)
        validate_face_analysis(frame_data, total_frames)
        if cfg.stabilization.enabled:
            smooth_pose(frame_data, window=cfg.stabilization.window)
        print(f"[landmarker] analyzed {len(frame_data)} frames in {time.time() - t_analysis:.1f}s")

        segment = Segment(
            segment_id=0,
            start_frame=0,
            end_frame=total_frames,
            length=total_frames,
            frame_data=frame_data,
            status="exported",
        )
        output_size = compute_output_size(segment, frame_w, frame_h)
        segment.output_size = output_size
        prepare_segment_crop_geometry(segment, frame_w, frame_h, cfg.stabilization)
        if not cfg.stabilization.enabled:
            clear_stable_crop_fields(frame_data)

        save_frame_log(frame_data, str(frame_log_path), segments=[segment])
        segment_json_path.write_text(
            json.dumps(
                segment.to_dict(
                    source_video=os.path.basename(args.face),
                    export_mode=cfg.export.mode,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        t_crop = time.time()
        export_segment(
            segment=segment,
            video_path=str(normalized_path),
            output_path=str(face_video_path),
            frame_w=frame_w,
            frame_h=frame_h,
            output_size=output_size,
            config=cfg.export,
            source_video_path=str(args.face),
            use_stabilized_crop=cfg.stabilization.enabled,
        )
        print(
            f"[crop] exported {total_frames} faceclip frames at "
            f"{output_size}x{output_size} in {time.time() - t_crop:.1f}s"
        )
        if args.dump_face_dir:
            dump_face_crops_png(read_video_frames(face_video_path), args.dump_face_dir)

        print("[passthrough] Restoring extracted face clips directly...")
        restore_segment(
            segment_json_path=str(segment_json_path),
            face_video_path=str(face_video_path),
            frame_log_path=str(frame_log_path),
            normalized_video_path=str(normalized_path),
            output_path=str(outfile),
            source_audio_path=None,
        )

        if args.keep_intermediates:
            copy_debug_artifacts(work_dir, outfile)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t_start = time.time()
    device = resolve_device(args.device)

    print("=" * 50)
    print("Tilt-Aware x96 Benchmark")
    print(f"Model device: {device}")
    print(f"Landmarker device: {args.landmarker_device}")
    print("=" * 50)

    if not args.passthrough_faces and not args.audio:
        raise RuntimeError("--audio is required unless --passthrough_faces is set")
    if not args.passthrough_faces and not args.checkpoint:
        raise RuntimeError("--checkpoint is required unless --passthrough_faces is set")

    outfile = Path(args.outfile or default_output_path(args))
    outfile.parent.mkdir(parents=True, exist_ok=True)

    landmarker_path = resolve_landmarker_path(args.face_landmarker_path)
    landmarker_use_gpu = resolve_landmarker_use_gpu(args.landmarker_device)
    print(f"[landmarker] asset: {landmarker_path}")
    print(f"[landmarker] use_gpu={landmarker_use_gpu}")

    if args.passthrough_faces:
        run_passthrough_roundtrip(args, outfile, landmarker_path, landmarker_use_gpu)
        total_time = time.time() - t_start
        print("\n" + "=" * 50)
        print(f"Done! Total: {total_time:.1f}s")
        print(f"Output: {outfile}")
        print("=" * 50)
        return

    cache_root = None if args.no_cache else (
        Path(args.cache_root).resolve() if args.cache_root else default_cache_root()
    )
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        print(f"[cache] root: {cache_root}")

    is_image = is_image_path(args.face)
    static = bool(args.static or is_image)

    (frames, video_fps), _ = load_frames_cached(args.face, args.resize_factor, cache_root)
    fps = float(video_fps or args.fps)

    model, model_meta = load_model(args.checkpoint, device=device)
    if int(model_meta["img_size"]) != IMG_SIZE:
        raise RuntimeError(
            f"tilt_aware_x96_benchmark expects img_size={IMG_SIZE}, got {model_meta['img_size']}"
        )

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
        f"{len(prepared_frames)} frames for {len(mel_chunks)} mel chunks ({sequence_mode})"
    )

    frame_h, frame_w = prepared_frames[0].shape[:2]

    with tempfile.TemporaryDirectory(prefix="tilt_aware_x96_") as temp_dir:
        work_dir = Path(temp_dir)
        prepared_video_path = work_dir / "prepared_source.avi"
        extracted_face_path = work_dir / "extracted_face.avi"
        generated_face_path = work_dir / "generated_face.avi"
        frame_log_path = work_dir / "frame_log.csv"
        segment_json_path = work_dir / "segment.json"

        write_video(prepared_video_path, prepared_frames, fps)

        cfg = build_pipeline_config(
            landmarker_path,
            landmarker_use_gpu,
            export_mode=BENCH_EXPORT_MODE,
            stabilization_enabled=BENCH_STABILIZED,
            export_fps=fps,
        )

        t_analysis = time.time()
        frame_data = analyze_frames(str(prepared_video_path), cfg.detection)
        validate_face_analysis(frame_data, len(prepared_frames))
        if cfg.stabilization.enabled:
            smooth_pose(frame_data, window=cfg.stabilization.window)
        print(f"[landmarker] analyzed {len(frame_data)} frames in {time.time() - t_analysis:.1f}s")

        segment = Segment(
            segment_id=0,
            start_frame=0,
            end_frame=len(prepared_frames),
            length=len(prepared_frames),
            frame_data=frame_data,
        )
        native_output_size = compute_output_size(segment, frame_w, frame_h)
        segment.output_size = native_output_size
        segment.status = "exported"
        prepare_segment_crop_geometry(segment, frame_w, frame_h, cfg.stabilization)
        if not cfg.stabilization.enabled:
            clear_stable_crop_fields(frame_data)

        save_frame_log(frame_data, str(frame_log_path), segments=[segment])
        segment_json_path.write_text(
            json.dumps(
                {
                    **segment.to_dict(
                        source_video=os.path.basename(args.face),
                        export_mode=cfg.export.mode,
                    ),
                    "benchmark_kind": "tilt_aware_x96",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        t_crop = time.time()
        export_segment(
            segment=segment,
            video_path=str(prepared_video_path),
            output_path=str(extracted_face_path),
            frame_w=frame_w,
            frame_h=frame_h,
            output_size=native_output_size,
            config=cfg.export,
            source_video_path=None,
            use_stabilized_crop=cfg.stabilization.enabled,
        )
        native_face_crops = read_video_frames(extracted_face_path)
        if len(native_face_crops) != len(prepared_frames):
            raise RuntimeError(
                f"Extracted face frame count mismatch: {len(native_face_crops)} vs {len(prepared_frames)}"
            )
        print(
            f"[crop] exported {len(native_face_crops)} faceclip frames at "
            f"{native_output_size}x{native_output_size} in {time.time() - t_crop:.1f}s"
        )
        face_crops = resize_face_crops(native_face_crops, IMG_SIZE)
        dump_face_crops_png(face_crops, args.dump_face_dir)

        face_video_path = str(extracted_face_path)
        if args.passthrough_faces:
            print("[passthrough] Skipping generator inference; restoring extracted face clips directly...")
        else:
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
                f"({len(predicted_patches) / max(inf_time, 1e-6):.1f} FPS)"
            )

            restore_patches = resize_face_crops(predicted_patches, native_output_size)
            write_face_video(generated_face_path, restore_patches, fps)
            face_video_path = str(generated_face_path)

        print("[restore] Pasting generated face back into full frames...")
        restore_segment(
            segment_json_path=str(segment_json_path),
            face_video_path=face_video_path,
            frame_log_path=str(frame_log_path),
            normalized_video_path=str(prepared_video_path),
            output_path=str(outfile),
            source_audio_path=args.audio,
        )

        if args.keep_intermediates:
            copy_debug_artifacts(work_dir, outfile)

    total_time = time.time() - t_start
    print("\n" + "=" * 50)
    print(f"Done! Total: {total_time:.1f}s")
    print(f"Output: {outfile}")
    print("=" * 50)


if __name__ == "__main__":
    main()
