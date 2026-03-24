#!/usr/bin/env python3
"""
Export a directory of short talking-head videos into trimmed 256x256 faceclip
videos with mono 16kHz audio and processing metadata.

Output layout:
  output_dir/
    sample_name.mp4
    sample_name.json
    export_manifest.jsonl
    summary.json

Bad samples are discarded using the same quality gates as preprocess_dataset:
  - unstable / missing face track => fail
  - bad_sample=true (short/heavily-trimmed/near-edge/jumpy) => discard
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
sys.path.insert(0, TRAINING_ROOT)
sys.path.insert(0, REPO_ROOT)

from scripts.preprocess_dataset import (  # noqa: E402
    build_face_track,
    resolve_detector_device,
    resolve_resize_device,
    resize_face_crops,
)
from scripts.sort_talkvid_processed_by_quality import classify_sample  # noqa: E402
from scripts.transcode_video import (  # noqa: E402
    VIDEO_ENCODER_CHOICES,
    media_file_is_valid,
    normalize_video_clip,
    resolve_ffmpeg_bin,
    select_video_encoder,
)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def sample_complete(output_dir: Path, name: str, ffmpeg_bin: str) -> bool:
    for tier in ("confident", "medium", "unconfident"):
        mp4_path = output_dir / tier / f"{name}.mp4"
        meta_path = output_dir / tier / f"{name}.json"
        if not mp4_path.exists() or not meta_path.exists():
            continue
        if not media_file_is_valid(str(mp4_path), ffmpeg_bin):
            continue
        try:
            with open(meta_path) as f:
                json.load(f)
        except Exception:
            continue
        return True
    return False


def resolve_export_resize_device(resize_device: str) -> str:
    resolved = resolve_resize_device(resize_device)
    # Current MPS grid-sample crops are incorrect for this exporter and produce
    # top-left corner crops instead of the detected face. Keep the local
    # overnight pipeline on CPU resize until the MPS path is fixed properly.
    if resolved == "mps":
        return "cpu"
    return resolved


def iter_videos(input_dir: Path):
    for path in sorted(input_dir.glob("*.mp4")):
        if path.is_file():
            yield path


def detect_dataset_kind(archive_hint: str, input_dir: Path, explicit_kind: str) -> str:
    if explicit_kind != "auto":
        return explicit_kind
    lower_hint = (archive_hint or "").lower()
    if "talkvid" in lower_hint:
        return "talkvid"
    if "hdtf" in lower_hint:
        return "hdtf"
    if any(path.with_suffix(".json").exists() for path in input_dir.glob("*.mp4")):
        return "talkvid"
    return "hdtf"


def write_video_only_mp4(frames_bgr, fps: float, output_path: Path) -> None:
    if len(frames_bgr) == 0:
        raise RuntimeError("no_frames_to_write")

    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"video_writer_open_failed: {output_path}")
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


def mux_video_and_audio(
    video_only_path: Path,
    wav_path: Path,
    output_path: Path,
    *,
    ffmpeg_bin: str,
    ffmpeg_threads: int,
    video_encoder: str,
    video_bitrate: str,
    sample_rate: int = 16000,
    audio_bitrate: str = "128k",
    timeout: int = 180,
) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(wav_path),
    ]
    if ffmpeg_threads and ffmpeg_threads > 0:
        cmd.extend(["-threads", str(int(ffmpeg_threads))])
    if video_encoder == "libx264":
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
    elif video_encoder == "h264_videotoolbox":
        cmd.extend(["-c:v", "h264_videotoolbox", "-allow_sw", "1", "-b:v", str(video_bitrate)])
    elif video_encoder == "h264_nvenc":
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", str(video_bitrate)])
    else:
        raise ValueError(f"unsupported video_encoder={video_encoder}")
    cmd.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            str(audio_bitrate),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-shortest",
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)


def extract_audio_wav_detailed(
    video_path: Path,
    output_wav: Path,
    *,
    sample_rate: int,
    start_time: float,
    duration: float | None,
    ffmpeg_bin: str,
    ffmpeg_threads: int,
) -> tuple[bool, str]:
    cmd = [ffmpeg_bin, "-y"]
    if start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    cmd.extend(["-i", str(video_path)])
    if duration is not None and duration > 0:
        cmd.extend(["-t", f"{duration:.3f}"])
    if ffmpeg_threads and ffmpeg_threads > 0:
        cmd.extend(["-threads", str(int(ffmpeg_threads))])
    cmd.extend(["-ar", str(sample_rate), "-ac", "1", "-f", "wav", str(output_wav)])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        return True, "ok"
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        detail = " ".join(detail.split()) if detail else "subprocess_failed"
        return False, detail[-320:]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def normalize_if_needed(input_video: Path, normalized_path: Path, args) -> tuple[bool, str]:
    if args.input_is_normalized:
        if input_video != normalized_path:
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_video, normalized_path)
        return True, "already_normalized"

    ok, detail, encoder = normalize_video_clip(
        str(input_video),
        str(normalized_path),
        args.fps,
        ffmpeg_bin=args.ffmpeg_bin,
        ffmpeg_threads=args.ffmpeg_threads,
        video_encoder=args.video_encoder,
        video_bitrate=args.video_bitrate,
        timeout=args.ffmpeg_timeout,
    )
    if not ok:
        return False, f"normalize_fail encoder={encoder} detail={detail}"
    return True, f"normalized encoder={encoder} detail={detail}"


def export_one(video_path: Path, output_dir: Path, normalized_dir: Path, dataset_kind: str, args):
    name = video_path.stem
    if sample_complete(output_dir, name, args.ffmpeg_bin):
        return "skip", f"{name}: already exported"

    normalized_path = normalized_dir / f"{name}.mp4"
    ok, normalize_detail = normalize_if_needed(video_path, normalized_path, args)
    if not ok:
        return "fail", f"{name}: {normalize_detail}"

    t0 = time.time()
    cap = cv2.VideoCapture(str(normalized_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or float(args.fps)
    frames = []
    while len(frames) < args.max_frames:
        read_ok, frame = cap.read()
        if not read_ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 25:
        return "discard", f"{name}: too_short ({len(frames)} frames)"

    try:
        track = build_face_track(
            frames,
            detect_every=args.detect_every,
            smooth_window=args.smooth_window,
            detector_backend=args.detector_backend,
            detector_device=args.detector_device,
            detector_batch_size=args.detector_batch_size,
        )
    except Exception as exc:
        return "fail", f"{name}: detector_fail ({type(exc).__name__}: {exc})"

    if not track["ok"]:
        return "discard", f"{name}: {track['status']}"
    if track["bad_sample"]:
        return "discard", f"{name}: bad_sample reasons={','.join(track['bad_reasons'])}"

    trim_start = track["trim_start"]
    trim_end = track["trim_end"]
    trimmed_frames = frames[trim_start : trim_end + 1]
    trimmed_bboxes = track["bboxes"]
    trimmed_indices = track["frame_indices"]
    effective_resize_device = resolve_export_resize_device(args.resize_device)
    crops_arr = resize_face_crops(
        trimmed_frames,
        trimmed_bboxes,
        args.size,
        resize_device=effective_resize_device,
    )

    source_meta = load_json(video_path.with_suffix(".json"))
    if dataset_kind == "talkvid":
        tier, tier_reasons = classify_sample(
            {
                "bad_sample": False,
                "bad_reasons": [],
                "quality": track["quality"],
            },
            source_meta or {},
        )
        if tier == "rejected":
            return "discard", f"{name}: rejected_by_tier reasons={','.join(tier_reasons)}"
    else:
        tier = "confident"
        tier_reasons = ["curated_hdtf"]

    sample_tmp_dir = output_dir / ".tmp" / name
    sample_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_video_only = sample_tmp_dir / "video_only.mp4"
    tmp_wav = sample_tmp_dir / "audio.wav"
    tmp_final = sample_tmp_dir / f"{name}.mp4"
    tmp_meta = sample_tmp_dir / f"{name}.json"

    try:
        write_video_only_mp4(crops_arr, fps=float(args.fps), output_path=tmp_video_only)
        trim_duration = len(trimmed_frames) / float(fps)
        audio_ok, audio_detail = extract_audio_wav_detailed(
            normalized_path,
            tmp_wav,
            sample_rate=16000,
            start_time=trim_start / float(fps),
            duration=trim_duration,
            ffmpeg_bin=args.ffmpeg_bin,
            ffmpeg_threads=args.ffmpeg_threads,
        )
        if not audio_ok:
            return "fail", f"{name}: audio_fail detail={audio_detail}"

        mux_video_and_audio(
            tmp_video_only,
            tmp_wav,
            tmp_final,
            ffmpeg_bin=args.ffmpeg_bin,
            ffmpeg_threads=args.ffmpeg_threads,
            video_encoder=args.video_encoder,
            video_bitrate=args.video_bitrate,
            timeout=args.ffmpeg_timeout,
        )
        sample_tmp_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": name,
            "source_dataset": dataset_kind,
            "source_archive": args.source_archive,
            "source_video": video_path.name,
            "source_sidecar_present": source_meta is not None,
            "source_meta": source_meta,
            "fps": float(args.fps),
            "source_fps": float(fps),
            "n_frames": int(crops_arr.shape[0]),
            "img_size": int(args.size),
            "detector_backend": args.detector_backend,
            "detector_device": resolve_detector_device(args.detector_backend, args.detector_device),
            "detector_batch_size": int(args.detector_batch_size),
            "resize_device": effective_resize_device,
            "video_encoder": args.video_encoder,
            "trim_start_frame": int(trim_start),
            "trim_end_frame": int(trim_end),
            "trimmed_frame_indices": [int(v) for v in trimmed_indices.tolist()],
            "bad_sample": False,
            "bad_reasons": [],
            "quality": track["quality"],
            "quality_tier": tier,
            "quality_tier_reasons": tier_reasons,
            "normalize_detail": normalize_detail,
        }
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, ensure_ascii=False)

        tier_dir = output_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        final_mp4 = tier_dir / f"{name}.mp4"
        final_meta = tier_dir / f"{name}.json"
        os.replace(tmp_final, final_mp4)
        os.replace(tmp_meta, final_meta)
    finally:
        shutil.rmtree(sample_tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    details = (
        f"{name}: ok frames={crops_arr.shape[0]} trim={trim_start}:{trim_end} "
        f"ratio={track['quality'].get('kept_ratio', 0.0):.3f} "
        f"cov={track['quality'].get('detection_coverage', 0.0):.3f} "
        f"tier={tier} "
        f"({elapsed:.1f}s)"
    )
    return "ok", details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with source .mp4 files")
    parser.add_argument("--output-dir", required=True, help="Directory for exported faceclip .mp4/.json")
    parser.add_argument("--normalized-dir", required=True, help="Directory for normalized 25fps clips")
    parser.add_argument("--source-archive", default="", help="Source archive name for metadata/provenance")
    parser.add_argument("--dataset-kind", choices=["auto", "talkvid", "hdtf"], default="auto")
    parser.add_argument("--input-is-normalized", action="store_true", help="Treat input videos as already normalized 25fps/16k clips")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--detect-every", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--detector-backend", choices=["opencv", "sfd"], default="sfd")
    parser.add_argument("--detector-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--detector-batch-size", type=int, default=4)
    parser.add_argument("--resize-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--ffmpeg-timeout", type=int, default=180)
    parser.add_argument("--video-encoder", choices=VIDEO_ENCODER_CHOICES, default="auto")
    parser.add_argument("--video-bitrate", default="2200k")
    args = parser.parse_args()

    args.ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    args.video_encoder = select_video_encoder(args.video_encoder, args.ffmpeg_bin)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    normalized_dir = Path(args.normalized_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    dataset_kind = detect_dataset_kind(args.source_archive, input_dir, args.dataset_kind)
    manifest_path = output_dir / "export_manifest.jsonl"
    summary_path = output_dir / "summary.json"

    counters = {
        "ok": 0,
        "skip": 0,
        "discard": 0,
        "fail": 0,
        "confident": 0,
        "medium": 0,
        "unconfident": 0,
    }

    log(f"[FaceclipExport] input_dir={input_dir}")
    log(f"[FaceclipExport] output_dir={output_dir}")
    log(f"[FaceclipExport] normalized_dir={normalized_dir}")
    log(f"[FaceclipExport] source_archive={args.source_archive or '<none>'}")
    log(f"[FaceclipExport] dataset_kind={dataset_kind}")
    log(f"[FaceclipExport] detector={args.detector_backend} device={resolve_detector_device(args.detector_backend, args.detector_device)}")
    requested_resize_device = resolve_resize_device(args.resize_device)
    effective_resize_device = resolve_export_resize_device(args.resize_device)
    if requested_resize_device != effective_resize_device:
        log(
            "[FaceclipExport] resize_device="
            f"{requested_resize_device} -> {effective_resize_device} (MPS disabled for exporter)"
        )
    else:
        log(f"[FaceclipExport] resize_device={effective_resize_device}")
    log(f"[FaceclipExport] video_encoder={args.video_encoder}")

    videos = list(iter_videos(input_dir))
    log(f"[FaceclipExport] videos={len(videos)}")

    for idx, video_path in enumerate(videos, start=1):
        try:
            status, message = export_one(video_path, output_dir, normalized_dir, dataset_kind, args)
        except Exception as exc:
            status = "fail"
            message = f"{video_path.stem}: unexpected_fail ({type(exc).__name__}: {exc})"
        counters[status] += 1
        tier = None
        if status == "ok":
            for candidate_tier in ("confident", "medium", "unconfident"):
                if (output_dir / candidate_tier / f"{video_path.stem}.json").exists():
                    tier = candidate_tier
                    counters[candidate_tier] += 1
                    break
        log(f"[FaceclipExport] [{idx}/{len(videos)}] {message}")
        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "index": idx,
                "total": len(videos),
                "name": video_path.stem,
                "status": status,
                "tier": tier,
                "message": message,
                "source_video": video_path.name,
            },
        )

    summary = {
        "ts": timestamp(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "normalized_dir": str(normalized_dir),
        "source_archive": args.source_archive,
        "dataset_kind": dataset_kind,
        "video_encoder": args.video_encoder,
        "detector_backend": args.detector_backend,
        "detector_device": resolve_detector_device(args.detector_backend, args.detector_device),
        "resize_device": effective_resize_device,
        "total_videos": len(videos),
        **counters,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[FaceclipExport] summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
