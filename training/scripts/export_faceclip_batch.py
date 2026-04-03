#!/usr/bin/env python3
"""
Export a directory of talking-head videos into trimmed faceclip videos with
mono 16kHz audio and processing metadata.

Long source videos are segmented before face export:
  - if remaining_frames <= max_frames: keep as one segment
  - if remaining_frames <= 2 * max_frames: split the remainder in half
  - otherwise cut max_frames and continue on the remainder

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
    build_video_codec_args,
    media_file_is_valid,
    normalize_video_clip,
    resolve_ffmpeg_bin,
    select_video_encoder,
)
from data.sync_alignment import default_sync_alignment_block  # noqa: E402


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


def serialize_detection_records(detections, frame_offset: int = 0):
    serialized = []
    for record in detections or []:
        raw_bbox = record.get("raw_bbox")
        bbox = record.get("bbox")
        serialized.append(
            {
                "frame_idx": int(record.get("frame_idx", -1)) + int(frame_offset),
                "raw_bbox": None if raw_bbox is None else [int(v) for v in raw_bbox],
                "bbox": None if bbox is None else [int(v) for v in bbox],
                "reason": str(record.get("reason", "")),
                "score": None if record.get("score") is None else float(record.get("score")),
                "passed_score_gate": bool(record.get("passed_score_gate", False)),
                "edge_margin_ratio": float(record.get("edge_margin_ratio", 0.0)),
                "passed_edge_gate": bool(record.get("passed_edge_gate", False)),
                "in_selected_span": bool(record.get("in_selected_span", False)),
            }
        )
    return serialized


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


def probe_video(path: Path, fallback_fps: float) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"video_open_failed: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or float(fallback_fps)
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if frame_count > 0:
            return float(fps), frame_count
        count = 0
        while True:
            read_ok, _ = cap.read()
            if not read_ok:
                break
            count += 1
        return float(fps), count
    finally:
        cap.release()


def load_frame_range(video_path: Path, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"video_open_failed: {video_path}")
    frames = []
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        frame_idx = int(start_frame)
        while frame_idx < int(end_frame):
            read_ok, frame = cap.read()
            if not read_ok:
                break
            frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()
    return frames


def compute_segment_ranges(total_frames: int, max_frames: int) -> list[tuple[int, int]]:
    total_frames = int(total_frames)
    max_frames = max(1, int(max_frames))
    if total_frames <= 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    remaining = total_frames
    while remaining > 0:
        if remaining <= max_frames:
            ranges.append((start, start + remaining))
            break
        if remaining <= (2 * max_frames):
            first = remaining // 2
            second = remaining - first
            ranges.append((start, start + first))
            ranges.append((start + first, start + first + second))
            break
        ranges.append((start, start + max_frames))
        start += max_frames
        remaining = total_frames - start
    return ranges


def build_segment_name(base_name: str, segment_index: int, segment_count: int) -> str:
    if int(segment_count) <= 1:
        return base_name
    return f"{base_name}_part{int(segment_index):03d}"


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
    cmd.extend(build_video_codec_args(video_encoder, video_bitrate))
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
        try:
            source_fps, _ = probe_video(input_video, float(args.fps))
        except Exception as exc:
            return False, f"normalize_probe_fail ({type(exc).__name__}: {exc})"

        # HDTF "already normalized" clips should be CFR 25fps. If a batch
        # contains a stray clip at another frame rate, re-normalize it instead
        # of blindly copying and skewing downstream timing.
        if abs(float(source_fps) - float(args.fps)) <= 0.01:
            if input_video != normalized_path:
                normalized_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_video, normalized_path)
            return True, f"already_normalized fps={source_fps:.3f}"

    ok, detail, encoder = normalize_video_clip(
        str(input_video),
        str(normalized_path),
        args.fps,
        ffmpeg_bin=args.ffmpeg_bin,
        ffmpeg_threads=args.ffmpeg_threads,
        video_encoder=args.video_encoder,
        video_bitrate=args.normalized_video_bitrate,
        timeout=args.ffmpeg_timeout,
    )
    if not ok:
        return False, f"normalize_fail encoder={encoder} detail={detail}"
    if args.input_is_normalized:
        return True, f"renormalized_from_fps={source_fps:.3f} encoder={encoder} detail={detail}"
    return True, f"normalized encoder={encoder} detail={detail}"


def _segment_result(
    *,
    name: str,
    status: str,
    message: str,
    tier: str | None,
    segment_index: int,
    segment_count: int,
    segment_start_frame: int,
    segment_end_frame: int,
) -> dict:
    return {
        "name": name,
        "status": status,
        "message": message,
        "tier": tier,
        "segment_index": int(segment_index),
        "segment_count": int(segment_count),
        "source_segment_start_frame": int(segment_start_frame),
        "source_segment_end_frame": int(segment_end_frame),
    }


def export_segment(
    *,
    video_path: Path,
    normalized_path: Path,
    output_dir: Path,
    dataset_kind: str,
    source_meta,
    normalize_detail: str,
    source_fps: float,
    source_total_frames: int,
    frames,
    segment_name: str,
    segment_index: int,
    segment_count: int,
    segment_start_frame: int,
    segment_end_exclusive: int,
    args,
) -> dict:
    if sample_complete(output_dir, segment_name, args.ffmpeg_bin):
        return _segment_result(
            name=segment_name,
            status="skip",
            message=f"{segment_name}: already exported",
            tier=None,
            segment_index=segment_index,
            segment_count=segment_count,
            segment_start_frame=segment_start_frame,
            segment_end_frame=segment_end_exclusive - 1,
        )

    t0 = time.time()
    if len(frames) < 25:
        return _segment_result(
            name=segment_name,
            status="discard",
            message=f"{segment_name}: too_short ({len(frames)} frames)",
            tier=None,
            segment_index=segment_index,
            segment_count=segment_count,
            segment_start_frame=segment_start_frame,
            segment_end_frame=segment_end_exclusive - 1,
        )

    try:
        track = build_face_track(
            frames,
            detect_every=args.detect_every,
            smooth_window=args.smooth_window,
            smoothing_style=args.smoothing_style,
            framing_style=args.framing_style,
            inference_pads=tuple(args.inference_pads),
            detector_backend=args.detector_backend,
            detector_device=args.detector_device,
            detector_batch_size=args.detector_batch_size,
            min_detector_score=args.min_detector_score,
        )
    except Exception as exc:
        return _segment_result(
            name=segment_name,
            status="fail",
            message=f"{segment_name}: detector_fail ({type(exc).__name__}: {exc})",
            tier=None,
            segment_index=segment_index,
            segment_count=segment_count,
            segment_start_frame=segment_start_frame,
            segment_end_frame=segment_end_exclusive - 1,
        )

    if not track["ok"]:
        return _segment_result(
            name=segment_name,
            status="discard",
            message=f"{segment_name}: {track['status']}",
            tier=None,
            segment_index=segment_index,
            segment_count=segment_count,
            segment_start_frame=segment_start_frame,
            segment_end_frame=segment_end_exclusive - 1,
        )
    if track["bad_sample"]:
        return _segment_result(
            name=segment_name,
            status="discard",
            message=f"{segment_name}: bad_sample reasons={','.join(track['bad_reasons'])}",
            tier=None,
            segment_index=segment_index,
            segment_count=segment_count,
            segment_start_frame=segment_start_frame,
            segment_end_frame=segment_end_exclusive - 1,
        )

    trim_start = int(track["trim_start"])
    trim_end = int(track["trim_end"])
    trimmed_frames = frames[trim_start : trim_end + 1]
    trimmed_bboxes = track["bboxes"]
    trimmed_indices = track["frame_indices"]
    absolute_trim_start = int(segment_start_frame + trim_start)
    absolute_trim_end = int(segment_start_frame + trim_end)
    absolute_trimmed_indices = [int(segment_start_frame + v) for v in trimmed_indices.tolist()]

    effective_resize_device = resolve_export_resize_device(args.resize_device)
    crops_arr = resize_face_crops(
        trimmed_frames,
        trimmed_bboxes,
        args.size,
        resize_device=effective_resize_device,
    )

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
            return _segment_result(
                name=segment_name,
                status="discard",
                message=f"{segment_name}: rejected_by_tier reasons={','.join(tier_reasons)}",
                tier=None,
                segment_index=segment_index,
                segment_count=segment_count,
                segment_start_frame=segment_start_frame,
                segment_end_frame=segment_end_exclusive - 1,
            )
    else:
        tier = "confident"
        tier_reasons = ["curated_hdtf"]

    sample_tmp_dir = output_dir / ".tmp" / segment_name
    sample_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_video_only = sample_tmp_dir / "video_only.mp4"
    tmp_wav = sample_tmp_dir / "audio.wav"
    tmp_final = sample_tmp_dir / f"{segment_name}.mp4"
    tmp_meta = sample_tmp_dir / f"{segment_name}.json"
    tmp_detections = sample_tmp_dir / f"{segment_name}.detections.json"

    try:
        write_video_only_mp4(crops_arr, fps=float(args.fps), output_path=tmp_video_only)
        trim_duration = len(trimmed_frames) / float(source_fps)
        audio_ok, audio_detail = extract_audio_wav_detailed(
            normalized_path,
            tmp_wav,
            sample_rate=16000,
            start_time=absolute_trim_start / float(source_fps),
            duration=trim_duration,
            ffmpeg_bin=args.ffmpeg_bin,
            ffmpeg_threads=args.ffmpeg_threads,
        )
        if not audio_ok:
            return _segment_result(
                name=segment_name,
                status="fail",
                message=f"{segment_name}: audio_fail detail={audio_detail}",
                tier=None,
                segment_index=segment_index,
                segment_count=segment_count,
                segment_start_frame=segment_start_frame,
                segment_end_frame=segment_end_exclusive - 1,
            )

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
            "name": segment_name,
            "source_dataset": dataset_kind,
            "source_archive": args.source_archive,
            "source_video": video_path.name,
            "source_sidecar_present": source_meta is not None,
            "source_meta": source_meta,
            "fps": float(args.fps),
            "source_fps": float(source_fps),
            "source_total_frames": int(source_total_frames),
            "n_frames": int(crops_arr.shape[0]),
            "img_size": int(args.size),
            "source_segment_index": int(segment_index),
            "source_segment_count": int(segment_count),
            "source_segment_start_frame": int(segment_start_frame),
            "source_segment_end_frame": int(segment_end_exclusive - 1),
            "source_segment_n_frames": int(segment_end_exclusive - segment_start_frame),
            "detector_backend": args.detector_backend,
            "detector_device": resolve_detector_device(args.detector_backend, args.detector_device),
            "detector_batch_size": int(args.detector_batch_size),
            "min_detector_score": float(args.min_detector_score),
            "framing_style": args.framing_style,
            "inference_pads": [int(v) for v in args.inference_pads],
            "smoothing_style": args.smoothing_style,
            "smooth_window": int(args.smooth_window),
            "resize_device": effective_resize_device,
            "video_encoder": args.video_encoder,
            "normalized_video_bitrate": args.normalized_video_bitrate,
            "video_bitrate": args.video_bitrate,
            "segment_trim_start_frame": int(trim_start),
            "segment_trim_end_frame": int(trim_end),
            "trim_start_frame": int(absolute_trim_start),
            "trim_end_frame": int(absolute_trim_end),
            "trimmed_frame_indices": absolute_trimmed_indices,
            "bad_sample": False,
            "bad_reasons": [],
            "quality": track["quality"],
            "quality_tier": tier,
            "quality_tier_reasons": tier_reasons,
            "normalize_detail": normalize_detail,
            "sync_alignment": default_sync_alignment_block(),
        }
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        with open(tmp_detections, "w") as f:
            json.dump(
                {
                    "name": segment_name,
                    "source_video": video_path.name,
                    "source_total_frames": int(source_total_frames),
                    "source_segment_index": int(segment_index),
                    "source_segment_count": int(segment_count),
                    "source_segment_start_frame": int(segment_start_frame),
                    "source_segment_end_frame": int(segment_end_exclusive - 1),
                    "detector_backend": args.detector_backend,
                    "detector_device": resolve_detector_device(args.detector_backend, args.detector_device),
                    "detector_batch_size": int(args.detector_batch_size),
                    "min_detector_score": float(args.min_detector_score),
                    "framing_style": args.framing_style,
                    "inference_pads": [int(v) for v in args.inference_pads],
                    "smoothing_style": args.smoothing_style,
                    "smooth_window": int(args.smooth_window),
                    "detect_every": int(args.detect_every),
                    "sampled_frames": int(len(track["detections"])),
                    "detections": serialize_detection_records(
                        track["detections"],
                        frame_offset=segment_start_frame,
                    ),
                },
                f,
                ensure_ascii=False,
            )

        tier_dir = output_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        final_mp4 = tier_dir / f"{segment_name}.mp4"
        final_meta = tier_dir / f"{segment_name}.json"
        final_detections = tier_dir / f"{segment_name}.detections.json"
        os.replace(tmp_final, final_mp4)
        os.replace(tmp_meta, final_meta)
        os.replace(tmp_detections, final_detections)
    finally:
        shutil.rmtree(sample_tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    details = (
        f"{segment_name}: ok frames={crops_arr.shape[0]} "
        f"source_range={segment_start_frame}:{segment_end_exclusive - 1} "
        f"trim={absolute_trim_start}:{absolute_trim_end} "
        f"ratio={track['quality'].get('kept_ratio', 0.0):.3f} "
        f"cov={track['quality'].get('detection_coverage', 0.0):.3f} "
        f"tier={tier} ({elapsed:.1f}s)"
    )
    return _segment_result(
        name=segment_name,
        status="ok",
        message=details,
        tier=tier,
        segment_index=segment_index,
        segment_count=segment_count,
        segment_start_frame=segment_start_frame,
        segment_end_frame=segment_end_exclusive - 1,
    )


def export_one(video_path: Path, output_dir: Path, normalized_dir: Path, dataset_kind: str, args):
    base_name = video_path.stem
    normalized_path = normalized_dir / f"{base_name}.mp4"
    ok, normalize_detail = normalize_if_needed(video_path, normalized_path, args)
    if not ok:
        return [
            _segment_result(
                name=base_name,
                status="fail",
                message=f"{base_name}: {normalize_detail}",
                tier=None,
                segment_index=1,
                segment_count=1,
                segment_start_frame=0,
                segment_end_frame=-1,
            )
        ]

    source_fps, source_total_frames = probe_video(normalized_path, float(args.fps))
    segment_ranges = compute_segment_ranges(source_total_frames, int(args.max_frames))
    source_meta = load_json(video_path.with_suffix(".json"))
    segment_count = len(segment_ranges)
    results = []
    for segment_zero_idx, (segment_start, segment_end) in enumerate(segment_ranges):
        segment_index = segment_zero_idx + 1
        segment_name = build_segment_name(base_name, segment_index, segment_count)
        frames = load_frame_range(normalized_path, segment_start, segment_end)
        results.append(
            export_segment(
                video_path=video_path,
                normalized_path=normalized_path,
                output_dir=output_dir,
                dataset_kind=dataset_kind,
                source_meta=source_meta,
                normalize_detail=normalize_detail,
                source_fps=source_fps,
                source_total_frames=source_total_frames,
                frames=frames,
                segment_name=segment_name,
                segment_index=segment_index,
                segment_count=segment_count,
                segment_start_frame=segment_start,
                segment_end_exclusive=segment_end,
                args=args,
            )
        )

    # Keep normalized staging only while the current video is being processed.
    # Once every segment has reached a final non-failure state, the normalized
    # mp4 is no longer needed and just burns disk for the rest of the archive.
    if normalized_path.exists() and all(
        result["status"] in {"ok", "skip", "discard"} for result in results
    ):
        try:
            normalized_path.unlink()
        except OSError:
            pass
    return results


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
    parser.add_argument("--max-frames", type=int, default=250, help="Maximum source frames per exported segment")
    parser.add_argument("--detect-every", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--smoothing-style", choices=["legacy_centered", "official_inference", "none"], default="official_inference")
    parser.add_argument("--framing-style", choices=["legacy_square", "official_inference"], default="official_inference")
    parser.add_argument("--inference-pads", nargs=4, type=int, default=[0, 10, 0, 0], metavar=("PADY1", "PADY2", "PADX1", "PADX2"))
    parser.add_argument("--detector-backend", choices=["opencv", "sfd"], default="sfd")
    parser.add_argument("--detector-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--detector-batch-size", type=int, default=4)
    parser.add_argument("--min-detector-score", type=float, default=0.99999)
    parser.add_argument("--resize-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument("--ffmpeg-threads", type=int, default=4)
    parser.add_argument("--ffmpeg-timeout", type=int, default=180)
    parser.add_argument("--video-encoder", choices=VIDEO_ENCODER_CHOICES, default="auto")
    parser.add_argument("--normalized-video-bitrate", default="")
    parser.add_argument("--video-bitrate", default="600k")
    args = parser.parse_args()

    args.ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    args.video_encoder = select_video_encoder(args.video_encoder, args.ffmpeg_bin)
    args.normalized_video_bitrate = args.normalized_video_bitrate or args.video_bitrate

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
    log(
        f"[FaceclipExport] video_encoder={args.video_encoder} "
        f"normalized_video_bitrate={args.normalized_video_bitrate} "
        f"video_bitrate={args.video_bitrate}"
    )

    videos = list(iter_videos(input_dir))
    log(f"[FaceclipExport] videos={len(videos)}")

    total_segments = 0
    for idx, video_path in enumerate(videos, start=1):
        try:
            segment_results = export_one(video_path, output_dir, normalized_dir, dataset_kind, args)
        except Exception as exc:
            segment_results = [
                _segment_result(
                    name=video_path.stem,
                    status="fail",
                    message=f"{video_path.stem}: unexpected_fail ({type(exc).__name__}: {exc})",
                    tier=None,
                    segment_index=1,
                    segment_count=1,
                    segment_start_frame=0,
                    segment_end_frame=-1,
                )
            ]
        total_segments += len(segment_results)
        for result in segment_results:
            status = result["status"]
            tier = result.get("tier")
            counters[status] += 1
            if status == "ok" and tier in ("confident", "medium", "unconfident"):
                counters[tier] += 1
            log(
                f"[FaceclipExport] [{idx}/{len(videos)}]"
                f"[{result['segment_index']}/{result['segment_count']}] {result['message']}"
            )
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "index": idx,
                    "total": len(videos),
                    "segment_index": result["segment_index"],
                    "segment_count": result["segment_count"],
                    "name": result["name"],
                    "status": status,
                    "tier": tier,
                    "message": result["message"],
                    "source_video": video_path.name,
                    "source_segment_start_frame": result["source_segment_start_frame"],
                    "source_segment_end_frame": result["source_segment_end_frame"],
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
        "normalized_video_bitrate": args.normalized_video_bitrate,
        "video_bitrate": args.video_bitrate,
        "detector_backend": args.detector_backend,
        "detector_device": resolve_detector_device(args.detector_backend, args.detector_device),
        "resize_device": effective_resize_device,
        "total_videos": len(videos),
        "total_segments": total_segments,
        **counters,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[FaceclipExport] summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
