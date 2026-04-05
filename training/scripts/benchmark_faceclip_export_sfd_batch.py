#!/usr/bin/env python3
"""
Benchmark end-to-end x288 faceclip export throughput for different SFD detector
batch sizes on the same fixed subset of source videos.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def parse_batch_sizes(raw: str) -> list[int]:
    seen = []
    for chunk in raw.split(","):
        value = int(chunk.strip())
        if value <= 0:
            raise ValueError(f"batch size must be positive: {value}")
        if value not in seen:
            seen.append(value)
    return seen


def probe_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"video_open_failed: {path}")
    try:
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if frame_count > 0:
            return frame_count
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        return count
    finally:
        cap.release()


def select_subset(input_dir: Path, subset_dir: Path, max_videos: int) -> list[Path]:
    videos = sorted(path for path in input_dir.glob("*.mp4") if path.is_file())
    if max_videos > 0:
        videos = videos[:max_videos]
    if not videos:
        raise RuntimeError(f"no_videos_found: {input_dir}")

    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    for video in videos:
        target = subset_dir / video.name
        target.symlink_to(video)
        json_sidecar = video.with_suffix(".json")
        if json_sidecar.exists():
            (subset_dir / json_sidecar.name).symlink_to(json_sidecar)
    return videos


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_md(path: Path, payload: dict) -> None:
    rows = payload["results"]
    lines = [
        f"# {payload['title']}",
        "",
        f"- input_dir: `{payload['input_dir']}`",
        f"- subset_dir: `{payload['subset_dir']}`",
        f"- subset_videos: `{payload['subset_videos']}`",
        f"- subset_source_frames: `{payload['subset_source_frames']}`",
        f"- size: `{payload['size']}`",
        f"- fps: `{payload['fps']}`",
        f"- max_frames: `{payload['max_frames']}`",
        f"- detector: `{payload['detector_backend']}/{payload['detector_device']}`",
        f"- framing: `{payload['framing_style']}`",
        f"- smoothing: `{payload['smoothing_style']}`",
        "",
        "| batch | wall_s | frames_per_s | segments_per_s | ok | discard | fail | total_segments | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['detector_batch_size']} | {row['wall_seconds']:.3f} | "
            f"{row['frames_per_second']:.2f} | {row['segments_per_second']:.3f} | "
            f"{row.get('ok', 0)} | {row.get('discard', 0)} | {row.get('fail', 0)} | "
            f"{row.get('total_segments', 0)} | {row.get('notes', '')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--batch-sizes", default="1,2,4,8,12,16,24,32")
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument("--dataset-kind", default="hdtf", choices=["auto", "hdtf", "talkvid"])
    parser.add_argument("--size", type=int, default=288)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument(
        "--smoothing-style",
        choices=["legacy_centered", "official_inference", "none"],
        default="official_inference",
    )
    parser.add_argument(
        "--framing-style",
        choices=["legacy_square", "official_inference"],
        default="official_inference",
    )
    parser.add_argument("--detector-backend", choices=["opencv", "sfd"], default="sfd")
    parser.add_argument("--detector-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--min-detector-score", type=float, default=0.0)
    parser.add_argument("--resize-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--ffmpeg-bin", default="")
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--ffmpeg-timeout", type=int, default=180)
    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--normalized-video-bitrate", default="15m")
    parser.add_argument("--video-bitrate", default="420k")
    parser.add_argument("--input-is-normalized", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    work_root = Path(args.work_root).resolve()
    subset_dir = work_root / "subset"
    runs_dir = work_root / "runs"
    results_path = work_root / "results.json"
    markdown_path = work_root / "results.md"
    export_script = Path(__file__).with_name("export_faceclip_batch.py").resolve()

    if work_root.exists():
        shutil.rmtree(work_root)
    runs_dir.mkdir(parents=True, exist_ok=True)

    videos = select_subset(input_dir, subset_dir, args.max_videos)
    total_source_frames = sum(probe_frame_count(path) for path in videos)
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    results = []
    for detector_batch_size in batch_sizes:
        run_root = runs_dir / f"bs_{detector_batch_size:02d}"
        output_dir = run_root / "output"
        normalized_dir = run_root / "normalized"
        log_path = run_root / "run.log"
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python_bin,
            str(export_script),
            "--input-dir",
            str(subset_dir),
            "--output-dir",
            str(output_dir),
            "--normalized-dir",
            str(normalized_dir),
            "--source-archive",
            f"bench_subset_{input_dir.name}.tar",
            "--dataset-kind",
            args.dataset_kind,
            "--size",
            str(args.size),
            "--fps",
            str(args.fps),
            "--max-frames",
            str(args.max_frames),
            "--detect-every",
            str(args.detect_every),
            "--smooth-window",
            str(args.smooth_window),
            "--smoothing-style",
            args.smoothing_style,
            "--framing-style",
            args.framing_style,
            "--detector-backend",
            args.detector_backend,
            "--detector-device",
            args.detector_device,
            "--detector-batch-size",
            str(detector_batch_size),
            "--min-detector-score",
            str(args.min_detector_score),
            "--resize-device",
            args.resize_device,
            "--ffmpeg-bin",
            args.ffmpeg_bin,
            "--ffmpeg-threads",
            str(args.ffmpeg_threads),
            "--ffmpeg-timeout",
            str(args.ffmpeg_timeout),
            "--video-encoder",
            args.video_encoder,
            "--normalized-video-bitrate",
            args.normalized_video_bitrate,
            "--video-bitrate",
            args.video_bitrate,
        ]
        if args.input_is_normalized:
            cmd.append("--input-is-normalized")

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall_seconds = time.perf_counter() - t0
        log_path.write_text((proc.stdout or "") + ("\n" if proc.stdout else "") + (proc.stderr or ""))

        summary = load_summary(output_dir / "summary.json")
        total_segments = int(summary.get("total_segments", 0))
        frames_per_second = (float(total_source_frames) / wall_seconds) if wall_seconds > 0 else 0.0
        segments_per_second = (float(total_segments) / wall_seconds) if wall_seconds > 0 else 0.0

        notes = ""
        if proc.returncode != 0:
            notes = f"rc={proc.returncode}"

        row = {
            "detector_batch_size": detector_batch_size,
            "wall_seconds": wall_seconds,
            "frames_per_second": frames_per_second,
            "segments_per_second": segments_per_second,
            "returncode": int(proc.returncode),
            "notes": notes,
            **summary,
        }
        results.append(row)
        log(
            f"[SFDExportBench] batch={detector_batch_size} wall={wall_seconds:.2f}s "
            f"frames/s={frames_per_second:.2f} segments/s={segments_per_second:.3f} "
            f"rc={proc.returncode}"
        )

    results.sort(key=lambda row: (row["returncode"] != 0, row["wall_seconds"]))
    payload = {
        "title": "SFD x288 Export Batch Benchmark",
        "ts": timestamp(),
        "input_dir": str(input_dir),
        "subset_dir": str(subset_dir),
        "subset_videos": len(videos),
        "subset_source_frames": int(total_source_frames),
        "batch_sizes": batch_sizes,
        "size": args.size,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "detector_backend": args.detector_backend,
        "detector_device": args.detector_device,
        "framing_style": args.framing_style,
        "smoothing_style": args.smoothing_style,
        "results": results,
    }
    write_json(results_path, payload)
    write_md(markdown_path, payload)
    log(f"[SFDExportBench] wrote {results_path}")
    log(f"[SFDExportBench] wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
