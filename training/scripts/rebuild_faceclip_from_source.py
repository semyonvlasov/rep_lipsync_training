#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild a faceclip video directly from source frames and detections.json"
    )
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--detections-json", required=True)
    parser.add_argument("--trimmed-meta-json", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--video-codec", default="mpeg4")
    parser.add_argument("--audio-codec", default="aac")
    return parser.parse_args()


def load_json(path: str):
    return json.loads(Path(path).read_text())


def crop_and_resize(frame, bbox, size: int):
    y1, y2, x1, x2 = [int(round(v)) for v in bbox]
    y1 = max(0, min(y1, frame.shape[0] - 1))
    y2 = max(y1 + 1, min(y2, frame.shape[0]))
    x1 = max(0, min(x1, frame.shape[1] - 1))
    x2 = max(x1 + 1, min(x2, frame.shape[1]))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def main():
    args = parse_args()
    detections = load_json(args.detections_json)["detections"]
    trimmed_meta = load_json(args.trimmed_meta_json)
    frame_indices = trimmed_meta["trimmed_frame_indices"]

    det_by_frame = {int(row["frame_idx"]): row for row in detections}

    cap = cv2.VideoCapture(args.source_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {args.source_video}")

    with tempfile.TemporaryDirectory(prefix="rebuild_faceclip_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        raw_video = tmpdir_path / "video_only.mp4"
        writer = cv2.VideoWriter(
            str(raw_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.size, args.size),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer for {raw_video}")

        wanted = set(frame_indices)
        next_target_iter = iter(frame_indices)
        current_target = next(next_target_iter, None)
        frame_idx = 0
        wrote = 0

        while current_target is not None:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx == current_target:
                row = det_by_frame.get(frame_idx)
                if not row or row.get("bbox") is None:
                    raise RuntimeError(f"Missing bbox for frame {frame_idx}")
                writer.write(crop_and_resize(frame, row["bbox"], args.size))
                wrote += 1
                current_target = next(next_target_iter, None)
            frame_idx += 1

        writer.release()
        cap.release()

        if wrote != len(frame_indices):
            raise RuntimeError(
                f"Wrote {wrote} frames, expected {len(frame_indices)} from source {args.source_video}"
            )

        start_sec = frame_indices[0] / float(args.fps)
        duration_sec = len(frame_indices) / float(args.fps)
        output_path = Path(args.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.ffmpeg_bin,
            "-y",
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration_sec:.6f}",
            "-i",
            args.source_video,
            "-i",
            str(raw_video),
            "-map",
            "1:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            args.video_codec,
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{args.fps:.6f}",
            "-c:a",
            args.audio_codec,
            "-shortest",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        print(
            json.dumps(
                {
                    "output_video": str(output_path),
                    "frames_written": wrote,
                    "trim_start_frame": frame_indices[0],
                    "trim_end_frame": frame_indices[-1],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
