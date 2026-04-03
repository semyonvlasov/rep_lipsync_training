#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Copy 25fps clips as-is and re-encode non-25fps clips to 25fps")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--video-codec", default="libopenh264")
    parser.add_argument("--video-bitrate", default="12000k")
    parser.add_argument("--manifest-json", default="")
    return parser.parse_args()


def iter_videos(root: Path):
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def probe_fps(path: Path, ffprobe_bin: str) -> float:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True).strip()
    if not raw or raw == "0/0":
        return 0.0
    if "/" in raw:
        num, den = raw.split("/", 1)
        return float(num) / float(den)
    return float(raw)


def normalize_one(src: Path, dst: Path, args):
    fps = probe_fps(src, args.ffprobe_bin)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if abs(fps - 25.0) < 1.0e-3:
        shutil.copy2(src, dst)
        return {"source": str(src), "output": str(dst), "source_fps": fps, "action": "copied"}

    cmd = [
        args.ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-map",
        "0",
        "-c:v",
        args.video_codec,
        "-b:v",
        args.video_bitrate,
        "-r",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    out_fps = probe_fps(dst, args.ffprobe_bin)
    return {
        "source": str(src),
        "output": str(dst),
        "source_fps": fps,
        "output_fps": out_fps,
        "action": "reencoded",
        "video_codec": args.video_codec,
        "video_bitrate": args.video_bitrate,
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in iter_videos(input_dir):
        dst = output_dir / src.name
        row = normalize_one(src, dst, args)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.manifest_json:
        manifest_path = Path(args.manifest_json)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"videos": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
