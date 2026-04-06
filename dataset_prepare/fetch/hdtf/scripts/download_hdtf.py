#!/usr/bin/env python3
"""
Download raw HDTF dataset videos from YouTube.

HDTF annotations: name + YouTube URL per line in RD/WDA/WRA_video_url.txt.
This fetch-stage script only downloads raw `.mp4` files into `raw/`.
Any normalization to training-friendly FPS/bitrate happens later in process.
"""

import argparse
import os
import subprocess
import sys
import time


HDTF_URL_FILES = [
    "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset/RD_video_url.txt",
    "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset/WDA_video_url.txt",
    "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset/WRA_video_url.txt",
]


def get_dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def download_annotations(output_dir):
    annotations = {}

    for url in HDTF_URL_FILES:
        fname = url.split("/")[-1]
        local_path = os.path.join(output_dir, fname)

        if not os.path.exists(local_path):
            print(f"[HDTF] Downloading {fname}...")
            subprocess.run(["curl", "-sL", url, "-o", local_path], check=True)

        with open(local_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    annotations[parts[0]] = parts[1]

    return annotations


def download_video(youtube_url, output_path, max_height=720):
    if os.path.exists(output_path):
        return True
    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]",
                "--merge-output-format",
                "mp4",
                "-o",
                output_path,
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                youtube_url,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  Failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download raw HDTF dataset videos")
    parser.add_argument("--output", default="data/hdtf", help="Output directory")
    parser.add_argument("--max-videos", type=int, default=0, help="Max videos (0=all)")
    parser.add_argument("--max-height", type=int, default=720, help="Max video height")
    parser.add_argument("--delay-seconds", type=int, default=0, help="Delay execution before starting downloads")
    parser.add_argument(
        "--target-size-gb",
        type=float,
        default=0.0,
        help="Keep downloading until raw/ reaches this size in GB (0=disabled)",
    )
    args = parser.parse_args()

    if args.delay_seconds > 0:
        start_ts = time.time() + args.delay_seconds
        start_str = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(start_ts))
        print(f"[HDTF] Delaying start for {args.delay_seconds}s until {start_str}", flush=True)
        time.sleep(args.delay_seconds)
        print("[HDTF] Delay finished, starting downloads", flush=True)

    os.makedirs(args.output, exist_ok=True)
    raw_dir = os.path.join(args.output, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install: pip install yt-dlp")
        sys.exit(1)

    annotations = download_annotations(args.output)
    print(f"[HDTF] {len(annotations)} videos in annotations")

    items = list(annotations.items())
    if args.max_videos > 0:
        items = items[:args.max_videos]
        print(f"[HDTF] Limited to {len(items)} videos")

    target_bytes = int(args.target_size_gb * (1024 ** 3)) if args.target_size_gb > 0 else 0
    if target_bytes:
        current_bytes = get_dir_size_bytes(raw_dir)
        print(f"[HDTF] Target raw size: {args.target_size_gb:.2f} GB")
        print(f"[HDTF] Current raw size: {current_bytes / (1024 ** 3):.2f} GB")

    success = 0
    failed = 0

    for name, url in items:
        if target_bytes:
            current_bytes = get_dir_size_bytes(raw_dir)
            if current_bytes >= target_bytes:
                print(f"[HDTF] Reached target raw size: {current_bytes / (1024 ** 3):.2f} GB")
                break

        raw_path = os.path.join(raw_dir, f"{name}.mp4")
        if os.path.exists(raw_path):
            success += 1
            continue

        print(f"[{success + failed + 1}/{len(items)}] {name}...", end=" ", flush=True)
        if download_video(url, raw_path, args.max_height):
            success += 1
            print("OK (download)")
        else:
            failed += 1
            print("FAIL (download)")

    print(f"\n[HDTF] Done: {success} success, {failed} failed")
    print(f"[HDTF] Raw videos saved to: {raw_dir}")
    final_bytes = get_dir_size_bytes(raw_dir)
    print(f"[HDTF] Raw size: {final_bytes / (1024 ** 3):.2f} GB")

    manifest = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".mp4"))
    manifest_path = os.path.join(args.output, "manifest.txt")
    with open(manifest_path, "w") as mf:
        mf.write("\n".join(manifest))
    print(f"[HDTF] Manifest: {manifest_path} ({len(manifest)} videos)")


if __name__ == "__main__":
    main()
