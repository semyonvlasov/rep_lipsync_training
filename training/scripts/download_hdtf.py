#!/usr/bin/env python3
"""
Download HDTF dataset videos from YouTube.

HDTF annotations: name + YouTube URL per line in RD/WDA/WRA_video_url.txt
This script downloads them via yt-dlp, optionally transcodes to standard format.

Usage:
    python scripts/download_hdtf.py --output data/hdtf --max-videos 20
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.transcode_video import (
    VIDEO_ENCODER_CHOICES,
    normalize_video_clip,
    resolve_ffmpeg_bin,
    select_video_encoder,
)

# HDTF annotation files on GitHub
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
    """Download and parse HDTF video URL annotations."""
    annotations = {}  # name → youtube_url

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
                    name = parts[0]
                    yt_url = parts[1]
                    annotations[name] = yt_url

    return annotations


def download_video(youtube_url, output_path, max_height=720):
    """Download a single video via yt-dlp."""
    if os.path.exists(output_path):
        return True
    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]",
                "--merge-output-format", "mp4",
                "-o", output_path,
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
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  Failed: {e}")
        return False


def extract_segment(input_path, output_path, start, end, fps, ffmpeg_bin=None, ffmpeg_threads=0):
    """Transcode video to standard format, forcing CFR output + 16kHz mono audio."""
    ok, _, _ = normalize_video_clip(
        input_path,
        output_path,
        fps=fps,
        start_time=start,
        duration=end - start,
        ffmpeg_bin=ffmpeg_bin,
        ffmpeg_threads=ffmpeg_threads,
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download HDTF dataset")
    parser.add_argument("--output", default="data/hdtf", help="Output directory")
    parser.add_argument("--max-videos", type=int, default=0, help="Max videos (0=all)")
    parser.add_argument("--max-height", type=int, default=720, help="Max video height")
    parser.add_argument("--fps", type=int, default=25, help="Force output clip FPS")
    parser.add_argument(
        "--ffmpeg-bin",
        default=None,
        help="Path to ffmpeg binary for transcoding; defaults to system ffmpeg when available",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=1,
        help="Threads per ffmpeg invocation",
    )
    parser.add_argument(
        "--video-encoder",
        choices=VIDEO_ENCODER_CHOICES,
        default="auto",
        help="Video encoder for clip normalization",
    )
    parser.add_argument(
        "--video-bitrate",
        default="2200k",
        help="Target video bitrate for hardware encoders",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download raw videos to raw/, skip ffmpeg transcoding",
    )
    parser.add_argument(
        "--delay-seconds",
        type=int,
        default=0,
        help="Delay execution before starting downloads",
    )
    parser.add_argument(
        "--target-size-gb",
        type=float,
        default=0.0,
        help="Keep downloading until the active target dir reaches this size in GB (0=disabled)",
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
    clips_dir = os.path.join(args.output, "clips")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    # Check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install: pip install yt-dlp")
        sys.exit(1)

    annotations = download_annotations(args.output)
    print(f"[HDTF] {len(annotations)} videos in annotations")
    if not args.download_only:
        args.video_encoder = select_video_encoder(args.video_encoder, args.ffmpeg_bin)
        print(f"[HDTF] Using ffmpeg: {resolve_ffmpeg_bin(args.ffmpeg_bin)}")
        print(f"[HDTF] video_encoder={args.video_encoder}")
        print(f"[HDTF] video_bitrate={args.video_bitrate}")

    items = list(annotations.items())
    if args.max_videos > 0:
        items = items[:args.max_videos]
        print(f"[HDTF] Limited to {len(items)} videos")

    target_bytes = int(args.target_size_gb * (1024 ** 3)) if args.target_size_gb > 0 else 0
    target_dir = raw_dir if args.download_only else clips_dir
    target_label = "raw" if args.download_only else "clips"
    if target_bytes:
        current_bytes = get_dir_size_bytes(target_dir)
        print(f"[HDTF] Target {target_label} size: {args.target_size_gb:.2f} GB")
        print(f"[HDTF] Current {target_label} size: {current_bytes / (1024 ** 3):.2f} GB")

    success = 0
    failed = 0

    for name, url in items:
        if target_bytes:
            current_bytes = get_dir_size_bytes(target_dir)
            if current_bytes >= target_bytes:
                print(f"[HDTF] Reached target {target_label} size: {current_bytes / (1024 ** 3):.2f} GB")
                break

        raw_path = os.path.join(raw_dir, f"{name}.mp4")
        clip_path = os.path.join(clips_dir, f"{name}.mp4")

        if args.download_only and os.path.exists(raw_path):
            success += 1
            continue

        if not args.download_only and os.path.exists(clip_path):
            success += 1
            continue

        print(f"[{success + failed + 1}/{len(items)}] {name}...", end=" ", flush=True)

        if download_video(url, raw_path, args.max_height):
            if args.download_only:
                success += 1
                print("OK (download)")
            elif normalize_video_clip(
                raw_path,
                clip_path,
                fps=args.fps,
                start_time=0,
                duration=999,
                ffmpeg_bin=args.ffmpeg_bin,
                ffmpeg_threads=args.ffmpeg_threads,
                video_encoder=args.video_encoder,
                video_bitrate=args.video_bitrate,
            )[0]:
                success += 1
                print("OK")
                # Remove raw to save space
                if os.path.exists(raw_path):
                    os.remove(raw_path)
            else:
                failed += 1
                print("FAIL (ffmpeg)")
        else:
            failed += 1
            print("FAIL (download)")

    print(f"\n[HDTF] Done: {success} success, {failed} failed")
    if args.download_only:
        print(f"[HDTF] Raw videos saved to: {raw_dir}")
    else:
        print(f"[HDTF] Clips saved to: {clips_dir}")
    final_bytes = get_dir_size_bytes(target_dir)
    print(f"[HDTF] {target_label.capitalize()} size: {final_bytes / (1024 ** 3):.2f} GB")

    manifest_dir = raw_dir if args.download_only else clips_dir
    manifest = sorted(
        os.path.join(manifest_dir, f)
        for f in os.listdir(manifest_dir) if f.endswith(".mp4")
    )
    manifest_path = os.path.join(args.output, "manifest.txt")
    with open(manifest_path, "w") as mf:
        mf.write("\n".join(manifest))
    print(f"[HDTF] Manifest: {manifest_path} ({len(manifest)} clips)")


if __name__ == "__main__":
    main()
