#!/usr/bin/env python3
"""
Incrementally normalize HDTF raw videos into 25fps/16kHz clips and processed samples.

Pipeline:
    raw/<name>.mp4 -> clips/<name>.mp4 -> processed/<name>/{frames,mel,bbox}
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.process_video_incremental_common import print_startup_banner, run_incremental_loop
from scripts.transcode_video import VIDEO_ENCODER_CHOICES, resolve_ffmpeg_bin, select_video_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Directory with raw downloaded mp4 files")
    parser.add_argument("--clips-dir", required=True, help="Directory for normalized 25fps clips")
    parser.add_argument("--processed-dir", required=True, help="Directory for processed training samples")
    parser.add_argument("--size", type=int, default=256, help="Face crop size")
    parser.add_argument("--fps", type=int, default=25, help="Force output clip FPS")
    parser.add_argument("--max-frames", type=int, default=750, help="Max frames per video")
    parser.add_argument("--detect-every", type=int, default=10, help="Run face detection every N frames")
    parser.add_argument("--smooth-window", type=int, default=9, help="Temporal smoothing window for bbox track")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers over videos")
    parser.add_argument(
        "--worker-backend",
        choices=["auto", "process", "thread"],
        default="auto",
        help="Parallel backend for multi-worker mode",
    )
    parser.add_argument("--no-preview", action="store_true", help="Do not save preview contact sheets")
    parser.add_argument(
        "--detector-backend",
        choices=["opencv", "sfd"],
        default="opencv",
        help="Face detector backend",
    )
    parser.add_argument(
        "--detector-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for detector backend",
    )
    parser.add_argument(
        "--detector-batch-size",
        type=int,
        default=4,
        help="Batch size for SFD face detection on sampled frames",
    )
    parser.add_argument(
        "--resize-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Device for face crop resize stage",
    )
    parser.add_argument("--follow", action="store_true", help="Keep rescanning raw dir for new files")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Sleep between rescans in follow mode")
    parser.add_argument(
        "--idle-exit-cycles",
        type=int,
        default=0,
        help="Exit after this many empty follow cycles (0=never exit in follow mode)",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default=None,
        help="Path to ffmpeg binary; defaults to system ffmpeg when available",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=1,
        help="Threads per ffmpeg invocation (safer with parallel workers)",
    )
    parser.add_argument(
        "--ffmpeg-timeout",
        type=int,
        default=120,
        help="Timeout per ffmpeg normalization call in seconds",
    )
    parser.add_argument(
        "--video-encoder",
        choices=VIDEO_ENCODER_CHOICES,
        default="auto",
        help="Video encoder for the normalization stage",
    )
    parser.add_argument(
        "--video-bitrate",
        default="2200k",
        help="Target video bitrate for hardware encoders",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry clips previously recorded in processed/_failed_samples.jsonl",
    )
    args = parser.parse_args()

    args.ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    args.video_encoder = select_video_encoder(args.video_encoder, args.ffmpeg_bin)
    os.makedirs(args.clips_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    print_startup_banner(args, log_prefix="Incremental")
    run_incremental_loop(args, log_prefix="Incremental")


if __name__ == "__main__":
    main()
