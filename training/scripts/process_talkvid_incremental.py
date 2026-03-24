#!/usr/bin/env python3
"""
Incrementally normalize TalkVid raw clips into 25fps/16kHz clips and processed samples.

Pipeline:
    raw/<clip>.mp4 -> clips_25fps/<clip>.mp4 -> processed/<clip>/{frames,mel,bbox}

This worker is dataset-agnostic and works best when pointed at a sealed TalkVid
batch, for example:

    --raw-dir training/data/talkvid_local/batches/batch_0006/raw
    --clips-dir training/data/talkvid_local/batches/batch_0006/clips_25fps
    --processed-dir training/data/talkvid_local/processed

When experimenting with multiple pose-quality presets, use separate
`--processed-dir` values per preset so the generated datasets stay comparable.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.process_video_incremental_common import (
    load_failed_names,
    list_raw_videos,
    print_startup_banner,
    processed_exists,
    run_candidates,
)
from scripts.talkvid_head_filter import (
    HEAD_FILTER_MODE_CHOICES,
    describe_head_filter_mode,
    evaluate_head_filter_raw_clip,
)
from scripts.transcode_video import VIDEO_ENCODER_CHOICES, resolve_ffmpeg_bin, select_video_encoder


def collect_talkvid_candidates(args):
    raw_items = list_raw_videos(args.raw_dir)
    candidates = []
    filtered_head = 0
    failed_skipped = 0
    failed_names = set() if args.retry_failed else load_failed_names(args.processed_dir)

    for _, fname in raw_items:
        if not fname.endswith(".mp4"):
            continue
        name = os.path.splitext(fname)[0]
        if processed_exists(args.processed_dir, name):
            continue
        if name in failed_names:
            failed_skipped += 1
            continue

        raw_path = os.path.join(args.raw_dir, fname)
        ok, _ = evaluate_head_filter_raw_clip(raw_path, args.head_filter_mode)
        if not ok:
            filtered_head += 1
            continue
        candidates.append(raw_path)

    return candidates, filtered_head, failed_skipped


def run_talkvid_incremental_loop(args):
    idle_cycles = 0
    while True:
        candidates, filtered_head, failed_skipped = collect_talkvid_candidates(args)

        if not candidates:
            idle_cycles += 1
            if args.head_filter_mode == "off":
                detail = f", failed_skipped={failed_skipped}"
            else:
                detail = f", head_filtered={filtered_head}, failed_skipped={failed_skipped}"
            print(
                f"[TalkVidProcess] No pending raw videos "
                f"(idle_cycle={idle_cycles}, follow={args.follow}{detail})",
                flush=True,
            )
            if not args.follow:
                break
            if args.idle_exit_cycles > 0 and idle_cycles >= args.idle_exit_cycles:
                print("[TalkVidProcess] Idle exit threshold reached, stopping", flush=True)
                break
            time.sleep(max(args.poll_seconds, 1))
            continue

        idle_cycles = 0
        if args.head_filter_mode == "off":
            print(
                f"[TalkVidProcess] Pending raw videos: {len(candidates)} "
                f"(failed_skipped={failed_skipped})",
                flush=True,
            )
        else:
            print(
                f"[TalkVidProcess] Pending raw videos: {len(candidates)} "
                f"(head_filtered={filtered_head}, failed_skipped={failed_skipped}, mode={args.head_filter_mode})",
                flush=True,
            )
        run_candidates(candidates, args, "TalkVidProcess")

        if not args.follow:
            break

    print("[TalkVidProcess] Done", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Directory with TalkVid raw mp4 clips")
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
        "--head-filter-mode",
        choices=HEAD_FILTER_MODE_CHOICES,
        default="off",
        help="TalkVid head-pose preset for dataset generation",
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
    print_startup_banner(args, log_prefix="TalkVidProcess")
    print(f"[TalkVidProcess] head_filter_mode={describe_head_filter_mode(args.head_filter_mode)}")
    run_talkvid_incremental_loop(args)


if __name__ == "__main__":
    main()
