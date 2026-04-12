from __future__ import annotations

import argparse
import logging
import sys

from face_processing.config import PipelineConfig
from face_processing.pipeline import process_video


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="face-processing",
        description="Face video preprocessing pipeline for dataset preparation.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video (.mp4, .mov, .mkv)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to JSON config file to override defaults",
    )
    parser.add_argument(
        "--save-frame-log",
        action="store_true",
        help="Save per-frame CSV log for debugging",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU (Metal) for MediaPipe inference",
    )
    parser.add_argument(
        "--keep-normalized",
        action="store_true",
        help="Keep normalized video for later restore",
    )

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig()

    config.output_dir = args.output_dir
    config.save_frame_log = args.save_frame_log
    if args.gpu:
        config.detection.use_gpu = True
    if args.keep_normalized:
        config.keep_normalized = True

    # Run pipeline
    result = process_video(args.input, config)

    # Print summary
    if result.status == "dropped":
        print(f"\nVideo DROPPED: {result.drop_reason}")
    else:
        exported = [s for s in result.segments if s.status == "exported"]
        print(f"\nVideo processed: {len(exported)} segments exported")
        for s in exported:
            print(f"  Segment {s.segment_id}: {s.length} frames, rank={s.rank}, size={s.output_size}")


if __name__ == "__main__":
    main()
