#!/usr/bin/env python3
"""Service launcher for HDTF raw-Drive processing over shared process defaults."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prepare.common.config import (
    ConfigError,
    discover_stage_paths,
    exit_with_config_error,
    get_float,
    get_int,
    get_str,
    load_stage_config,
    resolve_repo_path,
    run_command,
)


EXPECTED_STAGE = "process_hdtf_raw_from_gdrive"


def parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(default_config))
    return parser.parse_args()


def main() -> int:
    paths = discover_stage_paths(__file__)
    args = parse_args(paths.stage_root / "configs" / "default.yaml")

    try:
        _, config = load_stage_config(args.config, EXPECTED_STAGE)

        python_bin = get_str(config, "runtime", "python_bin")
        ffmpeg_bin = get_str(config, "runtime", "ffmpeg_bin", allow_empty=True)
        ffmpeg_threads = get_int(config, "runtime", "ffmpeg_threads")
        ffmpeg_timeout = get_int(config, "runtime", "ffmpeg_timeout")
        video_encoder = get_str(config, "runtime", "video_encoder")

        gdrive_remote = get_str(config, "gdrive", "remote")
        raw_folder_id = get_str(config, "gdrive", "raw", "folder_id")
        processed_folder_id = get_str(config, "gdrive", "processed", "folder_id")

        data_root = resolve_repo_path(
            paths.repo_root, get_str(config, "paths", "processing_folder")
        )
        manifest_path = resolve_repo_path(paths.repo_root, get_str(config, "paths", "manifest_path"))
        assert data_root is not None
        assert manifest_path is not None
        data_root.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_bin,
            str(paths.script_dir / "process_raw_archives_to_lazy_faceclips_gdrive.py"),
            "--source-folder-id",
            raw_folder_id,
            "--dest-folder-id",
            processed_folder_id,
            "--remote",
            gdrive_remote,
            "--data-root",
            str(data_root),
            "--manifest-path",
            str(manifest_path),
            "--archive-glob",
            get_str(config, "source", "archive_glob"),
            "--max-archives",
            str(get_int(config, "process", "max_archives")),
            "--python-bin",
            python_bin,
            "--size",
            str(get_int(config, "process", "faceclip_size")),
            "--fps",
            str(get_int(config, "process", "processed_fps")),
            "--max-frames",
            str(get_int(config, "process", "max_frames")),
            "--detect-every",
            str(get_int(config, "process", "detect_every")),
            "--smooth-window",
            str(get_int(config, "process", "smooth_window")),
            "--smoothing-style",
            get_str(config, "process", "smoothing_style"),
            "--framing-style",
            get_str(config, "process", "framing_style"),
            "--detector-backend",
            get_str(config, "process", "detector_backend"),
            "--detector-device",
            get_str(config, "process", "detector_device"),
            "--detector-batch-size",
            str(get_int(config, "process", "detector_batch_size")),
            "--min-detector-score",
            str(get_float(config, "process", "min_detector_score")),
            "--resize-device",
            get_str(config, "process", "resize_device"),
            "--ffmpeg-bin",
            ffmpeg_bin,
            "--ffmpeg-threads",
            str(ffmpeg_threads),
            "--ffmpeg-timeout",
            str(ffmpeg_timeout),
            "--video-encoder",
            video_encoder,
            "--normalized-video-bitrate",
            get_str(config, "process", "normalized_video_bitrate"),
            "--video-bitrate",
            get_str(config, "process", "processed_video_bitrate"),
        ]
        run_command(cmd)
        return 0
    except ConfigError as exc:
        return exit_with_config_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
