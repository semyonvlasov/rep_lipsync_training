#!/usr/bin/env python3
"""Fetch raw HDTF videos, package raw tar batches, and upload them to Drive."""

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
    log,
    open_stage_log,
    resolve_repo_path,
    run_command,
)


EXPECTED_STAGE = "fetch_hdtf_public_to_gdrive"


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
        gdrive_remote = get_str(config, "gdrive", "remote")
        raw_folder_id = get_str(config, "gdrive", "raw", "folder_id")

        workspace_root = resolve_repo_path(
            paths.repo_root, get_str(config, "paths", "workspace_root")
        )
        log_folder = resolve_repo_path(paths.repo_root, get_str(config, "paths", "log_folder"))
        assert workspace_root is not None

        data_root = workspace_root
        archives_dir = workspace_root / "archives_raw_batches"

        data_root.mkdir(parents=True, exist_ok=True)
        archives_dir.mkdir(parents=True, exist_ok=True)

        log_fp = open_stage_log(log_folder, "stage.log")
        try:
            log("[hdtf-fetch] start", log_fp=log_fp)
            run_command(
                [
                    python_bin,
                    str(paths.script_dir / "download_hdtf.py"),
                    "--output",
                    str(data_root),
                    "--max-videos",
                    str(get_int(config, "fetch", "max_videos")),
                    "--target-size-gb",
                    str(get_float(config, "fetch", "target_size_gb")),
                    "--max-height",
                    str(get_int(config, "fetch", "download_max_height")),
                    "--timeout",
                    str(get_int(config, "fetch", "download_timeout")),
                ],
                log_fp=log_fp,
            )
            run_command(
                [
                    python_bin,
                    str(paths.script_dir / "package_media_batches.py"),
                    "--input-dir",
                    str(data_root / "raw"),
                    "--archives-dir",
                    str(archives_dir),
                    "--prefix",
                    "hdtf_raw",
                    "--pattern",
                    "*.mp4",
                    "--max-gb",
                    str(get_float(config, "fetch", "batch_size_gb")),
                    "--max-batches",
                    str(get_int(config, "fetch", "max_batches")),
                    "--allow-partial-tail",
                ],
                log_fp=log_fp,
            )
            run_command(
                [
                    python_bin,
                    str(paths.script_dir / "upload_archives_no_cleanup.py"),
                    "--archives-dir",
                    str(archives_dir),
                    "--remote",
                    gdrive_remote,
                    "--drive-root-folder-id",
                    raw_folder_id,
                    "--prefix",
                    "hdtf_raw",
                    "--max-batches",
                    "0",
                ],
                log_fp=log_fp,
            )
            log("[hdtf-fetch] done", log_fp=log_fp)
            return 0
        finally:
            if log_fp is not None:
                log_fp.close()
    except ConfigError as exc:
        return exit_with_config_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
