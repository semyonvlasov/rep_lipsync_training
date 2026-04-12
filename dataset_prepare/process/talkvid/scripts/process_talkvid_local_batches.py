#!/usr/bin/env python3
"""Service launcher for TalkVid local-batch processing over shared process defaults."""

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
    get_bool,
    get_int,
    get_str,
    load_stage_config,
    resolve_repo_path,
    run_command,
)


EXPECTED_STAGE = "process_talkvid_local_batches"


def parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(default_config))
    return parser.parse_args()


def main() -> int:
    paths = discover_stage_paths(__file__)
    args = parse_args(paths.stage_root / "configs" / "default_local_batches.yaml")

    try:
        _, config = load_stage_config(args.config, EXPECTED_STAGE)

        python_bin = get_str(config, "runtime", "python_bin")

        gdrive_remote = get_str(config, "gdrive", "remote")
        processed_folder_id = get_str(config, "gdrive", "processed", "folder_id")

        batches_dir = resolve_repo_path(paths.repo_root, get_str(config, "paths", "batches_folder"))
        data_root = resolve_repo_path(
            paths.repo_root, get_str(config, "paths", "processing_folder")
        )
        manifest_path = resolve_repo_path(paths.repo_root, get_str(config, "paths", "manifest_path"))
        producer_done_flag = resolve_repo_path(
            paths.repo_root,
            get_str(config, "paths", "producer_done_flag", allow_empty=True),
        )
        assert batches_dir is not None
        assert data_root is not None
        assert manifest_path is not None
        data_root.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_bin,
            str(paths.script_dir / "process_local_talkvid_batches.py"),
            "--batches-dir",
            str(batches_dir),
            "--data-root",
            str(data_root),
            "--manifest-path",
            str(manifest_path),
            "--dest-folder-id",
            processed_folder_id,
            "--remote",
            gdrive_remote,
            "--python-bin",
            python_bin,
            "--dataset-kind",
            get_str(config, "source", "dataset_kind"),
            "--source-archive-prefix",
            get_str(config, "source", "source_archive_prefix"),
            "--processed-archive-prefix",
            get_str(config, "source", "processed_archive_prefix"),
            "--batch-glob",
            get_str(config, "source", "batch_glob"),
            "--complete-flag-name",
            get_str(config, "source", "complete_flag_name"),
            "--max-batches",
            str(get_int(config, "local_batches", "max_batches")),
            "--process-config",
            str(resolve_repo_path(paths.repo_root, args.config)),
        ]

        if get_bool(config, "local_batches", "follow"):
            cmd.extend(
                ["--follow", "--poll-seconds", str(get_int(config, "local_batches", "poll_seconds"))]
            )

        producer_done_flag_str = get_str(
            config, "paths", "producer_done_flag", allow_empty=True
        )
        if producer_done_flag_str:
            assert producer_done_flag is not None
            cmd.extend(["--producer-done-flag", str(producer_done_flag)])

        run_command(cmd)
        return 0
    except ConfigError as exc:
        return exit_with_config_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
