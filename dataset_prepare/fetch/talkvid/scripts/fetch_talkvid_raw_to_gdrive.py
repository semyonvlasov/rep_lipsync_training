#!/usr/bin/env python3
"""Fetch raw TalkVid clips, package raw tar batches, and upload them to Drive.

Supports CLI-only batch numbering overrides via `--resume-batch` and
`--start-from-batch`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
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
    format_cmd,
    get_float,
    get_int,
    get_str,
    load_stage_config,
    log,
    open_stage_log,
    resolve_repo_path,
)


EXPECTED_STAGE = "fetch_talkvid_raw_to_gdrive"


def parse_batch_name(raw_value: str) -> str:
    try:
        batch_idx = int(raw_value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"batch id must be an integer like 0034, got: {raw_value!r}"
        ) from exc
    if batch_idx < 0:
        raise argparse.ArgumentTypeError("batch id must be non-negative")
    return f"{batch_idx:04d}"


def parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(default_config))
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--resume-batch",
        type=parse_batch_name,
        help="Continue filling an existing batch directory like batch_0034.",
    )
    batch_group.add_argument(
        "--start-from-batch",
        type=parse_batch_name,
        help="Start a new numbering sequence from a specific batch id like 0035.",
    )
    return parser.parse_args()


def read_next_batch_index(counter_file: Path) -> int:
    if not counter_file.exists():
        return 0
    text = counter_file.read_text(encoding="utf-8").strip()
    return int(text) if text.isdigit() else 0


def write_next_batch_index(counter_file: Path, next_idx: int) -> None:
    counter_file.write_text(f"{next_idx}\n", encoding="utf-8")


def next_batch_name(counter_file: Path) -> str:
    next_idx = read_next_batch_index(counter_file)
    write_next_batch_index(counter_file, next_idx + 1)
    return f"{next_idx:04d}"


def choose_batch_root(
    *,
    batch_runs_dir: Path,
    batch_counter_file: Path,
    resume_batch: str | None,
    start_from_batch: str | None,
    first_iteration: bool,
    log_fp: object | None,
) -> tuple[str, Path]:
    if first_iteration and resume_batch is not None:
        batch_root = batch_runs_dir / f"batch_{resume_batch}"
        if not batch_root.exists():
            raise RuntimeError(
                f"--resume-batch={resume_batch} requested but {batch_root} does not exist"
            )
        resume_next_idx = int(resume_batch) + 1
        current_next_idx = read_next_batch_index(batch_counter_file)
        if resume_next_idx > current_next_idx:
            write_next_batch_index(batch_counter_file, resume_next_idx)
        log(f"[cycle] resuming existing batch=batch_{resume_batch}", log_fp=log_fp)
        return resume_batch, batch_root

    if first_iteration and start_from_batch is not None:
        batch_root = batch_runs_dir / f"batch_{start_from_batch}"
        if batch_root.exists():
            raise RuntimeError(
                f"--start-from-batch={start_from_batch} conflicts with existing {batch_root}; "
                f"use --resume-batch {start_from_batch} instead"
            )
        write_next_batch_index(batch_counter_file, int(start_from_batch) + 1)
        log(f"[cycle] starting new batch numbering at batch_{start_from_batch}", log_fp=log_fp)
        return start_from_batch, batch_root

    batch_name = next_batch_name(batch_counter_file)
    batch_root = batch_runs_dir / f"batch_{batch_name}"
    return batch_name, batch_root


def merge_batch_manifest(batch_root: Path, global_manifest: Path, log_fp: object | None) -> None:
    batch_manifest = batch_root / "download_manifest.jsonl"
    if not batch_manifest.exists() or batch_manifest.stat().st_size == 0:
        log(f"[cycle] batch_root={batch_root} has no download manifest to merge", log_fp=log_fp)
        return

    global_manifest.parent.mkdir(parents=True, exist_ok=True)
    with batch_manifest.open("r", encoding="utf-8") as src, global_manifest.open(
        "a", encoding="utf-8"
    ) as dst:
        shutil.copyfileobj(src, dst)
    log(f"[cycle] merged manifest from {batch_root} into {global_manifest}", log_fp=log_fp)


def package_batch_root(
    python_bin: str,
    script_dir: Path,
    batch_root: Path,
    archives_dir: Path,
    log_fp: object | None,
) -> None:
    batch_raw_dir = batch_root / "raw"
    mp4_files = list(batch_raw_dir.glob("*.mp4"))
    if not mp4_files:
        log(f"[cycle] batch_root={batch_root} has no ready mp4 files to package", log_fp=log_fp)
        return

    cmd = [
        python_bin,
        str(script_dir / "package_raw_batches.py"),
        "--raw-dir",
        str(batch_raw_dir),
        "--archives-dir",
        str(archives_dir),
        "--prefix",
        "talkvid_raw",
        "--batch-root",
        str(batch_root),
        "--max-clips",
        "0",
        "--max-gb",
        "0",
        "--max-batches",
        "1",
    ]
    log(f"[package_batch] {format_cmd(cmd)}", log_fp=log_fp)
    subprocess.run(cmd, check=True, stdout=log_fp, stderr=subprocess.STDOUT)


def launch_uploader(
    python_bin: str,
    script_dir: Path,
    raw_dir: Path,
    archives_dir: Path,
    gdrive_remote: str,
    raw_folder_id: str,
    log_fp: object | None,
) -> subprocess.Popen[bytes]:
    cmd = [
        python_bin,
        str(script_dir / "upload_batches_and_cleanup.py"),
        "--raw-dir",
        str(raw_dir),
        "--archives-dir",
        str(archives_dir),
        "--remote",
        gdrive_remote,
        "--drive-root-folder-id",
        raw_folder_id,
        "--max-batches",
        "0",
    ]
    log(f"[upload_live] {format_cmd(cmd)}", log_fp=log_fp)
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
    log(f"[cycle] uploader pid={proc.pid}", log_fp=log_fp)
    return proc


def launch_fetch(
    python_bin: str,
    script_dir: Path,
    batch_root: Path,
    variant: str,
    download_max_height: int,
    download_min_duration: float,
    download_max_duration: float,
    download_min_width: int,
    download_min_height: int,
    download_min_dover: float,
    download_min_cotracker: float,
    fetch_target_gb: float,
    min_free_gb: float,
    download_timeout: int,
    global_manifest: Path,
    archives_dir: Path,
    request_min_interval_seconds: float,
    rate_limit_cooldown_seconds: int,
    max_rate_limit_cooldowns: int,
    cookies_file: str,
    cookies_from_browser: str,
    download_jobs: int,
    log_fp: object | None,
) -> subprocess.Popen[bytes]:
    cmd = [
        python_bin,
        str(script_dir / "download_talkvid.py"),
        "--output",
        str(batch_root),
        "--variant",
        variant,
        "--max-height",
        str(download_max_height),
        "--min-duration",
        str(download_min_duration),
        "--max-duration",
        str(download_max_duration),
        "--min-width",
        str(download_min_width),
        "--min-height",
        str(download_min_height),
        "--min-dover",
        str(download_min_dover),
        "--min-cotracker",
        str(download_min_cotracker),
        "--target-additional-gb",
        str(fetch_target_gb),
        "--min-free-gb",
        str(min_free_gb),
        "--timeout",
        str(download_timeout),
        "--skip-manifest",
        str(global_manifest),
        "--skip-manifest",
        str(archives_dir / "batches_manifest.jsonl"),
        "--skip-manifest",
        str(archives_dir / "uploaded_manifest.jsonl"),
        "--request-min-interval-seconds",
        str(request_min_interval_seconds),
        "--rate-limit-cooldown-seconds",
        str(rate_limit_cooldown_seconds),
        "--max-rate-limit-cooldowns",
        str(max_rate_limit_cooldowns),
    ]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    else:
        cmd.extend(["--cookies-file", cookies_file])
    cmd.extend(["--jobs", str(download_jobs)])
    log(f"[download_live] {format_cmd(cmd)}", log_fp=log_fp)
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
    return proc


def main() -> int:
    paths = discover_stage_paths(__file__)
    args = parse_args(paths.stage_root / "configs" / "default.yaml")

    try:
        _, config = load_stage_config(args.config, EXPECTED_STAGE)

        python_bin = get_str(config, "runtime", "python_bin")
        gdrive_remote = get_str(config, "gdrive", "remote")
        raw_folder_id = get_str(config, "gdrive", "raw", "folder_id")

        data_root = resolve_repo_path(
            paths.repo_root, get_str(config, "paths", "processing_folder")
        )
        archives_dir = resolve_repo_path(
            paths.repo_root, get_str(config, "paths", "raw_archives_folder")
        )
        out_dir = resolve_repo_path(paths.repo_root, get_str(config, "paths", "log_folder"))
        assert data_root is not None
        assert archives_dir is not None
        assert out_dir is not None

        raw_dir = data_root / "raw"
        batch_runs_dir = data_root / "batches"
        global_manifest = data_root / "download_manifest.jsonl"
        batch_counter_file = data_root / "next_fetch_batch_index.txt"

        raw_dir.mkdir(parents=True, exist_ok=True)
        batch_runs_dir.mkdir(parents=True, exist_ok=True)
        archives_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        log_fp = open_stage_log(out_dir, "cycle.log")
        try:
            variant = get_str(config, "fetch", "variant")
            fetch_target_gb = get_float(config, "fetch", "target_size_gb")
            batch_gb = get_float(config, "fetch", "archive_batch_gb", default=0.0)
            download_jobs = get_int(config, "fetch", "download_jobs")
            request_min_interval_seconds = get_float(
                config, "fetch", "request_min_interval_seconds"
            )
            rate_limit_cooldown_seconds = get_int(
                config, "fetch", "rate_limit_cooldown_seconds"
            )
            max_rate_limit_cooldowns = get_int(
                config, "fetch", "max_rate_limit_cooldowns"
            )
            download_max_height = get_int(config, "fetch", "download_max_height")
            download_min_duration = get_float(config, "fetch", "download_min_duration")
            download_max_duration = get_float(config, "fetch", "download_max_duration")
            download_min_width = get_int(config, "fetch", "download_min_width")
            download_min_height = get_int(config, "fetch", "download_min_height")
            download_min_dover = get_float(config, "fetch", "download_min_dover")
            download_min_cotracker = get_float(config, "fetch", "download_min_cotracker")
            min_free_gb = get_float(config, "fetch", "min_free_gb")
            download_timeout = get_int(config, "fetch", "download_timeout")
            cookies_file = get_str(config, "cookies", "file", allow_empty=True)
            cookies_from_browser = get_str(
                config, "cookies", "from_browser", allow_empty=True
            )

            log("[cycle] start", log_fp=log_fp)
            log(f"[cycle] data_root={data_root}", log_fp=log_fp)
            log(f"[cycle] batch_runs_dir={batch_runs_dir}", log_fp=log_fp)
            log(f"[cycle] raw_gdrive_folder_id={raw_folder_id}", log_fp=log_fp)
            if args.resume_batch is not None:
                log(f"[cycle] cli_resume_batch={args.resume_batch}", log_fp=log_fp)
            if args.start_from_batch is not None:
                log(f"[cycle] cli_start_from_batch={args.start_from_batch}", log_fp=log_fp)
            log(
                "[cycle] "
                f"remote={gdrive_remote} batch_gb={batch_gb} "
                f"fetch_target_gb={fetch_target_gb} jobs={download_jobs} "
                f"request_min_interval_s={request_min_interval_seconds} "
                f"rate_limit_cooldown_s={rate_limit_cooldown_seconds} "
                f"max_rate_limit_cooldowns={max_rate_limit_cooldowns}",
                log_fp=log_fp,
            )
            if cookies_from_browser:
                log(f"[cycle] cookies_from_browser={cookies_from_browser}", log_fp=log_fp)
            else:
                log(f"[cycle] cookies_file={cookies_file}", log_fp=log_fp)

            first_batch_selection: tuple[str, Path] | None = None
            if args.resume_batch is not None or args.start_from_batch is not None:
                first_batch_selection = choose_batch_root(
                    batch_runs_dir=batch_runs_dir,
                    batch_counter_file=batch_counter_file,
                    resume_batch=args.resume_batch,
                    start_from_batch=args.start_from_batch,
                    first_iteration=True,
                    log_fp=log_fp,
                )

            upload_proc: subprocess.Popen[bytes] | None = launch_uploader(
                python_bin,
                paths.script_dir,
                raw_dir,
                archives_dir,
                gdrive_remote,
                raw_folder_id,
                log_fp,
            )
            final_rc = 0
            first_iteration = True

            while True:
                if upload_proc is not None and upload_proc.poll() is not None:
                    rc = upload_proc.wait()
                    log(f"[cycle] uploader pid={upload_proc.pid} finished rc={rc}", log_fp=log_fp)
                    if final_rc == 0 and rc != 0:
                        final_rc = rc
                    upload_proc = None

                if first_iteration and first_batch_selection is not None:
                    batch_name, batch_root = first_batch_selection
                else:
                    batch_name, batch_root = choose_batch_root(
                        batch_runs_dir=batch_runs_dir,
                        batch_counter_file=batch_counter_file,
                        resume_batch=args.resume_batch,
                        start_from_batch=args.start_from_batch,
                        first_iteration=first_iteration,
                        log_fp=log_fp,
                    )
                first_iteration = False
                batch_root.mkdir(parents=True, exist_ok=True)

                fetch_proc = launch_fetch(
                    python_bin=python_bin,
                    script_dir=paths.script_dir,
                    batch_root=batch_root,
                    variant=variant,
                    download_max_height=download_max_height,
                    download_min_duration=download_min_duration,
                    download_max_duration=download_max_duration,
                    download_min_width=download_min_width,
                    download_min_height=download_min_height,
                    download_min_dover=download_min_dover,
                    download_min_cotracker=download_min_cotracker,
                    fetch_target_gb=fetch_target_gb,
                    min_free_gb=min_free_gb,
                    download_timeout=download_timeout,
                    global_manifest=global_manifest,
                    archives_dir=archives_dir,
                    request_min_interval_seconds=request_min_interval_seconds,
                    rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
                    max_rate_limit_cooldowns=max_rate_limit_cooldowns,
                    cookies_file=cookies_file,
                    cookies_from_browser=cookies_from_browser,
                    download_jobs=download_jobs,
                    log_fp=log_fp,
                )
                download_rc = fetch_proc.wait()
                log(
                    f"[cycle] fetch batch={batch_name} rc={download_rc}",
                    log_fp=log_fp,
                )

                merge_batch_manifest(batch_root, global_manifest, log_fp)
                package_batch_root(python_bin, paths.script_dir, batch_root, archives_dir, log_fp)
                if upload_proc is None:
                    upload_proc = launch_uploader(
                        python_bin,
                        paths.script_dir,
                        raw_dir,
                        archives_dir,
                        gdrive_remote,
                        raw_folder_id,
                        log_fp,
                    )

                if download_rc == 10:
                    log(
                        f"[cycle] batch={batch_name} sealed at target size; starting next fetch run",
                        log_fp=log_fp,
                    )
                    continue

                if download_rc != 0:
                    final_rc = download_rc
                    log(
                        f"[cycle] download stage exited rc={download_rc}; stopping fetch loop after packaging current batch",
                        log_fp=log_fp,
                    )
                break

            if upload_proc is not None:
                upload_rc = upload_proc.wait()
                log(f"[cycle] uploader pid={upload_proc.pid} finished rc={upload_rc}", log_fp=log_fp)
                if final_rc == 0 and upload_rc != 0:
                    final_rc = upload_rc

            tail_cmd = [
                python_bin,
                str(paths.script_dir / "upload_batches_and_cleanup.py"),
                "--raw-dir",
                str(raw_dir),
                "--archives-dir",
                str(archives_dir),
                "--remote",
                gdrive_remote,
                "--drive-root-folder-id",
                raw_folder_id,
                "--max-batches",
                "0",
            ]
            log(f"[upload_tail] {format_cmd(tail_cmd)}", log_fp=log_fp)
            tail_rc = subprocess.run(tail_cmd, stdout=log_fp, stderr=subprocess.STDOUT).returncode
            if tail_rc != 0:
                log(f"[cycle] stage=upload_tail rc={tail_rc}; continuing", log_fp=log_fp)

            log("[cycle] done", log_fp=log_fp)
            return final_rc
        finally:
            if log_fp is not None:
                log_fp.close()
    except ConfigError as exc:
        return exit_with_config_error(exc)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
