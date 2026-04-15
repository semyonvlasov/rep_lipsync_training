#!/usr/bin/env python3
"""
Upload a packaged checkpoint tar to Drive and clean local artifacts on success.

Intended to be spawned in the background by the GAN trainer. Failures do not
delete local artifacts so the step bundle can be retried manually.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def run_logged(cmd: list[str], prefix: str) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(f"{prefix} {line}")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload checkpoint tar bundle and cleanup on success")
    parser.add_argument("--tar-path", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--drive-root-folder-id", required=True)
    parser.add_argument("--remote-path", required=True)
    return parser.parse_args()


def cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    args = parse_args()
    tar_path = Path(args.tar_path)
    artifact_dir = Path(args.artifact_dir)

    if not tar_path.exists():
        log(f"[CheckpointUpload] tar missing, nothing to upload: {tar_path}")
        return 1

    cmd = [
        "rclone",
        "copyto",
        "--drive-root-folder-id",
        args.drive_root_folder_id,
        str(tar_path),
        f"{args.remote}{args.remote_path}",
    ]

    log(f"[CheckpointUpload] start {tar_path} -> {args.remote}{args.remote_path}")
    try:
        run_logged(cmd, prefix="[CheckpointUpload:rclone]")
    except subprocess.CalledProcessError as exc:
        log(
            f"[CheckpointUpload] upload failed rc={exc.returncode}; "
            f"keeping local artifacts tar={tar_path} dir={artifact_dir}"
        )
        return exc.returncode or 1
    except Exception as exc:  # noqa: BLE001
        log(
            f"[CheckpointUpload] unrecoverable error: {exc}; "
            f"keeping local artifacts tar={tar_path} dir={artifact_dir}"
        )
        return 1

    cleanup_path(tar_path)
    cleanup_path(artifact_dir)
    log(f"[CheckpointUpload] uploaded and cleaned tar={tar_path} dir={artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
