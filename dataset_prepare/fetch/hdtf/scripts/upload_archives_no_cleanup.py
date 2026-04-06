#!/usr/bin/env python3
"""
Upload tar archives to Google Drive with rclone while keeping local data intact.

This is intended for datasets like HDTF clip batches, where we want resumable
uploads but do not want any cleanup side effects.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def run_logged(cmd: list[str]) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            log(f"[UploadArchive:rclone] {line}")
    rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def unlink_local_tar(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:
        raise RuntimeError(f"Failed to remove uploaded local archive {path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-dir", required=True)
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--drive-root-folder-id", required=True)
    parser.add_argument("--uploaded-manifest", default=None)
    parser.add_argument("--prefix", default=None, help="Optional filename prefix filter")
    parser.add_argument("--max-batches", type=int, default=0, help="Upload at most this many new archives (0=all)")
    parser.add_argument("--rclone-transfers", type=int, default=1)
    parser.add_argument("--delete-local-after-upload", action="store_true", help="Delete each local tar after successful upload")
    args = parser.parse_args()

    archives_dir = Path(args.archives_dir)
    uploaded_manifest = Path(args.uploaded_manifest) if args.uploaded_manifest else (archives_dir / "uploaded_manifest.jsonl")
    uploaded = load_jsonl(uploaded_manifest)
    uploaded_names = {str(item.get("remote_name") or item.get("name") or "") for item in uploaded}

    pending = []
    for tar_path in sorted(archives_dir.glob("*.tar")):
        if args.prefix and not tar_path.name.startswith(args.prefix):
            continue
        if tar_path.name in uploaded_names:
            continue
        pending.append(tar_path)

    log(f"[UploadArchive] archives_dir={archives_dir}")
    log(f"[UploadArchive] pending_archives={len(pending)}")

    uploaded_count = 0
    for tar_path in pending:
        cmd = [
            "rclone",
            "copyto",
            "--drive-root-folder-id", args.drive_root_folder_id,
            "--transfers", str(args.rclone_transfers),
            str(tar_path),
            f"{args.remote}{tar_path.name}",
        ]
        log(f"[UploadArchive] uploading {tar_path.name}")
        run_logged(cmd)
        append_jsonl(
            uploaded_manifest,
            {
                "name": tar_path.name,
                "tar_path": str(tar_path),
                "remote_name": tar_path.name,
                "size_bytes": tar_path.stat().st_size,
            },
        )
        if args.delete_local_after_upload:
            unlink_local_tar(tar_path)
            log(f"[UploadArchive] deleted local archive {tar_path.name}")
        uploaded_count += 1
        if args.max_batches > 0 and uploaded_count >= args.max_batches:
            break

    log(f"[UploadArchive] uploaded_archives={uploaded_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
