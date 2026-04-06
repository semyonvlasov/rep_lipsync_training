#!/usr/bin/env python3
"""
Upload packaged tar batches to Google Drive with rclone and delete local data
 only after successful upload.
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
            log(f"[Upload:rclone] {line}")
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


def file_matches(path: Path, size_bytes: object, mtime_ns: object) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    if size_bytes is not None:
        try:
            if stat.st_size != int(size_bytes):
                return False
        except (TypeError, ValueError):
            return False
    if mtime_ns is not None:
        try:
            if stat.st_mtime_ns != int(mtime_ns):
                return False
        except (TypeError, ValueError):
            return False
    return True


def unlink_if_matching(path: Path, size_bytes: object = None, mtime_ns: object = None) -> bool:
    if not path.exists():
        return False
    if size_bytes is not None or mtime_ns is not None:
        if not file_matches(path, size_bytes, mtime_ns):
            log(f"[Upload] keeping {path} because it no longer matches the sealed batch snapshot")
            return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def cleanup_uploaded_files(raw_dir: Path, files: list[str], file_entries: list[dict], tar_path: Path) -> tuple[int, int]:
    cleaned = 0
    skipped = 0

    if file_entries:
        for entry in file_entries:
            fname = str(entry.get("name") or "")
            if not fname:
                continue
            mp4_path = raw_dir / fname
            json_path = raw_dir / Path(fname).with_suffix(".json")
            if unlink_if_matching(mp4_path, entry.get("mp4_bytes"), entry.get("mp4_mtime_ns")):
                cleaned += 1
            elif mp4_path.exists():
                skipped += 1
            if unlink_if_matching(json_path, entry.get("json_bytes"), entry.get("json_mtime_ns")):
                cleaned += 1
            elif json_path.exists():
                skipped += 1
    else:
        for fname in files:
            mp4_path = raw_dir / fname
            json_path = raw_dir / Path(fname).with_suffix(".json")
            for path in (mp4_path, json_path):
                try:
                    if path.exists():
                        path.unlink()
                        cleaned += 1
                except OSError:
                    skipped += 1

    try:
        if tar_path.exists():
            tar_path.unlink()
    except OSError:
        pass
    return cleaned, skipped


def cleanup_batch_root(batch_root: Path) -> None:
    current = batch_root
    while True:
        try:
            current.rmdir()
        except OSError:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--archives-dir", required=True)
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--drive-root-folder-id", required=True)
    parser.add_argument("--uploaded-manifest", default=None)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--rclone-transfers", type=int, default=1)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    archives_dir = Path(args.archives_dir)
    batches_manifest = archives_dir / "batches_manifest.jsonl"
    uploaded_manifest = Path(args.uploaded_manifest) if args.uploaded_manifest else (archives_dir / "uploaded_manifest.jsonl")

    batches = load_jsonl(batches_manifest)
    uploaded = load_jsonl(uploaded_manifest)
    uploaded_tars = {str(item.get("tar_path")) for item in uploaded}

    pending = []
    for batch in batches:
        tar_path = Path(str(batch.get("tar_path", "")))
        if not tar_path.exists():
            continue
        if str(tar_path) in uploaded_tars:
            continue
        pending.append(batch)

    log(f"[Upload] raw_dir={raw_dir}")
    log(f"[Upload] archives_dir={archives_dir}")
    log(f"[Upload] pending_batches={len(pending)}")

    uploaded_count = 0
    for batch in pending:
        tar_path = Path(str(batch["tar_path"]))
        batch_raw_dir = Path(str(batch.get("raw_dir") or args.raw_dir))
        batch_root = batch.get("batch_root")
        remote_name = tar_path.name
        cmd = [
            "rclone",
            "copyto",
            "--drive-root-folder-id", args.drive_root_folder_id,
            "--transfers", str(args.rclone_transfers),
            str(tar_path),
            f"{args.remote}{remote_name}",
        ]
        log(f"[Upload] uploading {tar_path.name}")
        run_logged(cmd)
        append_jsonl(
            uploaded_manifest,
            {
                "batch_index": batch.get("batch_index"),
                "tar_path": str(tar_path),
                "remote_name": remote_name,
                "files": batch.get("files", []),
                "file_entries": batch.get("file_entries", []),
                "raw_dir": str(batch_raw_dir),
                "batch_root": batch_root,
            },
        )
        cleaned, skipped = cleanup_uploaded_files(
            batch_raw_dir,
            list(batch.get("files", [])),
            list(batch.get("file_entries", [])),
            tar_path,
        )
        if batch_root:
            cleanup_batch_root(Path(str(batch_root)))
        log(f"[Upload] uploaded and cleaned {tar_path.name} cleaned_paths={cleaned} skipped_paths={skipped}")
        uploaded_count += 1
        if args.max_batches > 0 and uploaded_count >= args.max_batches:
            break

    log(f"[Upload] uploaded_batches={uploaded_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
