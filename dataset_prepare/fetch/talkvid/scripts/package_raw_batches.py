#!/usr/bin/env python3
"""
Pack raw video clips into tar batches limited by clip count and/or total size.
"""

import argparse
import json
import tarfile
import time
from pathlib import Path


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def iter_ready_files(raw_dir: Path):
    for path in sorted(raw_dir.glob("*.mp4")):
        sidecar = path.with_suffix(".json")
        if path.is_file() and sidecar.is_file():
            yield path


def read_done_archives(archives_dir: Path) -> set[str]:
    done = set()
    manifest = archives_dir / "batches_manifest.jsonl"
    if not manifest.exists():
        return done
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in obj.get("files", []):
                done.add(item)
    return done


def next_batch_index(archives_dir: Path) -> int:
    next_idx = 0
    manifest = archives_dir / "batches_manifest.jsonl"
    if not manifest.exists():
        return next_idx
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                next_idx = max(next_idx, int(obj.get("batch_index", -1)) + 1)
            except Exception:
                pass
    return next_idx


def parse_explicit_batch_index(batch_name: str) -> int:
    value = str(batch_name).strip()
    if value.startswith("batch_"):
        value = value[len("batch_") :]
    if not value.isdigit():
        raise ValueError(f"batch_name must look like 0039 or batch_0039, got: {batch_name!r}")
    return int(value)


def flush_batch(archives_dir: Path, prefix: str, batch_index: int, files: list[Path]) -> Path:
    archives_dir.mkdir(parents=True, exist_ok=True)
    tar_path = archives_dir / f"{prefix}_batch_{batch_index:04d}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for path in files:
            tar.add(path, arcname=path.name)
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                tar.add(sidecar, arcname=sidecar.name)
    return tar_path


def build_file_entries(files: list[Path]) -> list[dict]:
    entries = []
    for path in files:
        sidecar = path.with_suffix(".json")
        mp4_stat = path.stat()
        json_stat = sidecar.stat()
        entries.append(
            {
                "name": path.name,
                "mp4_bytes": mp4_stat.st_size,
                "mp4_mtime_ns": mp4_stat.st_mtime_ns,
                "json_bytes": json_stat.st_size,
                "json_mtime_ns": json_stat.st_mtime_ns,
            }
        )
    return entries


def append_manifest(archives_dir: Path, payload: dict) -> None:
    manifest = archives_dir / "batches_manifest.jsonl"
    with open(manifest, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Directory with raw .mp4 clips")
    parser.add_argument("--archives-dir", required=True, help="Output directory for tar archives")
    parser.add_argument("--prefix", default="talkvid_raw", help="Tar filename prefix")
    parser.add_argument(
        "--batch-root",
        default=None,
        help="Optional per-run root directory to remove after successful upload",
    )
    parser.add_argument(
        "--batch-name",
        default="",
        help="Optional explicit archive batch id like 0039 or batch_0039.",
    )
    parser.add_argument("--max-clips", type=int, default=10, help="Max clips per archive (0=unlimited)")
    parser.add_argument("--max-gb", type=float, default=1.0, help="Approximate max total mp4 size per archive")
    parser.add_argument("--max-batches", type=int, default=0, help="Stop after creating this many new batches (0=unlimited)")
    parser.add_argument(
        "--require-full-batch",
        action="store_true",
        help="Keep the tail batch local unless it is forced to close by count/size overflow",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    archives_dir = Path(args.archives_dir)
    max_bytes = int(args.max_gb * (1024 ** 3)) if args.max_gb > 0 else 0

    done = read_done_archives(archives_dir)
    files = [p for p in iter_ready_files(raw_dir) if p.name not in done]
    log(f"[Pack] raw_dir={raw_dir}")
    log(f"[Pack] archives_dir={archives_dir}")
    log(f"[Pack] ready_files={len(files)}")

    batch_files: list[Path] = []
    batch_bytes = 0
    explicit_batch_name = str(args.batch_name or "").strip()
    batch_index = (
        parse_explicit_batch_index(explicit_batch_name)
        if explicit_batch_name
        else next_batch_index(archives_dir)
    )
    created = 0

    for path in files:
        file_size = path.stat().st_size
        would_exceed_count = bool(args.max_clips > 0 and batch_files and len(batch_files) >= args.max_clips)
        would_exceed_size = bool(max_bytes > 0 and batch_files and (batch_bytes + file_size > max_bytes))
        if would_exceed_count or would_exceed_size:
            tar_path = flush_batch(archives_dir, args.prefix, batch_index, batch_files)
            append_manifest(
                archives_dir,
                {
                    "batch_index": batch_index,
                    "batch_name": f"{batch_index:04d}",
                    "tar_path": str(tar_path),
                    "raw_dir": str(raw_dir),
                    "batch_root": args.batch_root,
                    "files": [p.name for p in batch_files],
                    "file_entries": build_file_entries(batch_files),
                    "count": len(batch_files),
                    "mp4_bytes": batch_bytes,
                },
            )
            log(f"[Pack] created {tar_path.name}: clips={len(batch_files)} size_gb={batch_bytes / (1024 ** 3):.3f}")
            created += 1
            batch_index += 1
            batch_files = []
            batch_bytes = 0
            if args.max_batches > 0 and created >= args.max_batches:
                log("[Pack] Reached max new batches")
                return 0

        batch_files.append(path)
        batch_bytes += file_size

    if batch_files and (args.max_batches == 0 or created < args.max_batches):
        oversize_single = len(batch_files) == 1 and max_bytes > 0 and batch_bytes > max_bytes
        if args.require_full_batch and not oversize_single:
            log(
                f"[Pack] leaving partial batch unsealed: clips={len(batch_files)} "
                f"size_gb={batch_bytes / (1024 ** 3):.3f}"
            )
            return 0
        tar_path = flush_batch(archives_dir, args.prefix, batch_index, batch_files)
        append_manifest(
            archives_dir,
            {
                "batch_index": batch_index,
                "batch_name": f"{batch_index:04d}",
                "tar_path": str(tar_path),
                "raw_dir": str(raw_dir),
                "batch_root": args.batch_root,
                "files": [p.name for p in batch_files],
                "file_entries": build_file_entries(batch_files),
                "count": len(batch_files),
                "mp4_bytes": batch_bytes,
            },
        )
        log(f"[Pack] created {tar_path.name}: clips={len(batch_files)} size_gb={batch_bytes / (1024 ** 3):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
