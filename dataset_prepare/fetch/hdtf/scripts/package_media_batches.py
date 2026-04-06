#!/usr/bin/env python3
"""
Pack media files into tar batches limited by approximate total size.

Useful for datasets like HDTF clips, where we want smaller upload units
without deleting the original local media.
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


def iter_media_files(input_dir: Path, pattern: str):
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file():
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


def sidecar_path(path: Path, include_sidecars: bool, sidecar_ext: str) -> Path | None:
    if not include_sidecars:
        return None
    candidate = path.with_suffix(sidecar_ext)
    return candidate if candidate.is_file() else None


def flush_batch(
    archives_dir: Path,
    prefix: str,
    batch_index: int,
    files: list[Path],
    include_sidecars: bool,
    sidecar_ext: str,
) -> Path:
    archives_dir.mkdir(parents=True, exist_ok=True)
    tar_path = archives_dir / f"{prefix}_batch_{batch_index:04d}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for path in files:
            tar.add(path, arcname=path.name)
            sidecar = sidecar_path(path, include_sidecars, sidecar_ext)
            if sidecar is not None:
                tar.add(sidecar, arcname=sidecar.name)
    return tar_path


def build_file_entries(files: list[Path], include_sidecars: bool, sidecar_ext: str) -> list[dict]:
    entries = []
    for path in files:
        stat = path.stat()
        entry = {
            "name": path.name,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        sidecar = sidecar_path(path, include_sidecars, sidecar_ext)
        if sidecar is not None:
            sidecar_stat = sidecar.stat()
            entry["sidecar_name"] = sidecar.name
            entry["sidecar_bytes"] = sidecar_stat.st_size
            entry["sidecar_mtime_ns"] = sidecar_stat.st_mtime_ns
        entries.append(entry)
    return entries


def append_manifest(archives_dir: Path, payload: dict) -> None:
    manifest = archives_dir / "batches_manifest.jsonl"
    with open(manifest, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with media files to pack")
    parser.add_argument("--archives-dir", required=True, help="Output directory for tar archives")
    parser.add_argument("--prefix", required=True, help="Tar filename prefix")
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern for media files")
    parser.add_argument("--max-gb", type=float, default=1.0, help="Approximate max total file size per archive")
    parser.add_argument("--max-files", type=int, default=0, help="Optional max files per archive (0=unlimited)")
    parser.add_argument("--max-batches", type=int, default=0, help="Stop after creating this many new batches (0=unlimited)")
    parser.add_argument("--include-sidecars", action="store_true", help="Include sibling sidecar files")
    parser.add_argument("--sidecar-ext", default=".json", help="Sidecar extension when --include-sidecars is enabled")
    parser.add_argument("--allow-partial-tail", action="store_true", help="Seal the final partial batch too")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    archives_dir = Path(args.archives_dir)
    max_bytes = int(args.max_gb * (1024 ** 3)) if args.max_gb > 0 else 0

    done = read_done_archives(archives_dir)
    files = [p for p in iter_media_files(input_dir, args.pattern) if p.name not in done]
    log(f"[PackMedia] input_dir={input_dir}")
    log(f"[PackMedia] archives_dir={archives_dir}")
    log(f"[PackMedia] pattern={args.pattern} ready_files={len(files)}")

    batch_files: list[Path] = []
    batch_bytes = 0
    batch_index = next_batch_index(archives_dir)
    created = 0

    for path in files:
        file_size = path.stat().st_size
        would_exceed_count = bool(args.max_files > 0 and batch_files and len(batch_files) >= args.max_files)
        would_exceed_size = bool(max_bytes > 0 and batch_files and (batch_bytes + file_size > max_bytes))
        if would_exceed_count or would_exceed_size:
            tar_path = flush_batch(
                archives_dir,
                args.prefix,
                batch_index,
                batch_files,
                args.include_sidecars,
                args.sidecar_ext,
            )
            append_manifest(
                archives_dir,
                {
                    "batch_index": batch_index,
                    "tar_path": str(tar_path),
                    "source_dir": str(input_dir),
                    "files": [p.name for p in batch_files],
                    "file_entries": build_file_entries(batch_files, args.include_sidecars, args.sidecar_ext),
                    "count": len(batch_files),
                    "bytes": batch_bytes,
                },
            )
            log(f"[PackMedia] created {tar_path.name}: files={len(batch_files)} size_gb={batch_bytes / (1024 ** 3):.3f}")
            created += 1
            batch_index += 1
            batch_files = []
            batch_bytes = 0
            if args.max_batches > 0 and created >= args.max_batches:
                log("[PackMedia] Reached max new batches")
                return 0

        batch_files.append(path)
        batch_bytes += file_size

    if batch_files and (args.max_batches == 0 or created < args.max_batches):
        oversize_single = len(batch_files) == 1 and max_bytes > 0 and batch_bytes > max_bytes
        if not args.allow_partial_tail and not oversize_single:
            log(
                f"[PackMedia] leaving partial tail unsealed: files={len(batch_files)} "
                f"size_gb={batch_bytes / (1024 ** 3):.3f}"
            )
            return 0
        tar_path = flush_batch(
            archives_dir,
            args.prefix,
            batch_index,
            batch_files,
            args.include_sidecars,
            args.sidecar_ext,
        )
        append_manifest(
            archives_dir,
            {
                "batch_index": batch_index,
                "tar_path": str(tar_path),
                "source_dir": str(input_dir),
                "files": [p.name for p in batch_files],
                "file_entries": build_file_entries(batch_files, args.include_sidecars, args.sidecar_ext),
                "count": len(batch_files),
                "bytes": batch_bytes,
            },
        )
        log(f"[PackMedia] created {tar_path.name}: files={len(batch_files)} size_gb={batch_bytes / (1024 ** 3):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
