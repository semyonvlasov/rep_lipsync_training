#!/usr/bin/env python3
"""Import a dataset snapshot produced by export_dataset_snapshot.py."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lipsync.dataset_snapshot.v1"


def resolve_training_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return TRAINING_ROOT / path


def safe_extract_tar(tar_path: Path, target_root: Path, *, dry_run: bool) -> None:
    target_root_resolved = target_root.resolve()
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            target = (target_root / member.name).resolve()
            if target_root_resolved != target and target_root_resolved not in target.parents:
                raise RuntimeError(f"Unsafe tar member path in {tar_path}: {member.name}")
        if not dry_run:
            tar.extractall(target_root)


def copy_file(source: Path, target: Path, *, dry_run: bool) -> bool:
    if not source.exists():
        return False
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def copy_dir(source: Path, target: Path, *, dry_run: bool) -> bool:
    if not source.exists():
        return False
    if dry_run:
        return True
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()
    manifest_path = snapshot_dir / "dataset_snapshot_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise SystemExit(f"Unsupported snapshot schema: {manifest.get('schema')}")

    actions = []
    for shard in manifest.get("shards", []):
        shard_rel = shard.get("path")
        if not shard_rel:
            continue
        shard_path = TRAINING_ROOT / shard_rel
        if not shard_path.exists():
            shard_path = snapshot_dir / "shards" / Path(shard_rel).name
        actions.append(f"extract {shard_path} -> {TRAINING_ROOT}")
        safe_extract_tar(shard_path, TRAINING_ROOT, dry_run=args.dry_run)

    paths = manifest.get("paths") or {}
    copies = [
        (snapshot_dir / "merge_manifest.jsonl", resolve_training_path(paths.get("merge_manifest"))),
        (snapshot_dir / "sync_alignment_manifest.jsonl", resolve_training_path(paths.get("sync_alignment_registry"))),
    ]
    for source, target in copies:
        if target and copy_file(source, target, dry_run=args.dry_run):
            actions.append(f"copy {source} -> {target}")

    dir_copies = [
        (snapshot_dir / "prepared", resolve_training_path(paths.get("prepared_dir"))),
        (snapshot_dir / "split", resolve_training_path(paths.get("split_dir"))),
    ]
    for source, target in dir_copies:
        if target and copy_dir(source, target, dry_run=args.dry_run):
            actions.append(f"copytree {source} -> {target}")

    print(json.dumps({"snapshot_dir": str(snapshot_dir), "dry_run": args.dry_run, "actions": actions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
