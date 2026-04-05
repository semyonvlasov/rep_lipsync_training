#!/usr/bin/env python3
"""Merge standalone lazy cache entries into canonical per-dataset _lazy_cache roots.

This is mainly for migrating old generator-only cache roots like:
  data/_lazy_cache_generator_*

into the shared cache layout used by canonical dataset roots:
  data/hdtf/processed/_lazy_cache
  data/talkvid/processed/_lazy_cache
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        required=True,
        help="Path to training/ root that contains data/.",
    )
    parser.add_argument(
        "--source-cache-root",
        required=True,
        help="Standalone cache root to merge from.",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help="Canonical dataset root(s). Defaults to data/hdtf/processed and data/talkvid/processed.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move missing .npy files into shared caches instead of copying them.",
    )
    parser.add_argument(
        "--prune-source",
        action="store_true",
        help="Delete source cache leaf dirs only when every .npy file has been mirrored to target.",
    )
    return parser.parse_args()


def canonical_leaf_name(video_path: Path) -> str:
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{video_path.stem}--{digest}"


def iter_dataset_mp4s(dataset_root: Path):
    for path in dataset_root.rglob("*.mp4"):
        if "_lazy_cache" in path.parts:
            continue
        yield path


def build_leaf_index(dataset_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dataset_root in dataset_roots:
        for video_path in iter_dataset_mp4s(dataset_root):
            leaf = canonical_leaf_name(video_path)
            target_dir = dataset_root / "_lazy_cache" / leaf
            prev = index.get(leaf)
            if prev is not None and prev != target_dir:
                raise RuntimeError(f"Leaf collision for {leaf}: {prev} vs {target_dir}")
            index[leaf] = target_dir
    return index


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_leaf(source_leaf: Path, target_leaf: Path, move: bool) -> dict[str, int]:
    stats = {
        "files_copied": 0,
        "files_moved": 0,
        "files_already_present": 0,
        "conflicts": 0,
        "tracked_source_files": 0,
    }
    target_leaf.mkdir(parents=True, exist_ok=True)

    for source_file in sorted(source_leaf.glob("*.npy")):
        stats["tracked_source_files"] += 1
        target_file = target_leaf / source_file.name
        if not target_file.exists():
            if move:
                shutil.move(str(source_file), str(target_file))
                stats["files_moved"] += 1
            else:
                shutil.copy2(str(source_file), str(target_file))
                stats["files_copied"] += 1
            continue

        if source_file.stat().st_size == target_file.stat().st_size:
            if sha1_file(source_file) == sha1_file(target_file):
                stats["files_already_present"] += 1
                continue
        stats["conflicts"] += 1

    return stats


def maybe_prune_source_leaf(source_leaf: Path) -> bool:
    tracked = list(source_leaf.glob("*.npy"))
    if tracked:
        return False
    try:
        source_leaf.rmdir()
        return True
    except OSError:
        return False


def main() -> int:
    args = parse_args()
    training_root = Path(args.training_root).resolve()
    source_root = Path(args.source_cache_root).resolve()

    if not source_root.exists():
        print(f"[merge_lazy_caches] source cache does not exist: {source_root}")
        return 0

    dataset_roots = [Path(p).resolve() for p in args.dataset_root]
    if not dataset_roots:
        dataset_roots = [
            training_root / "data" / "hdtf" / "processed",
            training_root / "data" / "talkvid" / "processed",
        ]
    dataset_roots = [p for p in dataset_roots if p.is_dir()]
    if not dataset_roots:
        raise RuntimeError("No dataset roots found")

    leaf_index = build_leaf_index(dataset_roots)
    print(
        f"[merge_lazy_caches] indexed {len(leaf_index)} cache leaves "
        f"across {len(dataset_roots)} dataset roots"
    )

    source_leaves = sorted([p for p in source_root.iterdir() if p.is_dir()])
    merged_leaves = 0
    unmatched_leaves = 0
    pruned_leaves = 0
    files_copied = 0
    files_moved = 0
    files_already_present = 0
    conflicts = 0

    for source_leaf in source_leaves:
        target_leaf = leaf_index.get(source_leaf.name)
        if target_leaf is None:
            unmatched_leaves += 1
            print(f"[merge_lazy_caches] unmatched leaf: {source_leaf.name}")
            continue

        leaf_stats = merge_leaf(source_leaf, target_leaf, move=args.move)
        merged_leaves += 1
        files_copied += leaf_stats["files_copied"]
        files_moved += leaf_stats["files_moved"]
        files_already_present += leaf_stats["files_already_present"]
        conflicts += leaf_stats["conflicts"]

        if args.prune_source and leaf_stats["conflicts"] == 0:
            if maybe_prune_source_leaf(source_leaf):
                pruned_leaves += 1

    if args.prune_source:
        try:
            source_root.rmdir()
        except OSError:
            pass

    print(
        "[merge_lazy_caches] "
        f"merged_leaves={merged_leaves} unmatched_leaves={unmatched_leaves} "
        f"files_copied={files_copied} files_moved={files_moved} "
        f"files_already_present={files_already_present} conflicts={conflicts} "
        f"pruned_leaves={pruned_leaves}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
