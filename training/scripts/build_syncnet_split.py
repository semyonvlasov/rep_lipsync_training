#!/usr/bin/env python3
"""
Build a frozen SyncNet train/val split from a prepared eligible dataset manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def choose_val_split(
    eligible_manifest: list[dict],
    val_count: int,
    target_val_batches: int | None,
    batch_size: int,
) -> tuple[list[dict], list[dict], dict]:
    if target_val_batches is None or target_val_batches <= 0:
        if len(eligible_manifest) <= val_count:
            raise SystemExit(
                f"Need more than val_count={val_count} eligible samples, found {len(eligible_manifest)}"
            )
        train_items = eligible_manifest[:-val_count]
        val_items = eligible_manifest[-val_count:]
        return train_items, val_items, {
            "mode": "fixed_val_count",
            "target_val_batches": None,
            "target_val_frames": None,
        }

    target_val_frames = int(target_val_batches) * int(batch_size)
    selected_reversed: list[dict] = []
    val_frames = 0
    for item in reversed(eligible_manifest):
        selected_reversed.append(item)
        val_frames += int(item.get("frame_count") or 0)
        if val_frames >= target_val_frames:
            break
    val_items = list(reversed(selected_reversed))
    if len(eligible_manifest) <= len(val_items):
        raise SystemExit(
            f"Need at least one training sample after target_val_batches={target_val_batches}; "
            f"eligible={len(eligible_manifest)} val={len(val_items)}"
        )
    train_items = eligible_manifest[:-len(val_items)]
    return train_items, val_items, {
        "mode": "target_val_batches",
        "target_val_batches": int(target_val_batches),
        "target_val_frames": target_val_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--val-count", type=int, default=2048)
    parser.add_argument("--target-val-batches", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-count", type=int, default=0)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    split_dir = Path(args.split_dir)
    summary_path = prepared_dir / "summary.json"
    eligible_manifest_path = prepared_dir / "eligible_manifest.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    eligible_manifest = json.loads(eligible_manifest_path.read_text(encoding="utf-8"))
    train_items, val_items, split_meta = choose_val_split(
        eligible_manifest,
        val_count=args.val_count,
        target_val_batches=args.target_val_batches,
        batch_size=args.batch_size,
    )
    max_train_count = max(0, int(args.max_train_count or 0))
    if max_train_count > 0 and len(train_items) > max_train_count:
        train_items = train_items[:max_train_count]
    train_names = [item["name"] for item in train_items]
    val_names = [item["name"] for item in val_items]
    train_effective_frames = sum(int(item.get("frame_count") or 0) for item in train_items)
    val_effective_frames = sum(int(item.get("frame_count") or 0) for item in val_items)

    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_snapshot.txt").write_text("".join(f"{name}\n" for name in train_names), encoding="utf-8")
    (split_dir / "val_snapshot.txt").write_text("".join(f"{name}\n" for name in val_names), encoding="utf-8")
    (split_dir / "summary.json").write_text(
        json.dumps(
            {
                "prepared_dir": str(prepared_dir),
                "prepared_summary": summary,
                "eligible_total": len(eligible_manifest),
                "split_mode": split_meta["mode"],
                "target_val_batches": split_meta["target_val_batches"],
                "target_val_frames": split_meta["target_val_frames"],
                "max_train_count": max_train_count or None,
                "batch_size": int(args.batch_size),
                "train_count": len(train_names),
                "val_count": len(val_names),
                "train_effective_frames": train_effective_frames,
                "val_effective_frames": val_effective_frames,
                "train_batches_estimate": (train_effective_frames + int(args.batch_size) - 1) // int(args.batch_size),
                "val_batches_estimate": (val_effective_frames + int(args.batch_size) - 1) // int(args.batch_size),
                "val_tail": val_names[-10:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[syncnet-split] built split eligible_total={len(eligible_manifest)} "
        f"train={len(train_names)} val={len(val_names)} "
        f"train_frames={train_effective_frames} val_frames={val_effective_frames}",
        flush=True,
    )


if __name__ == "__main__":
    main()
