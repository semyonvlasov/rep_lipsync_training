#!/usr/bin/env python3
"""
Build a generator train/val split from a prepared eligible manifest and an
explicit validation snapshot, keeping only samples that passed sync filtering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--explicit-val-snapshot", required=True)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    split_dir = Path(args.split_dir)
    explicit_val_snapshot = Path(args.explicit_val_snapshot)

    if not explicit_val_snapshot.exists():
        raise SystemExit(f"Missing explicit val snapshot: {explicit_val_snapshot}")

    prepared_summary_path = prepared_dir / "summary.json"
    eligible_manifest_path = prepared_dir / "eligible_manifest.json"
    prepared_summary = (
        json.loads(prepared_summary_path.read_text(encoding="utf-8"))
        if prepared_summary_path.exists()
        else {}
    )
    eligible_manifest = json.loads(eligible_manifest_path.read_text(encoding="utf-8"))
    if not eligible_manifest:
        raise SystemExit(f"No eligible samples in {eligible_manifest_path}")

    ordered_names = [item["name"] for item in eligible_manifest]
    eligible_set = set(ordered_names)

    explicit_val_names = [
        line.strip()
        for line in explicit_val_snapshot.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(set(explicit_val_names)) != len(explicit_val_names):
        raise SystemExit(f"Explicit val snapshot contains duplicates: {explicit_val_snapshot}")

    missing_val = [name for name in explicit_val_names if name not in eligible_set]
    val_names = [name for name in explicit_val_names if name in eligible_set]
    if not val_names:
        raise SystemExit(
            f"Explicit val snapshot has no overlap with eligible manifest: {explicit_val_snapshot}"
        )

    val_set = set(val_names)
    train_names = [name for name in ordered_names if name not in val_set]
    if not train_names:
        raise SystemExit("No train samples remain after applying explicit eligible val snapshot")

    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_snapshot.txt").write_text(
        "".join(f"{name}\n" for name in train_names),
        encoding="utf-8",
    )
    (split_dir / "val_snapshot.txt").write_text(
        "".join(f"{name}\n" for name in val_names),
        encoding="utf-8",
    )
    (split_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": "prepared_manifest_plus_explicit_val",
                "prepared_dir": str(prepared_dir),
                "prepared_summary": prepared_summary,
                "eligible_total": len(ordered_names),
                "train_count": len(train_names),
                "val_count": len(val_names),
                "explicit_val_snapshot": str(explicit_val_snapshot),
                "explicit_val_total": len(explicit_val_names),
                "explicit_val_missing_due_to_filtering": len(missing_val),
                "explicit_val_missing_examples": missing_val[:20],
                "val_tail": val_names[-10:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[generator-split] built split eligible_total={len(ordered_names)} "
        f"train={len(train_names)} val={len(val_names)} "
        f"filtered_val_missing={len(missing_val)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
