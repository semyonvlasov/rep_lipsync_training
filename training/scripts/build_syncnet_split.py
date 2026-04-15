#!/usr/bin/env python3
"""
Build a frozen SyncNet train/val split from a prepared eligible dataset manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--val-count", type=int, default=2048)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    split_dir = Path(args.split_dir)
    summary_path = prepared_dir / "summary.json"
    eligible_manifest_path = prepared_dir / "eligible_manifest.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    eligible_manifest = json.loads(eligible_manifest_path.read_text(encoding="utf-8"))
    if len(eligible_manifest) <= args.val_count:
        raise SystemExit(
            f"Need more than val_count={args.val_count} eligible samples, found {len(eligible_manifest)}"
        )

    train_names = [item["name"] for item in eligible_manifest[:-args.val_count]]
    val_names = [item["name"] for item in eligible_manifest[-args.val_count:]]

    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_snapshot.txt").write_text("".join(f"{name}\n" for name in train_names), encoding="utf-8")
    (split_dir / "val_snapshot.txt").write_text("".join(f"{name}\n" for name in val_names), encoding="utf-8")
    (split_dir / "summary.json").write_text(
        json.dumps(
            {
                "prepared_dir": str(prepared_dir),
                "prepared_summary": summary,
                "eligible_total": len(eligible_manifest),
                "train_count": len(train_names),
                "val_count": len(val_names),
                "val_tail": val_names[-10:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[syncnet-split] built split eligible_total={len(eligible_manifest)} "
        f"train={len(train_names)} val={len(val_names)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
