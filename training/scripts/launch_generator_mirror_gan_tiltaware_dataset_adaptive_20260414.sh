#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="${CONFIG:-$TRAINING_ROOT/configs/generator_mirror_gan_tiltaware_dataset_adaptive_20260414.yaml}"
OUTPUT_DIR_REL="output/generator_mirror_gan_tiltaware_dataset_adaptive_20260414"
SPLIT_DIR="$TRAINING_ROOT/output/generator_mirror_gan_tiltaware_dataset_adaptive_20260414_split"
SPEAKER_LIST="$SPLIT_DIR/train_snapshot.txt"
VAL_SPEAKER_LIST="$SPLIT_DIR/val_snapshot.txt"
VAL_SNAPSHOT_SOURCE="${VAL_SNAPSHOT_SOURCE:-$TRAINING_ROOT/snapshots/val_snapshot_talkvid_balanced_eth_gender_20260408.txt}"
SYNCNET="${SYNCNET_CKPT:-$TRAINING_ROOT/output/syncnet_current_best_20260406/syncnet_best_off_eval.pth}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HDTF_TIERS="${HDTF_TIERS:-confident}"
TALKVID_TIERS="${TALKVID_TIERS:-confident,medium}"
HDTF_ROOT_REL="${HDTF_ROOT_REL:-data/hdtf/processed_tilt_aware_20260414}"
TALKVID_ROOT_REL="${TALKVID_ROOT_REL:-data/talkvid/processed_tilt_aware_20260414}"

mkdir -p "$SPLIT_DIR"

export TRAINING_ROOT SPLIT_DIR HDTF_TIERS TALKVID_TIERS VAL_SNAPSHOT_SOURCE HDTF_ROOT_REL TALKVID_ROOT_REL
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

training_root = Path(os.environ["TRAINING_ROOT"])
split_dir = Path(os.environ["SPLIT_DIR"])
val_snapshot_source = Path(os.environ["VAL_SNAPSHOT_SOURCE"])

def parse_tiers(value: str):
    return [part.strip() for part in value.split(",") if part.strip()]

hdtf_tiers = parse_tiers(os.environ.get("HDTF_TIERS", "confident"))
talkvid_tiers = parse_tiers(os.environ.get("TALKVID_TIERS", "confident,medium"))

if not val_snapshot_source.exists():
    raise SystemExit(f"Missing explicit val snapshot: {val_snapshot_source}")

explicit_val_names = [line.strip() for line in val_snapshot_source.read_text(encoding="utf-8").splitlines() if line.strip()]
explicit_val_set = set(explicit_val_names)
if len(explicit_val_set) != len(explicit_val_names):
    raise SystemExit(f"Explicit val snapshot contains duplicates: {val_snapshot_source}")

roots = []
for tier in hdtf_tiers:
    roots.append(("hdtf", tier, training_root / os.environ["HDTF_ROOT_REL"] / "_lazy_imports" / tier))
for tier in talkvid_tiers:
    roots.append(("talkvid", tier, training_root / os.environ["TALKVID_ROOT_REL"] / "_lazy_imports" / tier))

items = []
source_counts = {}
for source_name, tier_name, root in roots:
    if not root.exists():
        continue
    count = 0
    for json_path in root.rglob("*.json"):
        if json_path.name.endswith(".detections.json") or json_path.name == "summary.json":
            continue
        items.append((json_path.stat().st_mtime, json_path.stem))
        count += 1
    source_counts[f"{source_name}:{tier_name}"] = count

unique = {}
for mtime, name in sorted(items, key=lambda x: (x[0], x[1])):
    unique[name] = mtime

ordered = sorted(((mtime, name) for name, mtime in unique.items()), key=lambda x: (x[0], x[1]))
available_names = {name for _, name in ordered}
missing_val = [name for name in explicit_val_names if name not in available_names]
if missing_val:
    raise SystemExit(
        f"Explicit val snapshot contains {len(missing_val)} ids that are missing from current processed roots. "
        f"First missing ids: {missing_val[:10]}"
    )

train_names = [name for _, name in ordered if name not in explicit_val_set]
val_names = list(explicit_val_names)

split_dir.mkdir(parents=True, exist_ok=True)
(split_dir / "train_snapshot.txt").write_text("".join(f"{name}\n" for name in train_names), encoding="utf-8")
(split_dir / "val_snapshot.txt").write_text("".join(f"{name}\n" for name in val_names), encoding="utf-8")
(split_dir / "summary.json").write_text(json.dumps({
    "mode": "explicit_balanced_talkvid_val",
    "total": len(ordered),
    "train_count": len(train_names),
    "val_count": len(val_names),
    "hdtf_root": os.environ["HDTF_ROOT_REL"],
    "talkvid_root": os.environ["TALKVID_ROOT_REL"],
    "hdtf_tiers": hdtf_tiers,
    "talkvid_tiers": talkvid_tiers,
    "source_counts": source_counts,
    "explicit_val_snapshot": str(val_snapshot_source),
    "val_tail": val_names[-10:],
}, indent=2), encoding="utf-8")
print(
    f"[generator-mirror-gan-launcher] built split total={len(ordered)} "
    f"train={len(train_names)} val={len(val_names)}",
    flush=True,
)
PY

cd "$TRAINING_ROOT"
exec "$PYTHON_BIN" "$TRAINING_ROOT/scripts/train_generator_mirror_gan.py" \
  --config "$CONFIG" \
  --syncnet "$SYNCNET" \
  --speaker-list "$SPEAKER_LIST" \
  --val-speaker-list "$VAL_SPEAKER_LIST"
