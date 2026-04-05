#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRAINING_ROOT/.." && pwd)"

CONFIG="$TRAINING_ROOT/configs/syncnet_mirror_cuda3090_medium_init_from_drive_20260404.yaml"
OUTPUT_DIR_REL="output/training_cuda3090_syncnet_mirror_medium_init_from_drive_20260404"
RUN_OUTPUT="$TRAINING_ROOT/$OUTPUT_DIR_REL"
RUN_SYNCNET_DIR="$RUN_OUTPUT/syncnet"
SPLIT_DIR="$TRAINING_ROOT/output/syncnet_mirror_cuda3090_init_from_drive_20260404_split"
TRAIN_SNAPSHOT="$SPLIT_DIR/train_snapshot.txt"
VAL_SNAPSHOT="$SPLIT_DIR/val_snapshot.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VAL_COUNT="${VAL_COUNT:-2048}"
FORCE_REBUILD_SPLIT="${FORCE_REBUILD_SPLIT:-0}"
HDTF_TIERS="${HDTF_TIERS:-confident}"
TALKVID_TIERS="${TALKVID_TIERS:-confident,medium}"
INIT_SYNCNET_CKPT="${INIT_SYNCNET_CKPT:-}"

mkdir -p "$RUN_SYNCNET_DIR" "$SPLIT_DIR"

if [[ "$FORCE_REBUILD_SPLIT" == "1" || ! -f "$TRAIN_SNAPSHOT" || ! -f "$VAL_SNAPSHOT" ]]; then
  export TRAINING_ROOT SPLIT_DIR VAL_COUNT HDTF_TIERS TALKVID_TIERS
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

training_root = Path(os.environ["TRAINING_ROOT"])
split_dir = Path(os.environ["SPLIT_DIR"])
val_count = int(os.environ.get("VAL_COUNT", "2048"))

def parse_tiers(value):
    return [part.strip() for part in value.split(",") if part.strip()]

hdtf_tiers = parse_tiers(os.environ.get("HDTF_TIERS", "confident"))
talkvid_tiers = parse_tiers(os.environ.get("TALKVID_TIERS", "confident,medium"))
roots = []
for tier in hdtf_tiers:
    roots.append(("hdtf", tier, training_root / "data" / "hdtf" / "processed" / "_lazy_imports" / tier))
for tier in talkvid_tiers:
    roots.append(("talkvid", tier, training_root / "data" / "talkvid" / "processed" / "_lazy_imports" / tier))

items = []
source_counts = {}
for source_name, tier_name, root in roots:
    if not root.exists():
        continue
    count = 0
    for json_path in root.rglob("*.json"):
        if json_path.name == "summary.json":
            continue
        items.append((json_path.stat().st_mtime, json_path.stem))
        count += 1
    source_counts[f"{source_name}:{tier_name}"] = count

unique = {}
for mtime, name in sorted(items, key=lambda x: (x[0], x[1])):
    unique[name] = mtime

ordered = sorted(((mtime, name) for name, mtime in unique.items()), key=lambda x: (x[0], x[1]))
if len(ordered) <= val_count:
    raise SystemExit(f"Need more than val_count={val_count} samples, found {len(ordered)}")

train_names = [name for _, name in ordered[:-val_count]]
val_names = [name for _, name in ordered[-val_count:]]

split_dir.mkdir(parents=True, exist_ok=True)
(split_dir / "train_snapshot.txt").write_text("".join(f"{name}\n" for name in train_names))
(split_dir / "val_snapshot.txt").write_text("".join(f"{name}\n" for name in val_names))
(split_dir / "summary.json").write_text(json.dumps({
    "total": len(ordered),
    "train_count": len(train_names),
    "val_count": len(val_names),
    "hdtf_tiers": hdtf_tiers,
    "talkvid_tiers": talkvid_tiers,
    "source_counts": source_counts,
    "val_tail": val_names[-10:],
}, indent=2))
print(f"[syncnet-launcher] built split total={len(ordered)} train={len(train_names)} val={len(val_names)}", flush=True)
PY
fi

cd "$TRAINING_ROOT"
if [[ -n "$INIT_SYNCNET_CKPT" ]]; then
  echo "[syncnet-launcher] init_syncnet_ckpt=$INIT_SYNCNET_CKPT"
  exec "$PYTHON_BIN" "$TRAINING_ROOT/scripts/train_syncnet.py" \
    --config "$CONFIG" \
    --init-model "$INIT_SYNCNET_CKPT" \
    --speaker-list "$TRAIN_SNAPSHOT" \
    --val-speaker-list "$VAL_SNAPSHOT"
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$TRAINING_ROOT/scripts/run_syncnet_resume_from_latest.py" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR_REL" \
  --speaker-list "$TRAIN_SNAPSHOT" \
  --val-speaker-list "$VAL_SNAPSHOT"
