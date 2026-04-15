#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRAINING_ROOT/.." && pwd)"

CONFIG="$TRAINING_ROOT/configs/syncnet_mirror_cuda3090_medium.yaml"
OUTPUT_DIR_REL="output/training_cuda3090_syncnet_mirror_medium_20260329"
RUN_OUTPUT="$TRAINING_ROOT/$OUTPUT_DIR_REL"
RUN_SYNCNET_DIR="$RUN_OUTPUT/syncnet"
SPLIT_DIR="$TRAINING_ROOT/output/syncnet_mirror_medium_remote2_20260329_split"
PREPARED_DIR="$TRAINING_ROOT/output/syncnet_mirror_medium_remote2_20260329_prepared"
TRAIN_SNAPSHOT="$SPLIT_DIR/train_snapshot.txt"
VAL_SNAPSHOT="$SPLIT_DIR/val_snapshot.txt"
SPLIT_SUMMARY="$SPLIT_DIR/summary.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VAL_COUNT="${VAL_COUNT:-2048}"
FORCE_REBUILD_SPLIT="${FORCE_REBUILD_SPLIT:-0}"
HDTF_TIERS="${HDTF_TIERS:-confident}"
TALKVID_TIERS="${TALKVID_TIERS:-confident,medium}"

mkdir -p "$RUN_SYNCNET_DIR" "$SPLIT_DIR"

if [[ "$FORCE_REBUILD_SPLIT" == "1" || ! -f "$TRAIN_SNAPSHOT" || ! -f "$VAL_SNAPSHOT" ]]; then
  "$PYTHON_BIN" "$TRAINING_ROOT/scripts/prepare_syncnet_dataset.py" \
    --config "$CONFIG" \
    --prepared-dir "$PREPARED_DIR" \
    --hdtf-tiers "$HDTF_TIERS" \
    --talkvid-tiers "$TALKVID_TIERS"
  "$PYTHON_BIN" "$TRAINING_ROOT/scripts/build_syncnet_split.py" \
    --prepared-dir "$PREPARED_DIR" \
    --split-dir "$SPLIT_DIR" \
    --val-count "$VAL_COUNT"
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$TRAINING_ROOT/scripts/run_syncnet_resume_from_latest.py" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR_REL" \
  --speaker-list "$TRAIN_SNAPSHOT" \
  --val-speaker-list "$VAL_SNAPSHOT"
