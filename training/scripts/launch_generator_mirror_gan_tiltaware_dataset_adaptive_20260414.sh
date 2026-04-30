#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="${CONFIG:-$TRAINING_ROOT/configs/generator_mirror_gan_tiltaware_dataset_adaptive_20260414.yaml}"
OUTPUT_DIR_REL="output/generator_mirror_gan_tiltaware_dataset_adaptive_20260414"
SPLIT_DIR="$TRAINING_ROOT/output/generator_mirror_gan_tiltaware_dataset_adaptive_20260414_split"
PREPARED_DIR="$TRAINING_ROOT/output/generator_mirror_gan_tiltaware_dataset_adaptive_20260414_prepared"
SPEAKER_LIST="$SPLIT_DIR/train_snapshot.txt"
VAL_SPEAKER_LIST="$SPLIT_DIR/val_snapshot.txt"
VAL_SNAPSHOT_SOURCE="${VAL_SNAPSHOT_SOURCE:-$TRAINING_ROOT/snapshots/val_snapshot_talkvid_balanced_eth_gender_20260408.txt}"
SYNCNET="${SYNCNET_CKPT:-$TRAINING_ROOT/output/syncnet_current_best_20260428/syncnet_best_our_eval.pth}"
EVAL_SEED="${EVAL_SEED:-20260408}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HDTF_TIERS="${HDTF_TIERS:-confident}"
TALKVID_TIERS="${TALKVID_TIERS:-confident,medium}"
HDTF_ROOT_REL="${HDTF_ROOT_REL:-data/hdtf/processed_tilt_aware_20260414}"
TALKVID_ROOT_REL="${TALKVID_ROOT_REL:-data/talkvid/processed_tilt_aware_20260414}"
FORCE_REBUILD_PREPARED="${FORCE_REBUILD_PREPARED:-0}"
FORCE_REBUILD_SPLIT="${FORCE_REBUILD_SPLIT:-0}"

mkdir -p "$SPLIT_DIR"

if [[ "$FORCE_REBUILD_PREPARED" == "1" || ! -f "$PREPARED_DIR/eligible_manifest.json" ]]; then
  "$PYTHON_BIN" "$TRAINING_ROOT/scripts/prepare_syncnet_dataset.py" \
    --config "$CONFIG" \
    --prepared-dir "$PREPARED_DIR" \
    --hdtf-tiers "$HDTF_TIERS" \
    --talkvid-tiers "$TALKVID_TIERS" \
    --hdtf-root-rel "$HDTF_ROOT_REL" \
    --talkvid-root-rel "$TALKVID_ROOT_REL"
fi

if [[ "$FORCE_REBUILD_SPLIT" == "1" || ! -f "$SPEAKER_LIST" || ! -f "$VAL_SPEAKER_LIST" ]]; then
  "$PYTHON_BIN" "$TRAINING_ROOT/scripts/build_generator_explicit_val_split.py" \
    --prepared-dir "$PREPARED_DIR" \
    --split-dir "$SPLIT_DIR" \
    --explicit-val-snapshot "$VAL_SNAPSHOT_SOURCE"
fi

cd "$TRAINING_ROOT"
exec "$PYTHON_BIN" "$TRAINING_ROOT/scripts/train_generator_mirror_gan.py" \
  --config "$CONFIG" \
  --syncnet "$SYNCNET" \
  --eval-seed "$EVAL_SEED" \
  --speaker-list "$SPEAKER_LIST" \
  --val-speaker-list "$VAL_SPEAKER_LIST"
