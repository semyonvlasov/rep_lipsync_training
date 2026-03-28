#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRAINING_ROOT/.." && pwd)"

CONFIG="$TRAINING_ROOT/configs/generator_official_syncnet_continue_from_ablation_winner_20260328.yaml"
OUTPUT_DIR_REL="output/generator_medium_x96_b32_official_syncnet_continue_from_ablation_winner_20260328"
OUTPUT_DIR="$TRAINING_ROOT/$OUTPUT_DIR_REL"
GENERATOR_DIR="$OUTPUT_DIR/generator"
INITIAL_RESUME="$TRAINING_ROOT/output/generator_sync_weight_ablation_e1_20260328/official_w0p010/generator/generator_epoch002.pth"
SPEAKER_LIST="$TRAINING_ROOT/output/generator_sync_weight_ablation_e1_20260328/train_confident_medium_allowlist_clean.txt"
VAL_SPEAKER_LIST="$TRAINING_ROOT/output/syncnet_medium_x96_b64_lazy_full_20260325/val_confident_medium_allowlist_officialish.txt"
SYNCNET="/root/lipsync_test/models/wav2lip/checkpoints/lipsync_expert.pth"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$GENERATOR_DIR"

# Seed the rolling latest checkpoint so an early reboot still has a valid resume source.
if [[ ! -f "$GENERATOR_DIR/generator_latest.pth" && -f "$INITIAL_RESUME" ]]; then
  cp -n "$INITIAL_RESUME" "$GENERATOR_DIR/generator_latest.pth"
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$TRAINING_ROOT/scripts/run_generator_resume_from_latest.py" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR_REL" \
  --syncnet "$SYNCNET" \
  --initial-resume "$INITIAL_RESUME" \
  --speaker-list "$SPEAKER_LIST" \
  --val-speaker-list "$VAL_SPEAKER_LIST"
