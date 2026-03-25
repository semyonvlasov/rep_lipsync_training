#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-/root/lipsync_test/rep_lipsync_training/training}"
OUT="${OUT:-output/syncnet_hop_ablation_25k_6ep_20260325}"
HOP200_CONFIG="${HOP200_CONFIG:-configs/syncnet_ablation_hop200_25k_6ep_20260325.yaml}"
HOP160_CONFIG="${HOP160_CONFIG:-configs/syncnet_ablation_hop160_25k_6ep_20260325.yaml}"
HOP200_CKPT="${HOP200_CKPT:-output/syncnet_hop200_ablation_25k_6ep_20260325/syncnet/syncnet_epoch005.pth}"
HOP160_CKPT="${HOP160_CKPT:-output/syncnet_hop160_ablation_25k_6ep_20260325/syncnet/syncnet_epoch005.pth}"
HDTF_PROCESSED_ROOT="${HDTF_PROCESSED_ROOT:-data/hdtf/processed}"
TALKVID_PROCESSED_ROOT="${TALKVID_PROCESSED_ROOT:-data/talkvid/processed_medium}"
OFFICIAL_SYNCNET_PATH="${OFFICIAL_SYNCNET_PATH:-../../models/wav2lip/checkpoints/lipsync_expert.pth}"
TRAIN_SELECTION="${TRAIN_SELECTION:-newest}"
MAX_TRAIN_FRAMES="${MAX_TRAIN_FRAMES:-25000}"
SYNCNET_HOLDOUT_COUNT="${SYNCNET_HOLDOUT_COUNT:-20}"
SYNCNET_COMPARE_SAMPLES="${SYNCNET_COMPARE_SAMPLES:-200}"
SYNCNET_COMPARE_SEED="${SYNCNET_COMPARE_SEED:-123}"
SYNCNET_COMPARE_DEVICE="${SYNCNET_COMPARE_DEVICE:-cuda}"
LAZY_CACHE_ROOT="${LAZY_CACHE_ROOT:-data/_lazy_cache_syncnet_hop_ablation_20260325}"
MATERIALIZE_FRAMES_SIZE="${MATERIALIZE_FRAMES_SIZE:-96}"

mkdir -p "$OUT"
LOG="$OUT/pipeline.log"
: > "$LOG"
exec >> "$LOG" 2>&1

cd "$TRAINING_ROOT"

echo "[hop-ablation] started at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[hop-ablation] hop200_config=$HOP200_CONFIG"
echo "[hop-ablation] hop160_config=$HOP160_CONFIG"
echo "[hop-ablation] train_selection=$TRAIN_SELECTION max_train_frames=$MAX_TRAIN_FRAMES holdout_count=$SYNCNET_HOLDOUT_COUNT compare_samples=$SYNCNET_COMPARE_SAMPLES"

TRAIN_LIST="$OUT/train_speakers_25k.txt"
HOLDOUT_LIST="$OUT/holdout_unseen_speakers.txt"
SPLIT_SUMMARY="$OUT/split_summary.json"
COMPARE_JSON="$OUT/syncnet_teacher_compare_200.json"
SELECTED_JSON="$OUT/syncnet_selected_teacher_200.json"

python3 -u scripts/build_syncnet_holdout_split.py \
  --processed-root "$HDTF_PROCESSED_ROOT" \
  --processed-root "$TALKVID_PROCESSED_ROOT" \
  --holdout-count "$SYNCNET_HOLDOUT_COUNT" \
  --max-train-frames "$MAX_TRAIN_FRAMES" \
  --train-selection "$TRAIN_SELECTION" \
  --train-out "$TRAIN_LIST" \
  --holdout-out "$HOLDOUT_LIST" \
  --summary-out "$SPLIT_SUMMARY"

echo "[hop-ablation] training hop=200 at $(date '+%Y-%m-%d %H:%M:%S %Z')"
python3 -u scripts/train_syncnet.py \
  --config "$HOP200_CONFIG" \
  --speaker-list "$TRAIN_LIST"

if [[ ! -f "$HOP200_CKPT" ]]; then
  echo "[hop-ablation] ERROR: expected hop200 checkpoint missing: $HOP200_CKPT" >&2
  exit 1
fi

echo "[hop-ablation] training hop=160 at $(date '+%Y-%m-%d %H:%M:%S %Z')"
python3 -u scripts/train_syncnet.py \
  --config "$HOP160_CONFIG" \
  --speaker-list "$TRAIN_LIST"

if [[ ! -f "$HOP160_CKPT" ]]; then
  echo "[hop-ablation] ERROR: expected hop160 checkpoint missing: $HOP160_CKPT" >&2
  exit 1
fi

echo "[hop-ablation] comparing teachers at $(date '+%Y-%m-%d %H:%M:%S %Z')"
python3 -u scripts/compare_syncnet_teachers.py \
  --processed-root "$HDTF_PROCESSED_ROOT" \
  --processed-root "$TALKVID_PROCESSED_ROOT" \
  --speaker-snapshot "$TRAIN_LIST" \
  --speaker-list "$HOLDOUT_LIST" \
  --official-checkpoint "$OFFICIAL_SYNCNET_PATH" \
  --checkpoints "$HOP200_CKPT" "$HOP160_CKPT" \
  --output "$COMPARE_JSON" \
  --samples "$SYNCNET_COMPARE_SAMPLES" \
  --seed "$SYNCNET_COMPARE_SEED" \
  --device "$SYNCNET_COMPARE_DEVICE" \
  --fps 25 \
  --T 5 \
  --cache-size 16 \
  --lazy-cache-root "$LAZY_CACHE_ROOT" \
  --materialize-frames-size "$MATERIALIZE_FRAMES_SIZE"

echo "[hop-ablation] selecting teacher at $(date '+%Y-%m-%d %H:%M:%S %Z')"
python3 -u scripts/select_best_syncnet_teacher.py \
  --compare-json "$COMPARE_JSON" \
  --official-checkpoint "$OFFICIAL_SYNCNET_PATH" \
  --checkpoints "$HOP200_CKPT" "$HOP160_CKPT" \
  --output "$SELECTED_JSON"

echo "[hop-ablation] finished at $(date '+%Y-%m-%d %H:%M:%S %Z')"
