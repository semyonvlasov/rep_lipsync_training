#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT_DIR="output/training_cuda3090_generator_gan_official_20260325"
CONFIG="configs/generator_gan_compare_official_teacher_20260325.yaml"
OFFICIAL_TEACHER="/root/lipsync_test/models/wav2lip/checkpoints/lipsync_expert.pth"
TRAIN_SNAPSHOT="output/training_cuda3090_syncnet_full_minus10_20260321/train_speakers.txt"
HOLDOUT="output/training_cuda3090_syncnet_full_minus10_20260321/holdout_10_speakers.txt"
BASELINE_CKPT="output/training_cuda3090_syncnet_full_minus10_20260321/generator_epoch008.pth"
BASELINE_CONFIG="configs/cuda_3090_generator_full_official_20260321.yaml"

mkdir -p "${OUT_DIR}"

echo "[gangan] start $(date)"
echo "[gangan] config=${CONFIG}"
echo "[gangan] teacher=${OFFICIAL_TEACHER}"
echo "[gangan] train_snapshot=${TRAIN_SNAPSHOT}"
echo "[gangan] holdout=${HOLDOUT}"
echo "[gangan] baseline_checkpoint=${BASELINE_CKPT}"

python3 -u scripts/train_generator.py \
  --config "${CONFIG}" \
  --syncnet "${OFFICIAL_TEACHER}" \
  --speaker-list "${TRAIN_SNAPSHOT}"

LATEST_CKPT="${OUT_DIR}/generator/generator_epoch001.pth"
echo "[gangan] latest_checkpoint=${LATEST_CKPT}"

python3 -u scripts/check_audio_sensitivity.py \
  --checkpoint "${LATEST_CKPT}" \
  --config "${CONFIG}" \
  --syncnet "${OFFICIAL_TEACHER}" \
  --speaker-list "${HOLDOUT}" \
  --samples 32 \
  --seed 0 \
  --output-dir "${OUT_DIR}/eval_official_teacher"

python3 -u scripts/check_audio_sensitivity.py \
  --checkpoint "${BASELINE_CKPT}" \
  --config "${BASELINE_CONFIG}" \
  --syncnet "${OFFICIAL_TEACHER}" \
  --speaker-list "${HOLDOUT}" \
  --samples 32 \
  --seed 0 \
  --output-dir "${OUT_DIR}/baseline_epoch008_eval_official_teacher"

echo "[gangan] done $(date)"
