#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$TRAINING_ROOT"

TRAIN_SNAPSHOT="output/training_syncnet_cuda3090_medium/train_speakers.txt"
HOLDOUT="output/training_syncnet_cuda3090_medium/holdout_10_speakers.txt"
OFFICIAL_TEACHER="/root/rep_lipsync_training/models/official_syncnet/checkpoints/lipsync_expert.pth"
LOCAL09_TEACHER="/root/lipsync_test/training/output/training_cuda3090_syncnet_partial_20260321/syncnet/syncnet_epoch009.pth"
LOCAL11_TEACHER="/root/lipsync_test/training/output/training_cuda3090_syncnet_full_minus10_20260321/syncnet/syncnet_epoch011.pth"

echo "[ablate3] start $(date)"
echo "[ablate3] train_snapshot=${TRAIN_SNAPSHOT}"
echo "[ablate3] holdout=${HOLDOUT}"

echo "[ablate3] official teacher train"
python3 -u scripts/train_generator.py \
  --config configs/generator_compare_official_teacher.yaml \
  --syncnet "${OFFICIAL_TEACHER}" \
  --speaker-list "${TRAIN_SNAPSHOT}" \
  > output/training_generator_compare_official_teacher/train.log 2>&1

echo "[ablate3] official teacher eval by official syncnet"
python3 -u scripts/check_audio_sensitivity.py \
  --checkpoint output/training_generator_compare_official_teacher/generator/generator_epoch001.pth \
  --config configs/generator_compare_official_teacher.yaml \
  --syncnet "${OFFICIAL_TEACHER}" \
  --speaker-list "${HOLDOUT}" \
  --samples 16 \
  --seed 0 \
  --output-dir output/training_generator_compare_official_teacher/eval_official_teacher \
  > output/training_generator_compare_official_teacher/eval_official_teacher.log 2>&1

echo "[ablate3] local09 teacher train"
python3 -u scripts/train_generator.py \
  --config configs/generator_compare_local_teacher_epoch009.yaml \
  --syncnet "${LOCAL09_TEACHER}" \
  --speaker-list "${TRAIN_SNAPSHOT}" \
  > output/training_generator_compare_local_teacher_epoch009/train.log 2>&1

echo "[ablate3] local09 eval by official syncnet"
python3 -u scripts/check_audio_sensitivity.py \
  --checkpoint output/training_generator_compare_local_teacher_epoch009/generator/generator_epoch001.pth \
  --config configs/generator_compare_local_teacher_epoch009.yaml \
  --syncnet "${OFFICIAL_TEACHER}" \
  --speaker-list "${HOLDOUT}" \
  --samples 16 \
  --seed 0 \
  --output-dir output/training_generator_compare_local_teacher_epoch009/eval_official_teacher \
  > output/training_generator_compare_local_teacher_epoch009/eval_official_teacher.log 2>&1

echo "[ablate3] local09 eval by local09 syncnet"
python3 -u scripts/check_audio_sensitivity.py \
  --checkpoint output/training_generator_compare_local_teacher_epoch009/generator/generator_epoch001.pth \
  --config configs/generator_compare_local_teacher_epoch009.yaml \
  --syncnet "${LOCAL09_TEACHER}" \
  --speaker-list "${HOLDOUT}" \
  --samples 16 \
  --seed 0 \
  --output-dir output/training_generator_compare_local_teacher_epoch009/eval_local_teacher \
  > output/training_generator_compare_local_teacher_epoch009/eval_local_teacher.log 2>&1

echo "[ablate3] local11 teacher train"
python3 -u scripts/train_generator.py \
  --config configs/generator_compare_local_teacher_epoch011.yaml \
  --syncnet "${LOCAL11_TEACHER}" \
  --speaker-list "${TRAIN_SNAPSHOT}" \
  > output/training_generator_compare_local_teacher_epoch011/train.log 2>&1

echo "[ablate3] local11 eval by official syncnet"
python3 -u scripts/check_audio_sensitivity.py \
  --checkpoint output/training_generator_compare_local_teacher_epoch011/generator/generator_epoch001.pth \
  --config configs/generator_compare_local_teacher_epoch011.yaml \
  --syncnet "${OFFICIAL_TEACHER}" \
  --speaker-list "${HOLDOUT}" \
  --samples 16 \
  --seed 0 \
  --output-dir output/training_generator_compare_local_teacher_epoch011/eval_official_teacher \
  > output/training_generator_compare_local_teacher_epoch011/eval_official_teacher.log 2>&1

echo "[ablate3] local11 eval by local11 syncnet"
python3 -u scripts/check_audio_sensitivity.py \
  --checkpoint output/training_generator_compare_local_teacher_epoch011/generator/generator_epoch001.pth \
  --config configs/generator_compare_local_teacher_epoch011.yaml \
  --syncnet "${LOCAL11_TEACHER}" \
  --speaker-list "${HOLDOUT}" \
  --samples 16 \
  --seed 0 \
  --output-dir output/training_generator_compare_local_teacher_epoch011/eval_local_teacher \
  > output/training_generator_compare_local_teacher_epoch011/eval_local_teacher.log 2>&1

echo "[ablate3] done $(date)"
