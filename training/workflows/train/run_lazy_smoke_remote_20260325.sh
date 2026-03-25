#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-/root/lipsync_test/rep_lipsync_training/training}"
SYNC_CFG="${SYNC_CFG:-configs/cuda_3090_syncnet_lazy_smoke_20260325.yaml}"
GEN_CFG="${GEN_CFG:-configs/cuda_3090_generator_lazy_smoke_20260325.yaml}"
SYNC_CKPT="${SYNC_CKPT:-output/lazy_smoke_20260325/syncnet/syncnet_epoch001.pth}"

cd "$TRAINING_ROOT"

python3 scripts/train_syncnet.py --config "$SYNC_CFG"

if [[ ! -f "$SYNC_CKPT" ]]; then
  echo "Expected SyncNet checkpoint missing: $SYNC_CKPT" >&2
  exit 1
fi

python3 scripts/train_generator.py --config "$GEN_CFG" --syncnet "$SYNC_CKPT"
