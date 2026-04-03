#!/bin/bash
set -euo pipefail

PORT="${PORT:-37576}"
REMOTE="${REMOTE:-root@ssh5.vast.ai}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/lipsync_test/rep_lipsync_training}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PORT")
SSH_CMD=(ssh "${SSH_OPTS[@]}")
RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LEGACY_REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
OFFICIAL_SYNCNET_CKPT="${OFFICIAL_SYNCNET_CKPT:-$REPO_ROOT/models/official_syncnet/checkpoints/lipsync_expert.pth}"
OFFICIAL_SYNCNET_SKIP_UPLOAD="${OFFICIAL_SYNCNET_SKIP_UPLOAD:-0}"
if [ ! -f "$OFFICIAL_SYNCNET_CKPT" ]; then
  FALLBACK_CKPT="$LEGACY_REPO_ROOT/models/wav2lip/checkpoints/lipsync_expert.pth"
  if [ -f "$FALLBACK_CKPT" ]; then
    OFFICIAL_SYNCNET_CKPT="$FALLBACK_CKPT"
  fi
fi
if [ "$OFFICIAL_SYNCNET_SKIP_UPLOAD" != "1" ] && [ ! -f "$OFFICIAL_SYNCNET_CKPT" ]; then
  echo "[sync] ERROR: official SyncNet checkpoint not found." >&2
  echo "[sync] Checked: $REPO_ROOT/models/official_syncnet/checkpoints/lipsync_expert.pth" >&2
  echo "[sync] Fallback: $LEGACY_REPO_ROOT/models/wav2lip/checkpoints/lipsync_expert.pth" >&2
  exit 1
fi

echo "[sync] Remote: $REMOTE:$REMOTE_ROOT"
"${SSH_CMD[@]}" "$REMOTE" "
  set -euo pipefail
  mkdir -p '$REMOTE_ROOT/training' '$REMOTE_ROOT/models/official_syncnet/checkpoints' '$REMOTE_ROOT/models/official_syncnet/models'
  if ! command -v rsync >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y rsync
  fi
"

echo "[sync] Uploading training/ code only (excluding local data/output)..."
rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
  --exclude 'output/' \
  --exclude 'data/' \
  --exclude 'backup/' \
  --exclude 'exported_dummy/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  "$REPO_ROOT/training/" \
  "$REMOTE:$REMOTE_ROOT/training/"

echo "[sync] Uploading training/data python package..."
rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
  "$REPO_ROOT/training/data/__init__.py" \
  "$REPO_ROOT/training/data/audio.py" \
  "$REPO_ROOT/training/data/dataset.py" \
  "$REMOTE:$REMOTE_ROOT/training/data/"

echo "[sync] Uploading top-level Makefile..."
rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
  "$REPO_ROOT/Makefile" \
  "$REMOTE:$REMOTE_ROOT/"

echo "[sync] Uploading official SyncNet reference code + checkpoint..."
rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
  "$REPO_ROOT/models/official_syncnet/models/" \
  "$REMOTE:$REMOTE_ROOT/models/official_syncnet/models/"

echo "[sync] Uploading official SyncNet face_detection package for SFD..."
rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
  "$REPO_ROOT/models/official_syncnet/face_detection/" \
  "$REMOTE:$REMOTE_ROOT/models/official_syncnet/face_detection/"

if [ "$OFFICIAL_SYNCNET_SKIP_UPLOAD" = "1" ]; then
  echo "[sync] Skipping local official SyncNet checkpoint upload; expecting remote fetch path."
else
  rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
    "$OFFICIAL_SYNCNET_CKPT" \
    "$REMOTE:$REMOTE_ROOT/models/official_syncnet/checkpoints/"
fi

echo "[sync] Done."
