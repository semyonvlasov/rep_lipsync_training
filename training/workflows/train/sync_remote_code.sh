#!/bin/bash
set -euo pipefail

PORT="${PORT:-37576}"
REMOTE="${REMOTE:-root@ssh5.vast.ai}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/lipsync_test}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PORT")
SSH_CMD=(ssh "${SSH_OPTS[@]}")
RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

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

echo "[sync] Uploading official SyncNet reference code + checkpoint..."
rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
  "$REPO_ROOT/models/official_syncnet/models/" \
  "$REMOTE:$REMOTE_ROOT/models/official_syncnet/models/"

rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
  "$REPO_ROOT/models/official_syncnet/checkpoints/lipsync_expert.pth" \
  "$REMOTE:$REMOTE_ROOT/models/official_syncnet/checkpoints/"

echo "[sync] Done."
