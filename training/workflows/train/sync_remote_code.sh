#!/bin/bash
set -euo pipefail

PORT="${PORT:-37576}"
REMOTE="${REMOTE:-root@ssh5.vast.ai}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/lipsync_test/rep_lipsync_training}"
REMOTE_GIT_URL="${REMOTE_GIT_URL:-}"
REMOTE_GIT_BRANCH="${REMOTE_GIT_BRANCH:-main}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PORT")
SSH_CMD=(ssh "${SSH_OPTS[@]}")
RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LEGACY_REPO_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

origin_url="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
if [ -z "$REMOTE_GIT_URL" ]; then
  REMOTE_GIT_URL="$origin_url"
fi

to_https_url() {
  local url="$1"
  if [[ "$url" =~ ^git@github.com:(.+)$ ]]; then
    echo "https://github.com/${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$url" =~ ^ssh://git@github.com/(.+)$ ]]; then
    echo "https://github.com/${BASH_REMATCH[1]}"
    return 0
  fi
  echo "$url"
}

REMOTE_GIT_URL="$(to_https_url "$REMOTE_GIT_URL")"

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
echo "[sync] Git URL: $REMOTE_GIT_URL"
echo "[sync] Git branch: $REMOTE_GIT_BRANCH"

"${SSH_CMD[@]}" "$REMOTE" \
  "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_GIT_URL='$REMOTE_GIT_URL' REMOTE_GIT_BRANCH='$REMOTE_GIT_BRANCH' bash -s" <<'EOS'
set -euo pipefail

preserve_root="$(dirname "$REMOTE_ROOT")/.rep_lipsync_training_preserve"
checkpoint_rel="models/official_syncnet/checkpoints"

ensure_git_checkout() {
  mkdir -p "$(dirname "$REMOTE_ROOT")"

  if [ -d "$REMOTE_ROOT/.git" ]; then
    cd "$REMOTE_ROOT"
    git config --global --add safe.directory "$REMOTE_ROOT"
    git remote set-url origin "$REMOTE_GIT_URL"
    git fetch --depth 1 origin "$REMOTE_GIT_BRANCH"
    git checkout -B "$REMOTE_GIT_BRANCH" "origin/$REMOTE_GIT_BRANCH"
    git reset --hard "origin/$REMOTE_GIT_BRANCH"
    return 0
  fi

  rm -rf "$preserve_root"
  mkdir -p "$preserve_root"

  if [ -d "$REMOTE_ROOT/training/data" ]; then
    mkdir -p "$preserve_root/training"
    mv "$REMOTE_ROOT/training/data" "$preserve_root/training/data"
  fi
  if [ -d "$REMOTE_ROOT/training/output" ]; then
    mkdir -p "$preserve_root/training"
    mv "$REMOTE_ROOT/training/output" "$preserve_root/training/output"
  fi
  if [ -d "$REMOTE_ROOT/$checkpoint_rel" ]; then
    mkdir -p "$preserve_root/models/official_syncnet"
    mv "$REMOTE_ROOT/$checkpoint_rel" "$preserve_root/$checkpoint_rel"
  fi

  rm -rf "$REMOTE_ROOT"
  git clone --branch "$REMOTE_GIT_BRANCH" --depth 1 "$REMOTE_GIT_URL" "$REMOTE_ROOT"
  git config --global --add safe.directory "$REMOTE_ROOT"

  if [ -d "$preserve_root/training/data" ]; then
    rm -rf "$REMOTE_ROOT/training/data"
    mv "$preserve_root/training/data" "$REMOTE_ROOT/training/data"
  fi
  if [ -d "$preserve_root/training/output" ]; then
    rm -rf "$REMOTE_ROOT/training/output"
    mv "$preserve_root/training/output" "$REMOTE_ROOT/training/output"
  fi
  if [ -d "$preserve_root/$checkpoint_rel" ]; then
    mkdir -p "$REMOTE_ROOT/models/official_syncnet"
    rm -rf "$REMOTE_ROOT/$checkpoint_rel"
    mv "$preserve_root/$checkpoint_rel" "$REMOTE_ROOT/$checkpoint_rel"
  fi

  rm -rf "$preserve_root"
}

if ! command -v git >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y git rsync ca-certificates
fi

ensure_git_checkout
mkdir -p "$REMOTE_ROOT/models/official_syncnet/checkpoints"
EOS

if [ "$OFFICIAL_SYNCNET_SKIP_UPLOAD" = "1" ]; then
  echo "[sync] Skipping local official SyncNet checkpoint upload; expecting remote fetch path."
else
  echo "[sync] Uploading official SyncNet checkpoint..."
  rsync -a --whole-file --partial --progress -e "$RSYNC_RSH" \
    "$OFFICIAL_SYNCNET_CKPT" \
    "$REMOTE:$REMOTE_ROOT/models/official_syncnet/checkpoints/"
fi

echo "[sync] Done."
