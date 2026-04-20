#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-rep-lipsync-talkvid-fetch-cpu:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-rep-lipsync-talkvid-fetch-cpu}"
DATA_ROOT="${DATA_ROOT:-$HOME/.cache/rep_lipsync_training/talkvid-fetch}"
RCLONE_CONFIG_PATH="${RCLONE_CONFIG_PATH:-$HOME/.config/rclone/rclone.conf}"
CONFIG_PATH="${CONFIG_PATH:-dataset_prepare/fetch/talkvid/docker/fetch_talkvid_raw_to_gdrive_container_cpu.yaml}"
RCLONE_CONFIG_DIR="$(dirname "$RCLONE_CONFIG_PATH")"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [ ! -f "$RCLONE_CONFIG_PATH" ]; then
  echo "[docker-talkvid-fetch] missing rclone config: $RCLONE_CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/talkvid/logs" "$DATA_ROOT/secrets"

if [ "$SKIP_BUILD" != "1" ]; then
  docker build \
    -f "$REPO_ROOT/dataset_prepare/fetch/talkvid/docker/Dockerfile.cpu" \
    -t "$IMAGE_TAG" \
    "$REPO_ROOT"
fi

docker run --rm --init \
  --name "$CONTAINER_NAME" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO_ROOT:/workspace/repo:ro" \
  -v "$DATA_ROOT:/workspace-data" \
  -v "$RCLONE_CONFIG_DIR:/root/.config/rclone:ro" \
  -w /workspace/repo \
  "$IMAGE_TAG" \
  python3 dataset_prepare/fetch/talkvid/scripts/fetch_talkvid_raw_to_gdrive.py \
    "$CONFIG_PATH" \
    "$@" \
  2>&1 | tee -a "$DATA_ROOT/talkvid/logs/cycle.log"
