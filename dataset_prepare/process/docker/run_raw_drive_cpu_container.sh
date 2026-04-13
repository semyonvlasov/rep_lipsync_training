#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-rep-lipsync-process-cpu:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-rep-lipsync-process-cpu}"
DATA_ROOT="${DATA_ROOT:-$HOME/.cache/rep_lipsync_training/process-docker}"
RCLONE_CONFIG_PATH="${RCLONE_CONFIG_PATH:-$HOME/.config/rclone/rclone.conf}"
CONFIG_PATH="${CONFIG_PATH:-dataset_prepare/process/docker/process_raw_archives_to_lazy_faceclips_gdrive_container_cpu.yaml}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/face_processing/face_landmarker_v2_with_blendshapes.task}"
RCLONE_CONFIG_DIR="$(dirname "$RCLONE_CONFIG_PATH")"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [ ! -f "$RCLONE_CONFIG_PATH" ]; then
  echo "[docker-process] missing rclone config: $RCLONE_CONFIG_PATH" >&2
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "[docker-process] missing face_processing model: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/process/logs"

if [ "$SKIP_BUILD" != "1" ]; then
  docker build \
    -f "$REPO_ROOT/dataset_prepare/process/docker/Dockerfile.cpu" \
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
  python3 dataset_prepare/process/common/process_raw_archives_to_lazy_faceclips_gdrive.py \
    "$CONFIG_PATH" \
    "$@" \
  2>&1 | tee -a "$DATA_ROOT/process/logs/raw_drive.log"
