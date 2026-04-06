#!/bin/bash
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$TRAINING_ROOT"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${HDTF_DATA_ROOT:-$TRAINING_ROOT/data/hdtf}"
ARCHIVES_DIR="${HDTF_ARCHIVES_DIR:-$DATA_ROOT/archives_raw_batches}"
REMOTE_NAME="${GDRIVE_REMOTE:-gdrive:}"
FOLDER_ID="${GDRIVE_FOLDER_ID:-1v06momk8fR-eqw79Z93zczBx_InWsCS9}"
MAX_VIDEOS="${HDTF_MAX_VIDEOS:-0}"
TARGET_SIZE_GB="${HDTF_TARGET_SIZE_GB:-0}"
MAX_HEIGHT="${HDTF_MAX_HEIGHT:-720}"
BATCH_SIZE_GB="${HDTF_BATCH_SIZE_GB:-1.0}"
MAX_BATCHES="${HDTF_MAX_BATCHES:-0}"

mkdir -p "$DATA_ROOT" "$ARCHIVES_DIR"

echo "[hdtf-fetch] download start $(date '+%Y-%m-%d %H:%M:%S %Z')"
"$PYTHON" scripts/download_hdtf.py \
  --output "$DATA_ROOT" \
  --max-videos "$MAX_VIDEOS" \
  --target-size-gb "$TARGET_SIZE_GB" \
  --max-height "$MAX_HEIGHT"

echo "[hdtf-fetch] package start $(date '+%Y-%m-%d %H:%M:%S %Z')"
"$PYTHON" scripts/package_media_batches.py \
  --input-dir "$DATA_ROOT/raw" \
  --archives-dir "$ARCHIVES_DIR" \
  --prefix hdtf_raw \
  --pattern '*.mp4' \
  --max-gb "$BATCH_SIZE_GB" \
  --max-batches "$MAX_BATCHES" \
  --allow-partial-tail

echo "[hdtf-fetch] upload start $(date '+%Y-%m-%d %H:%M:%S %Z')"
"$PYTHON" scripts/upload_archives_no_cleanup.py \
  --archives-dir "$ARCHIVES_DIR" \
  --remote "$REMOTE_NAME" \
  --drive-root-folder-id "$FOLDER_ID" \
  --prefix hdtf_raw \
  --max-batches 0

echo "[hdtf-fetch] done $(date '+%Y-%m-%d %H:%M:%S %Z')"
