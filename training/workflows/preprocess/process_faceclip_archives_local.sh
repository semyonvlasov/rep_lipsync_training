#!/bin/bash
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$TRAINING_ROOT"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

DATA_ROOT="${FACECLIP_DATA_ROOT:-$TRAINING_ROOT/data/raw_faceclips_x288_cycle}"
OUT_DIR="${FACECLIP_OUT_DIR:-$TRAINING_ROOT/output/raw_faceclips_x288_cycle}"
MANIFEST_PATH="${FACECLIP_MANIFEST_PATH:-$OUT_DIR/archive_manifest.jsonl}"
LOG_FILE="$OUT_DIR/cycle.log"
mkdir -p "$DATA_ROOT" "$OUT_DIR"

SOURCE_FOLDER_ID="${FACECLIP_SOURCE_FOLDER_ID:-1v06momk8fR-eqw79Z93zczBx_InWsCS9}"
DEST_FOLDER_ID="${FACECLIP_DEST_FOLDER_ID:-1xx2IlfiAYC1AFf3xJwcjeTEsAK-Uqt8n}"
REMOTE_NAME="${FACECLIP_GDRIVE_REMOTE:-gdrive:}"
PYTHON_BIN="${FACECLIP_PYTHON_BIN:-python3}"

IMG_SIZE="${FACECLIP_SIZE:-288}"
FPS="${FACECLIP_FPS:-25}"
MAX_FRAMES="${FACECLIP_MAX_FRAMES:-250}"
DETECT_EVERY="${FACECLIP_DETECT_EVERY:-1}"
SMOOTH_WINDOW="${FACECLIP_SMOOTH_WINDOW:-5}"
SMOOTHING_STYLE="${FACECLIP_SMOOTHING_STYLE:-official_inference}"
FRAMING_STYLE="${FACECLIP_FRAMING_STYLE:-official_inference}"
DETECTOR_BACKEND="${FACECLIP_DETECTOR_BACKEND:-sfd}"
DETECTOR_DEVICE="${FACECLIP_DETECTOR_DEVICE:-auto}"
DETECTOR_BATCH_SIZE="${FACECLIP_DETECTOR_BATCH_SIZE:-4}"
MIN_DETECTOR_SCORE="${FACECLIP_MIN_DETECTOR_SCORE:-0.99999}"
RESIZE_DEVICE="${FACECLIP_RESIZE_DEVICE:-auto}"
FFMPEG_BIN="${FACECLIP_FFMPEG_BIN:-}"
FFMPEG_THREADS="${FACECLIP_FFMPEG_THREADS:-4}"
FFMPEG_TIMEOUT="${FACECLIP_FFMPEG_TIMEOUT:-180}"
VIDEO_ENCODER="${FACECLIP_VIDEO_ENCODER:-auto}"
VIDEO_BITRATE="${FACECLIP_VIDEO_BITRATE:-600k}"
NORMALIZED_VIDEO_BITRATE="${FACECLIP_NORMALIZED_VIDEO_BITRATE:-15m}"
MAX_ARCHIVES="${FACECLIP_MAX_ARCHIVES:-0}"
ARCHIVE_GLOB="${FACECLIP_ARCHIVE_GLOB:-*.tar}"

exec >>"$LOG_FILE" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log_cycle() {
  echo "$(timestamp) [faceclip-cycle] $*"
}

log_exit() {
  local rc=$?
  echo "$(timestamp) [faceclip-cycle] exit rc=$rc"
}
trap log_exit EXIT

log_cycle "start"
log_cycle "data_root=$DATA_ROOT"
log_cycle "out_dir=$OUT_DIR"
log_cycle "source_folder_id=$SOURCE_FOLDER_ID"
log_cycle "dest_folder_id=$DEST_FOLDER_ID"
log_cycle "remote=$REMOTE_NAME archive_glob=$ARCHIVE_GLOB max_archives=$MAX_ARCHIVES"
log_cycle "detector=$DETECTOR_BACKEND/$DETECTOR_DEVICE detector_batch_size=$DETECTOR_BATCH_SIZE min_detector_score=$MIN_DETECTOR_SCORE resize_device=$RESIZE_DEVICE"
log_cycle "framing=$FRAMING_STYLE smoothing=$SMOOTHING_STYLE smooth_window=$SMOOTH_WINDOW detect_every=$DETECT_EVERY size=$IMG_SIZE"
log_cycle "video_encoder=$VIDEO_ENCODER normalized_video_bitrate=$NORMALIZED_VIDEO_BITRATE video_bitrate=$VIDEO_BITRATE ffmpeg_threads=$FFMPEG_THREADS"

"$PYTHON_BIN" scripts/process_faceclip_archives_from_gdrive.py \
  --source-folder-id "$SOURCE_FOLDER_ID" \
  --dest-folder-id "$DEST_FOLDER_ID" \
  --remote "$REMOTE_NAME" \
  --data-root "$DATA_ROOT" \
  --manifest-path "$MANIFEST_PATH" \
  --archive-glob "$ARCHIVE_GLOB" \
  --max-archives "$MAX_ARCHIVES" \
  --python-bin "$PYTHON_BIN" \
  --size "$IMG_SIZE" \
  --fps "$FPS" \
  --max-frames "$MAX_FRAMES" \
  --detect-every "$DETECT_EVERY" \
  --smooth-window "$SMOOTH_WINDOW" \
  --smoothing-style "$SMOOTHING_STYLE" \
  --framing-style "$FRAMING_STYLE" \
  --detector-backend "$DETECTOR_BACKEND" \
  --detector-device "$DETECTOR_DEVICE" \
  --detector-batch-size "$DETECTOR_BATCH_SIZE" \
  --min-detector-score "$MIN_DETECTOR_SCORE" \
  --resize-device "$RESIZE_DEVICE" \
  --ffmpeg-bin "$FFMPEG_BIN" \
  --ffmpeg-threads "$FFMPEG_THREADS" \
  --ffmpeg-timeout "$FFMPEG_TIMEOUT" \
  --video-encoder "$VIDEO_ENCODER" \
  --normalized-video-bitrate "$NORMALIZED_VIDEO_BITRATE" \
  --video-bitrate "$VIDEO_BITRATE"

log_cycle "done"
