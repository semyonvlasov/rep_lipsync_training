#!/bin/zsh
set -euo pipefail
unsetopt BG_NICE

TRAINING_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$TRAINING_ROOT"

DATA_ROOT="${TALKVID_DATA_ROOT:-$TRAINING_ROOT/data/talkvid_local}"
BATCH_RUNS_DIR="$DATA_ROOT/batches"
PROCESS_ROOT="${TALKVID_PROCESS_ROOT:-$DATA_ROOT/faceclip_pipeline}"
OUT_DIR="${TALKVID_OUT_DIR:-$TRAINING_ROOT/output/talkvid_local_cycle}"
GLOBAL_MANIFEST="$DATA_ROOT/download_manifest.jsonl"
PROCESS_MANIFEST="${TALKVID_PROCESS_MANIFEST:-$OUT_DIR/processed_batch_manifest.jsonl}"
BATCH_COUNTER_FILE="$DATA_ROOT/next_fetch_batch_index.txt"
PROCESS_DONE_FLAG="$OUT_DIR/fetch_done.flag"
mkdir -p "$BATCH_RUNS_DIR" "$PROCESS_ROOT" "$OUT_DIR"

LOG_FILE="$OUT_DIR/cycle.log"
DEST_FOLDER_ID="${GDRIVE_FOLDER_ID:-1xx2IlfiAYC1AFf3xJwcjeTEsAK-Uqt8n}"
REMOTE_NAME="${GDRIVE_REMOTE:-gdrive:}"
COOKIES_FILE="${YTDLP_COOKIES_FILE:-/tmp/talkvid_local_cookies.txt}"
COOKIES_FROM_BROWSER="${YTDLP_COOKIES_FROM_BROWSER:-}"
BATCH_GB="${TALKVID_BATCH_GB:-1.0}"
FETCH_TARGET_GB="${TALKVID_TARGET_ADDITIONAL_GB:-$BATCH_GB}"
DOWNLOAD_JOBS="${TALKVID_DOWNLOAD_JOBS:-4}"
RATE_LIMIT_COOLDOWN_SECONDS="${TALKVID_RATE_LIMIT_COOLDOWN_SECONDS:-600}"
MAX_RATE_LIMIT_COOLDOWNS="${TALKVID_MAX_RATE_LIMIT_COOLDOWNS:-2}"
PYTHON_BIN="${TALKVID_PYTHON_BIN:-/opt/homebrew/Caskroom/miniforge/base/envs/musetalk/bin/python}"
FACECLIP_SIZE="${TALKVID_FACECLIP_SIZE:-256}"
FACECLIP_FPS="${TALKVID_FACECLIP_FPS:-25}"
FACECLIP_MAX_FRAMES="${TALKVID_FACECLIP_MAX_FRAMES:-750}"
FACECLIP_DETECT_EVERY="${TALKVID_FACECLIP_DETECT_EVERY:-10}"
FACECLIP_SMOOTH_WINDOW="${TALKVID_FACECLIP_SMOOTH_WINDOW:-9}"
FACECLIP_DETECTOR_BACKEND="${TALKVID_FACECLIP_DETECTOR_BACKEND:-opencv}"
FACECLIP_DETECTOR_DEVICE="${TALKVID_FACECLIP_DETECTOR_DEVICE:-auto}"
FACECLIP_DETECTOR_BATCH_SIZE="${TALKVID_FACECLIP_DETECTOR_BATCH_SIZE:-4}"
FACECLIP_RESIZE_DEVICE="${TALKVID_FACECLIP_RESIZE_DEVICE:-cpu}"
FACECLIP_FFMPEG_BIN="${TALKVID_FACECLIP_FFMPEG_BIN:-}"
FACECLIP_FFMPEG_THREADS="${TALKVID_FACECLIP_FFMPEG_THREADS:-1}"
FACECLIP_FFMPEG_TIMEOUT="${TALKVID_FACECLIP_FFMPEG_TIMEOUT:-180}"
FACECLIP_VIDEO_ENCODER="${TALKVID_FACECLIP_VIDEO_ENCODER:-auto}"
FACECLIP_VIDEO_BITRATE="${TALKVID_FACECLIP_VIDEO_BITRATE:-420k}"
FACECLIP_NORMALIZED_VIDEO_BITRATE="${TALKVID_FACECLIP_NORMALIZED_VIDEO_BITRATE:-15m}"
SEAL_OPEN_BATCHES_ON_START="${TALKVID_SEAL_OPEN_BATCHES_ON_START:-1}"
SKIP_PROCESSOR_LAUNCH="${TALKVID_SKIP_PROCESSOR_LAUNCH:-0}"
CURRENT_STAGE="init"
FETCH_PID=""
PROCESSOR_PID=""
FINAL_RC=0
DOWNLOAD_COOKIE_ARGS=()
CURRENT_BATCH_ROOT=""
CURRENT_BATCH_NAME=""

exec >>"$LOG_FILE" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log_exit() {
  local rc=$?
  echo "$(timestamp) [cycle] exit stage=$CURRENT_STAGE rc=$rc"
}
trap log_exit EXIT

log_cycle() {
  echo "$(timestamp) [cycle] $*"
}

run_allow_fail() {
  local stage="$1"
  shift
  CURRENT_STAGE="$stage"
  local rc=0
  "$@" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    log_cycle "stage=$stage rc=$rc; continuing"
  fi
  return 0
}

run_strict() {
  local stage="$1"
  shift
  CURRENT_STAGE="$stage"
  "$@"
}

fetch_alive() {
  [[ -n "$FETCH_PID" ]] && kill -0 "$FETCH_PID" 2>/dev/null
}

processor_alive() {
  if [[ "$SKIP_PROCESSOR_LAUNCH" == "1" ]]; then
    return 1
  fi
  [[ -n "$PROCESSOR_PID" ]] && kill -0 "$PROCESSOR_PID" 2>/dev/null
}

reap_processor_if_done() {
  if [[ "$SKIP_PROCESSOR_LAUNCH" == "1" ]]; then
    return 0
  fi
  if [[ -z "$PROCESSOR_PID" ]] || processor_alive; then
    return 0
  fi

  local rc=0
  wait "$PROCESSOR_PID" || rc=$?
  log_cycle "processor pid=$PROCESSOR_PID finished rc=$rc"
  if [[ "$FINAL_RC" -eq 0 && "$rc" -ne 0 ]]; then
    FINAL_RC="$rc"
  fi
  PROCESSOR_PID=""
}

next_batch_name() {
  local next_idx=0
  if [[ -f "$BATCH_COUNTER_FILE" ]]; then
    next_idx="$(<"$BATCH_COUNTER_FILE")"
  fi
  if [[ ! "$next_idx" == <-> ]]; then
    next_idx=0
  fi
  printf '%04d' "$next_idx"
  print $(( next_idx + 1 )) >"$BATCH_COUNTER_FILE"
}

find_resumable_batch_root() {
  local batch_root
  for batch_root in "$BATCH_RUNS_DIR"/batch_*(N); do
    [[ -d "$batch_root" ]] || continue
    [[ -f "$batch_root/fetch_complete.json" ]] && continue
    print "$batch_root"
    return 0
  done
  return 1
}

seal_existing_open_batches() {
  local batch_root
  local sealed=0
  for batch_root in "$BATCH_RUNS_DIR"/batch_*(N); do
    [[ -d "$batch_root" ]] || continue
    [[ -f "$batch_root/fetch_complete.json" ]] && continue
    local batch_basename="${batch_root:t}"
    local batch_name="${batch_basename#batch_}"
    cat >"$batch_root/fetch_complete.json" <<EOF
{"ts":"$(timestamp)","batch_name":"$batch_name","download_rc":0,"recovered":true}
EOF
    log_cycle "recovered_open_batch batch=$batch_name marker=$batch_root/fetch_complete.json"
    sealed=$(( sealed + 1 ))
  done
  log_cycle "recovered_open_batches=$sealed"
}

merge_batch_manifest() {
  local batch_root="$1"
  local batch_manifest="$batch_root/download_manifest.jsonl"
  if [[ ! -s "$batch_manifest" ]]; then
    log_cycle "batch_root=$batch_root has no download manifest to merge"
    return 0
  fi
  touch "$GLOBAL_MANIFEST"
  cat "$batch_manifest" >>"$GLOBAL_MANIFEST"
  log_cycle "merged manifest from $batch_root into $GLOBAL_MANIFEST"
}

write_fetch_complete_marker() {
  local batch_root="$1"
  local download_rc="$2"
  cat >"$batch_root/fetch_complete.json" <<EOF
{"ts":"$(timestamp)","batch_name":"$CURRENT_BATCH_NAME","download_rc":$download_rc}
EOF
  log_cycle "sealed batch=$CURRENT_BATCH_NAME marker=$batch_root/fetch_complete.json rc=$download_rc"
}

launch_processor() {
  if [[ "$SKIP_PROCESSOR_LAUNCH" == "1" ]]; then
    return 0
  fi
  if processor_alive; then
    return 0
  fi
  CURRENT_STAGE="process_live"
  "$PYTHON_BIN" scripts/process_local_talkvid_batches.py \
    --batches-dir "$BATCH_RUNS_DIR" \
    --data-root "$PROCESS_ROOT" \
    --manifest-path "$PROCESS_MANIFEST" \
    --dest-folder-id "$DEST_FOLDER_ID" \
    --remote "$REMOTE_NAME" \
    --python-bin "$PYTHON_BIN" \
    --follow \
    --poll-seconds 20 \
    --producer-done-flag "$PROCESS_DONE_FLAG" \
    --size "$FACECLIP_SIZE" \
    --fps "$FACECLIP_FPS" \
    --max-frames "$FACECLIP_MAX_FRAMES" \
    --detect-every "$FACECLIP_DETECT_EVERY" \
    --smooth-window "$FACECLIP_SMOOTH_WINDOW" \
    --detector-backend "$FACECLIP_DETECTOR_BACKEND" \
    --detector-device "$FACECLIP_DETECTOR_DEVICE" \
    --detector-batch-size "$FACECLIP_DETECTOR_BATCH_SIZE" \
    --resize-device "$FACECLIP_RESIZE_DEVICE" \
    --ffmpeg-bin "$FACECLIP_FFMPEG_BIN" \
    --ffmpeg-threads "$FACECLIP_FFMPEG_THREADS" \
    --ffmpeg-timeout "$FACECLIP_FFMPEG_TIMEOUT" \
    --video-encoder "$FACECLIP_VIDEO_ENCODER" \
    --normalized-video-bitrate "$FACECLIP_NORMALIZED_VIDEO_BITRATE" \
    --video-bitrate "$FACECLIP_VIDEO_BITRATE" &
  PROCESSOR_PID=$!
  log_cycle "processor pid=$PROCESSOR_PID"
}

launch_fetch() {
  local resumable_batch_root=""
  resumable_batch_root="$(find_resumable_batch_root || true)"
  if [[ -n "$resumable_batch_root" ]]; then
    local batch_basename=""
    CURRENT_BATCH_ROOT="$resumable_batch_root"
    batch_basename="${CURRENT_BATCH_ROOT:t}"
    CURRENT_BATCH_NAME="${batch_basename#batch_}"
    log_cycle "resuming batch=$CURRENT_BATCH_NAME batch_root=$CURRENT_BATCH_ROOT"
  else
    CURRENT_BATCH_NAME="$(next_batch_name)"
    CURRENT_BATCH_ROOT="$BATCH_RUNS_DIR/batch_$CURRENT_BATCH_NAME"
    mkdir -p "$CURRENT_BATCH_ROOT"
    log_cycle "starting new batch=$CURRENT_BATCH_NAME batch_root=$CURRENT_BATCH_ROOT"
  fi

  CURRENT_STAGE="download_live"
  python3 scripts/download_talkvid.py \
    --output "$CURRENT_BATCH_ROOT" \
    --variant with_captions \
    --max-height 720 \
    --min-duration 4.0 \
    --max-duration 6.5 \
    --min-width 720 \
    --min-height 720 \
    --min-dover 8.0 \
    --min-cotracker 0.90 \
    --target-additional-gb "$FETCH_TARGET_GB" \
    --min-free-gb 5.0 \
    --timeout 300 \
    --skip-manifest "$GLOBAL_MANIFEST" \
    --skip-manifest "$PROCESS_MANIFEST" \
    --rate-limit-cooldown-seconds "$RATE_LIMIT_COOLDOWN_SECONDS" \
    --max-rate-limit-cooldowns "$MAX_RATE_LIMIT_COOLDOWNS" \
    "${DOWNLOAD_COOKIE_ARGS[@]}" \
    --jobs "$DOWNLOAD_JOBS" &
  FETCH_PID=$!
  log_cycle "fetch pid=$FETCH_PID batch=$CURRENT_BATCH_NAME batch_root=$CURRENT_BATCH_ROOT"
}

log_cycle "start"
log_cycle "data_root=$DATA_ROOT"
log_cycle "batch_runs_dir=$BATCH_RUNS_DIR"
log_cycle "process_root=$PROCESS_ROOT"
log_cycle "dest_folder_id=$DEST_FOLDER_ID"
log_cycle "remote=$REMOTE_NAME batch_gb=$BATCH_GB fetch_target_gb=$FETCH_TARGET_GB jobs=$DOWNLOAD_JOBS rate_limit_cooldown_s=$RATE_LIMIT_COOLDOWN_SECONDS max_rate_limit_cooldowns=$MAX_RATE_LIMIT_COOLDOWNS"
log_cycle "processor_python=$PYTHON_BIN detector=$FACECLIP_DETECTOR_BACKEND/$FACECLIP_DETECTOR_DEVICE resize_device=$FACECLIP_RESIZE_DEVICE video_encoder=$FACECLIP_VIDEO_ENCODER normalized_video_bitrate=$FACECLIP_NORMALIZED_VIDEO_BITRATE video_bitrate=$FACECLIP_VIDEO_BITRATE"
log_cycle "seal_open_batches_on_start=$SEAL_OPEN_BATCHES_ON_START"
log_cycle "skip_processor_launch=$SKIP_PROCESSOR_LAUNCH"
if [[ -n "$COOKIES_FROM_BROWSER" ]]; then
  DOWNLOAD_COOKIE_ARGS=(--cookies-from-browser "$COOKIES_FROM_BROWSER")
  log_cycle "cookies_from_browser=$COOKIES_FROM_BROWSER"
else
  DOWNLOAD_COOKIE_ARGS=(--cookies-file "$COOKIES_FILE")
  log_cycle "cookies_file=$COOKIES_FILE"
fi

rm -f "$PROCESS_DONE_FLAG"
if [[ "$SEAL_OPEN_BATCHES_ON_START" == "1" ]]; then
  seal_existing_open_batches
fi
launch_processor

while true; do
  reap_processor_if_done
  launch_fetch

  CURRENT_STAGE="download_wait"
  DOWNLOAD_RC=0
  wait "$FETCH_PID" || DOWNLOAD_RC=$?
  FETCH_PID=""
  log_cycle "fetch batch=$CURRENT_BATCH_NAME rc=$DOWNLOAD_RC"

  merge_batch_manifest "$CURRENT_BATCH_ROOT"
  write_fetch_complete_marker "$CURRENT_BATCH_ROOT" "$DOWNLOAD_RC"
  launch_processor

  if [[ "$DOWNLOAD_RC" -eq 10 ]]; then
    log_cycle "batch=$CURRENT_BATCH_NAME sealed at target size; starting next fetch run"
    continue
  fi

  if [[ "$DOWNLOAD_RC" -ne 0 ]]; then
    FINAL_RC="$DOWNLOAD_RC"
    log_cycle "download stage exited rc=$DOWNLOAD_RC; stopping fetch loop after packaging current batch"
  fi
  break
done

touch "$PROCESS_DONE_FLAG"
reap_processor_if_done
if [[ "$SKIP_PROCESSOR_LAUNCH" != "1" ]] && processor_alive; then
  CURRENT_STAGE="process_wait"
  PROCESS_RC=0
  wait "$PROCESSOR_PID" || PROCESS_RC=$?
  log_cycle "processor pid=$PROCESSOR_PID finished rc=$PROCESS_RC"
  if [[ "$FINAL_RC" -eq 0 && "$PROCESS_RC" -ne 0 ]]; then
    FINAL_RC="$PROCESS_RC"
  fi
  PROCESSOR_PID=""
fi

CURRENT_STAGE="done"
log_cycle "done"
exit "$FINAL_RC"
