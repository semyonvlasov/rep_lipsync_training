#!/bin/zsh
set -euo pipefail
unsetopt BG_NICE

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LEGACY_WORK_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${TALKVID_DATA_ROOT:-$LEGACY_WORK_ROOT/training/data/talkvid_local}"
LEGACY_RAW_DIR="$DATA_ROOT/raw"
BATCH_RUNS_DIR="$DATA_ROOT/batches"
ARCHIVES_DIR="$DATA_ROOT/archives"
OUT_DIR="${TALKVID_OUT_DIR:-$LEGACY_WORK_ROOT/training/output/talkvid_local_cycle}"
GLOBAL_MANIFEST="$DATA_ROOT/download_manifest.jsonl"
BATCH_COUNTER_FILE="$DATA_ROOT/next_fetch_batch_index.txt"
mkdir -p "$LEGACY_RAW_DIR" "$BATCH_RUNS_DIR" "$ARCHIVES_DIR" "$OUT_DIR"

LOG_FILE="$OUT_DIR/cycle.log"
FOLDER_ID="${GDRIVE_FOLDER_ID:-1v06momk8fR-eqw79Z93zczBx_InWsCS9}"
REMOTE_NAME="${GDRIVE_REMOTE:-gdrive:}"
COOKIES_FILE="${YTDLP_COOKIES_FILE:-/tmp/talkvid_local_cookies.txt}"
COOKIES_FROM_BROWSER="${YTDLP_COOKIES_FROM_BROWSER:-}"
BATCH_GB="${TALKVID_BATCH_GB:-1.0}"
FETCH_TARGET_GB="${TALKVID_TARGET_ADDITIONAL_GB:-$BATCH_GB}"
DOWNLOAD_JOBS="${TALKVID_DOWNLOAD_JOBS:-4}"
REQUEST_MIN_INTERVAL_SECONDS="${TALKVID_REQUEST_MIN_INTERVAL_SECONDS:-1.0}"
RATE_LIMIT_COOLDOWN_SECONDS="${TALKVID_RATE_LIMIT_COOLDOWN_SECONDS:-600}"
MAX_RATE_LIMIT_COOLDOWNS="${TALKVID_MAX_RATE_LIMIT_COOLDOWNS:-2}"
CURRENT_STAGE="init"
FETCH_PID=""
UPLOAD_PID=""
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

uploader_alive() {
  [[ -n "$UPLOAD_PID" ]] && kill -0 "$UPLOAD_PID" 2>/dev/null
}

reap_uploader_if_done() {
  if [[ -z "$UPLOAD_PID" ]] || uploader_alive; then
    return 0
  fi

  local rc=0
  wait "$UPLOAD_PID" || rc=$?
  log_cycle "uploader pid=$UPLOAD_PID finished rc=$rc"
  if [[ "$FINAL_RC" -eq 0 && "$rc" -ne 0 ]]; then
    FINAL_RC="$rc"
  fi
  UPLOAD_PID=""
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

package_batch_root() {
  local batch_root="$1"
  local batch_raw_dir="$batch_root/raw"
  local mp4_files=("$batch_raw_dir"/*.mp4(N))
  if (( ${#mp4_files[@]} == 0 )); then
    log_cycle "batch_root=$batch_root has no ready mp4 files to package"
    return 0
  fi

  run_strict package_batch \
    python3 training/scripts/package_raw_batches.py \
    --raw-dir "$batch_raw_dir" \
    --archives-dir "$ARCHIVES_DIR" \
    --prefix talkvid_raw \
    --batch-root "$batch_root" \
    --max-clips 0 \
    --max-gb 0 \
    --max-batches 1
}

launch_uploader() {
  if uploader_alive; then
    return 0
  fi
  CURRENT_STAGE="upload_live"
  python3 training/scripts/upload_batches_and_cleanup.py \
    --raw-dir "$LEGACY_RAW_DIR" \
    --archives-dir "$ARCHIVES_DIR" \
    --remote "$REMOTE_NAME" \
    --drive-root-folder-id "$FOLDER_ID" \
    --max-batches 0 &
  UPLOAD_PID=$!
  log_cycle "uploader pid=$UPLOAD_PID"
}

launch_fetch() {
  CURRENT_BATCH_NAME="$(next_batch_name)"
  CURRENT_BATCH_ROOT="$BATCH_RUNS_DIR/batch_$CURRENT_BATCH_NAME"
  mkdir -p "$CURRENT_BATCH_ROOT"

  CURRENT_STAGE="download_live"
  python3 training/scripts/download_talkvid.py \
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
    --skip-manifest "$ARCHIVES_DIR/batches_manifest.jsonl" \
    --skip-manifest "$ARCHIVES_DIR/uploaded_manifest.jsonl" \
    --request-min-interval-seconds "$REQUEST_MIN_INTERVAL_SECONDS" \
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
log_cycle "folder_id=$FOLDER_ID"
log_cycle "remote=$REMOTE_NAME batch_gb=$BATCH_GB fetch_target_gb=$FETCH_TARGET_GB jobs=$DOWNLOAD_JOBS request_min_interval_s=$REQUEST_MIN_INTERVAL_SECONDS rate_limit_cooldown_s=$RATE_LIMIT_COOLDOWN_SECONDS max_rate_limit_cooldowns=$MAX_RATE_LIMIT_COOLDOWNS"
if [[ -n "$COOKIES_FROM_BROWSER" ]]; then
  DOWNLOAD_COOKIE_ARGS=(--cookies-from-browser "$COOKIES_FROM_BROWSER")
  log_cycle "cookies_from_browser=$COOKIES_FROM_BROWSER"
else
  DOWNLOAD_COOKIE_ARGS=(--cookies-file "$COOKIES_FILE")
  log_cycle "cookies_file=$COOKIES_FILE"
fi

launch_uploader

while true; do
  reap_uploader_if_done
  launch_fetch

  CURRENT_STAGE="download_wait"
  DOWNLOAD_RC=0
  wait "$FETCH_PID" || DOWNLOAD_RC=$?
  FETCH_PID=""
  log_cycle "fetch batch=$CURRENT_BATCH_NAME rc=$DOWNLOAD_RC"

  merge_batch_manifest "$CURRENT_BATCH_ROOT"
  package_batch_root "$CURRENT_BATCH_ROOT"
  launch_uploader

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

reap_uploader_if_done
if uploader_alive; then
  CURRENT_STAGE="upload_wait"
  UPLOAD_RC=0
  wait "$UPLOAD_PID" || UPLOAD_RC=$?
  log_cycle "uploader pid=$UPLOAD_PID finished rc=$UPLOAD_RC"
  if [[ "$FINAL_RC" -eq 0 && "$UPLOAD_RC" -ne 0 ]]; then
    FINAL_RC="$UPLOAD_RC"
  fi
  UPLOAD_PID=""
fi

run_allow_fail upload_tail \
  python3 training/scripts/upload_batches_and_cleanup.py \
  --raw-dir "$LEGACY_RAW_DIR" \
  --archives-dir "$ARCHIVES_DIR" \
  --remote "$REMOTE_NAME" \
  --drive-root-folder-id "$FOLDER_ID" \
  --max-batches 0

CURRENT_STAGE="done"
log_cycle "done"
exit "$FINAL_RC"
