#!/bin/bash
set -euo pipefail

PORT="${PORT:-37576}"
REMOTE="${REMOTE:-root@ssh5.vast.ai}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/lipsync_test}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REMOTE_HOST="${REMOTE#*@}"
GDRIVE_FOLDER_URL="${GDRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1v06momk8fR-eqw79Z93zczBx_InWsCS9?usp=drive_link}"
LOCAL_IMPORT_DIR="${LOCAL_IMPORT_DIR:-}"
REMOTE_IMPORT_DIR="${REMOTE_IMPORT_DIR:-$REMOTE_ROOT/training/data/_imports/full_archives_20260323}"
DEFAULT_TALKVID_PROCESSED="${DEFAULT_TALKVID_PROCESSED:-talkvid_medium}"
TALKVID_HEAD_FILTER_MODE="${TALKVID_HEAD_FILTER_MODE:-medium}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-output/training_cuda3090_lipsync_hdtf_talkvid_latest}"
REMOTE_RUN_SCRIPT="${REMOTE_RUN_SCRIPT:-run_full_pipeline_hdtf_talkvid.sh}"
ARCHIVE_INCLUDE_GLOBS="${ARCHIVE_INCLUDE_GLOBS:-}"
ARCHIVE_EXCLUDE_GLOBS="${ARCHIVE_EXCLUDE_GLOBS:-}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PORT")
RSYNC_RSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$TRAINING_ROOT/output"
LOG_PATH="$LOG_DIR/vast_3090_sync_launch_hdtf_talkvid_20260323.log"
mkdir -p "$LOG_DIR"

exec >> "$LOG_PATH" 2>&1

echo "[watch] Started at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[watch] Remote: $REMOTE host $REMOTE_HOST port $PORT"
echo "[watch] GDrive folder: $GDRIVE_FOLDER_URL"
echo "[watch] Local import dir: ${LOCAL_IMPORT_DIR:-<none>}"
echo "[watch] Remote import dir: $REMOTE_IMPORT_DIR"
echo "[watch] Archive include globs: ${ARCHIVE_INCLUDE_GLOBS:-<none>}"
echo "[watch] Archive exclude globs: ${ARCHIVE_EXCLUDE_GLOBS:-<none>}"
echo "[watch] Default TalkVid processed target: $DEFAULT_TALKVID_PROCESSED"
echo "[watch] TalkVid head filter mode: $TALKVID_HEAD_FILTER_MODE"

until nc -z "$REMOTE_HOST" "$PORT"; do
  echo "[watch] $(date '+%H:%M:%S') port $PORT still closed, retry in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

echo "[watch] SSH port is reachable, starting no-data sync..."
"$SCRIPT_DIR/sync_remote_code.sh"

if [[ -n "$LOCAL_IMPORT_DIR" ]]; then
  echo "[watch] Syncing staged archives from $LOCAL_IMPORT_DIR to $REMOTE:$REMOTE_IMPORT_DIR ..."
  ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_IMPORT_DIR'"

  rsync_cmd=(rsync -a --whole-file --partial --progress -e "$RSYNC_RSH")
  if [[ -n "$ARCHIVE_INCLUDE_GLOBS" ]]; then
    rsync_cmd+=(--include '*/')
    IFS=',' read -r -a include_patterns <<< "$ARCHIVE_INCLUDE_GLOBS"
    for pattern in "${include_patterns[@]}"; do
      if [[ -n "$pattern" ]]; then
        rsync_cmd+=(--include "$pattern")
      fi
    done
    rsync_cmd+=(--exclude '*')
  fi
  if [[ -n "$ARCHIVE_EXCLUDE_GLOBS" ]]; then
    IFS=',' read -r -a exclude_patterns <<< "$ARCHIVE_EXCLUDE_GLOBS"
    for pattern in "${exclude_patterns[@]}"; do
      if [[ -n "$pattern" ]]; then
        rsync_cmd+=(--exclude "$pattern")
      fi
    done
  fi
  rsync_cmd+=("$LOCAL_IMPORT_DIR/" "$REMOTE:$REMOTE_IMPORT_DIR/")
  "${rsync_cmd[@]}"
fi

echo "[watch] Launching remote mixed training..."
ssh "${SSH_OPTS[@]}" "$REMOTE" "
  set -euo pipefail
  cd '$REMOTE_ROOT/training'
  mkdir -p '$REMOTE_OUTPUT_DIR'
  nohup env \
    GDRIVE_FOLDER_URL='$GDRIVE_FOLDER_URL' \
    IMPORT_SOURCE_DIR='${LOCAL_IMPORT_DIR:+$REMOTE_IMPORT_DIR}' \
    DEFAULT_TALKVID_PROCESSED='$DEFAULT_TALKVID_PROCESSED' \
    TALKVID_HEAD_FILTER_MODE='$TALKVID_HEAD_FILTER_MODE' \
    ARCHIVE_INCLUDE_GLOBS='$ARCHIVE_INCLUDE_GLOBS' \
    ARCHIVE_EXCLUDE_GLOBS='$ARCHIVE_EXCLUDE_GLOBS' \
    OUTPUT_DIR='$REMOTE_OUTPUT_DIR' \
    bash './workflows/train/$REMOTE_RUN_SCRIPT' \
    > '$REMOTE_OUTPUT_DIR/launcher.log' 2>&1 &
  echo \$! > '$REMOTE_OUTPUT_DIR/remote_pid.txt'
  echo remote_pid=\$!
"

echo "[watch] Remote launch command sent successfully"
echo "[watch] Remote pipeline log: $REMOTE_ROOT/training/$REMOTE_OUTPUT_DIR/pipeline.log"
