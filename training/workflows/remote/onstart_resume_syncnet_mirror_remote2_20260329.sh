#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/lipsync_test/rep_lipsync_training}"
RUN_OUTPUT="$REPO_ROOT/training/output/training_cuda3090_syncnet_mirror_medium_20260329"
LOG_FILE="$RUN_OUTPUT/reboot_resume.log"
LAUNCH_SCRIPT="$REPO_ROOT/training/scripts/launch_syncnet_mirror_remote2_20260329.sh"

mkdir -p "$RUN_OUTPUT"
echo "[onstart] $(date '+%Y-%m-%d %H:%M:%S %Z') boot hook fired" >> "$LOG_FILE"
nohup bash "$LAUNCH_SCRIPT" >> "$LOG_FILE" 2>&1 &
