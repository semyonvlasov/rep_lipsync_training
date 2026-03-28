#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/lipsync_test/rep_lipsync_training}"
RUN_OUTPUT="$REPO_ROOT/training/output/generator_medium_x96_b32_official_syncnet_continue_from_ablation_winner_20260328"
LOG_FILE="$RUN_OUTPUT/reboot_resume.log"
LAUNCH_SCRIPT="$REPO_ROOT/training/scripts/launch_generator_official_syncnet_continue_from_ablation_winner_20260328.sh"

mkdir -p "$RUN_OUTPUT"
echo "[onstart] $(date '+%Y-%m-%d %H:%M:%S %Z') boot hook fired" >> "$LOG_FILE"
nohup bash "$LAUNCH_SCRIPT" >> "$LOG_FILE" 2>&1 &
