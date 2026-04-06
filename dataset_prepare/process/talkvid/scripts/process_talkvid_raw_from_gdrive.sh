#!/bin/bash
set -euo pipefail

# Compatibility wrapper for the YAML-driven TalkVid raw-archive processor.
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$STAGE_ROOT/scripts"
CONFIG_PATH="${1:-$STAGE_ROOT/configs/default_gdrive.yaml}"

exec "${PYTHON_BIN:-python3}" "$SCRIPT_DIR/process_talkvid_raw_from_gdrive.py" "$CONFIG_PATH"
