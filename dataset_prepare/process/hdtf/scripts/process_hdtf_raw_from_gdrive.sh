#!/bin/bash
set -euo pipefail

# Compatibility wrapper for the YAML-driven HDTF lazy-processing launcher.
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$STAGE_ROOT/scripts"
CONFIG_PATH="${1:-$STAGE_ROOT/configs/default.yaml}"

exec "${PYTHON_BIN:-python3}" "$SCRIPT_DIR/process_hdtf_raw_from_gdrive.py" "$CONFIG_PATH"
