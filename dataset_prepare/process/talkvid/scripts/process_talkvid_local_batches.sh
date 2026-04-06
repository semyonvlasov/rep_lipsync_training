#!/bin/bash
set -euo pipefail

# Compatibility wrapper for the YAML-driven TalkVid local-batch processor.
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$STAGE_ROOT/scripts"
CONFIG_PATH="${1:-$STAGE_ROOT/configs/default_local_batches.yaml}"

exec "${PYTHON_BIN:-python3}" "$SCRIPT_DIR/process_talkvid_local_batches.py" "$CONFIG_PATH"
