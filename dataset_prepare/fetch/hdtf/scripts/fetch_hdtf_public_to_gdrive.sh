#!/bin/bash
set -euo pipefail

# Compatibility wrapper for the YAML-driven HDTF raw fetch launcher.
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$STAGE_ROOT/scripts"
CONFIG_PATH="${1:-$STAGE_ROOT/configs/default.yaml}"
REPO_ROOT="$(cd "$STAGE_ROOT/../../.." && pwd)"
DEFAULT_FETCH_PYTHON="$REPO_ROOT/.venv-fetch/bin/python"
LAUNCHER_PYTHON="${PYTHON_BIN:-python3}"

if [[ -x "$DEFAULT_FETCH_PYTHON" ]]; then
  LAUNCHER_PYTHON="$DEFAULT_FETCH_PYTHON"
fi

exec "$LAUNCHER_PYTHON" "$SCRIPT_DIR/fetch_hdtf_public_to_gdrive.py" "$CONFIG_PATH"
