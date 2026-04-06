#!/bin/bash
set -euo pipefail

# Compatibility wrapper for the YAML-driven TalkVid raw fetch launcher.
STAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$STAGE_ROOT/scripts"
REPO_ROOT="$(cd "$STAGE_ROOT/../../.." && pwd)"
DEFAULT_FETCH_PYTHON="$REPO_ROOT/.venv-fetch/bin/python"
LAUNCHER_PYTHON="${PYTHON_BIN:-python3}"
CONFIG_PATH="$STAGE_ROOT/configs/default.yaml"

if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi

if [[ -x "$DEFAULT_FETCH_PYTHON" ]]; then
  LAUNCHER_PYTHON="$DEFAULT_FETCH_PYTHON"
fi

exec "$LAUNCHER_PYTHON" "$SCRIPT_DIR/fetch_talkvid_raw_to_gdrive.py" "$CONFIG_PATH" "$@"
