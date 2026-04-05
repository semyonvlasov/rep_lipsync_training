#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

exec python3 training/scripts/reap_idle_faceclip_remotes.py "$@"
