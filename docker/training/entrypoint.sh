#!/usr/bin/env bash
set -euo pipefail

export LIPSYNC_REPO_ROOT="${LIPSYNC_REPO_ROOT:-/opt/lipsync}"
export PYTHONPATH="${LIPSYNC_REPO_ROOT}:${LIPSYNC_REPO_ROOT}/training:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -f /run/secrets/rclone.conf && -z "${RCLONE_CONFIG:-}" ]]; then
  export RCLONE_CONFIG=/run/secrets/rclone.conf
fi

mkdir -p /workspace/data /workspace/output /workspace/cache

link_runtime_dir() {
  local target="$1"
  local link_path="$2"
  mkdir -p "$target" "$(dirname "$link_path")"
  if [[ -L "$link_path" ]]; then
    ln -sfn "$target" "$link_path"
  elif [[ ! -e "$link_path" ]]; then
    ln -s "$target" "$link_path"
  fi
}

link_runtime_dir /workspace/output "${LIPSYNC_REPO_ROOT}/training/output"
link_runtime_dir /workspace/data/_imports "${LIPSYNC_REPO_ROOT}/training/data/_imports"
link_runtime_dir /workspace/data/hdtf "${LIPSYNC_REPO_ROOT}/training/data/hdtf"
link_runtime_dir /workspace/data/talkvid "${LIPSYNC_REPO_ROOT}/training/data/talkvid"
link_runtime_dir /workspace/data/talkvid_local "${LIPSYNC_REPO_ROOT}/training/data/talkvid_local"
link_runtime_dir /workspace/data/talkvid_remote "${LIPSYNC_REPO_ROOT}/training/data/talkvid_remote"
link_runtime_dir /workspace/data/faceclip_local "${LIPSYNC_REPO_ROOT}/training/data/faceclip_local"

case "${1:-}" in
  prepare|doctor|list-artifacts|merge-dataset|prewarm-cache|sync-align|train-syncnet|train-generator|train-generator-gan|benchmark)
    exec lipsyncctl "$@"
    ;;
  "")
    exec lipsyncctl doctor
    ;;
  *)
    exec "$@"
    ;;
esac
