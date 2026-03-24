#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <output_dir> <wrapper_pid> <train_pid>" >&2
  exit 1
fi

OUT_DIR="$1"
WRAPPER_PID="$2"
TRAIN_PID="$3"

cd "$(dirname "$0")/.."

PIPELINE_LOG="$OUT_DIR/pipeline.log"
ROLLOVER_LOG="$OUT_DIR/rollover_epoch0.log"
EFFECTIVE_CONFIG="$OUT_DIR/effective_syncnet_config.yaml"
TRAIN_LIST="$OUT_DIR/syncnet_train_speakers.txt"
CKPT="$OUT_DIR/syncnet/syncnet_epoch000.pth"

log() {
  printf '[rollover] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" | tee -a "$ROLLOVER_LOG"
}

: > "$ROLLOVER_LOG"
log "watching checkpoint=$CKPT wrapper_pid=$WRAPPER_PID train_pid=$TRAIN_PID"

while [[ ! -f "$CKPT" ]]; do
  sleep 15
done

log "checkpoint detected"

if kill -0 "$TRAIN_PID" 2>/dev/null; then
  log "sending INT to train pid $TRAIN_PID"
  kill -INT "$TRAIN_PID" || true
fi
if kill -0 "$WRAPPER_PID" 2>/dev/null; then
  log "sending INT to wrapper pid $WRAPPER_PID"
  kill -INT "$WRAPPER_PID" || true
fi

for _ in $(seq 1 30); do
  train_alive=0
  wrapper_alive=0
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    train_alive=1
  fi
  if kill -0 "$WRAPPER_PID" 2>/dev/null; then
    wrapper_alive=1
  fi
  if [[ "$train_alive" -eq 0 && "$wrapper_alive" -eq 0 ]]; then
    break
  fi
  sleep 2
done

if kill -0 "$TRAIN_PID" 2>/dev/null; then
  log "train pid still alive, sending KILL"
  kill -9 "$TRAIN_PID" || true
fi
if kill -0 "$WRAPPER_PID" 2>/dev/null; then
  log "wrapper pid still alive, sending KILL"
  kill -9 "$WRAPPER_PID" || true
fi

log "starting resumed train"
python3 -u scripts/train_syncnet.py \
  --config "$EFFECTIVE_CONFIG" \
  --speaker-list "$TRAIN_LIST" \
  --resume "$CKPT" >> "$PIPELINE_LOG" 2>&1

log "resumed train finished"
