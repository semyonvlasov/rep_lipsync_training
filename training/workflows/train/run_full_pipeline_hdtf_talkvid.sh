#!/bin/bash
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$TRAINING_ROOT"

PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd .. && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-output/training_cuda3090_lipsync_hdtf_talkvid_latest}"
LOG_PATH="$OUTPUT_DIR/pipeline.log"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/lipsync_cuda3090_hdtf_talkvid.yaml}"
OFFICIAL_SYNCNET_PATH="${OFFICIAL_SYNCNET_PATH:-${SYNCNET_PATH:-../models/official_syncnet/checkpoints/lipsync_expert.pth}}"
GDRIVE_FOLDER_URL="${GDRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1v06momk8fR-eqw79Z93zczBx_InWsCS9?usp=drive_link}"
IMPORT_SOURCE_DIR="${IMPORT_SOURCE_DIR:-}"
DEFAULT_TALKVID_PROCESSED="${DEFAULT_TALKVID_PROCESSED:-talkvid}"
TALKVID_HEAD_FILTER_MODE="${TALKVID_HEAD_FILTER_MODE:-off}"
TALKVID_MIN_QUALITY="${TALKVID_MIN_QUALITY:-medium}"
TALKVID_QUALITY_OUTPUT_DIR="${TALKVID_QUALITY_OUTPUT_DIR:-$OUTPUT_DIR/talkvid_quality}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-data/_imports/gdrive_downloads}"
EXTRACT_DIR="${EXTRACT_DIR:-data/_imports/gdrive_extracted}"
KEEP_DOWNLOADS="${KEEP_DOWNLOADS:-0}"
KEEP_EXTRACTED="${KEEP_EXTRACTED:-0}"
SKIP_DATA_DOWNLOAD="${SKIP_DATA_DOWNLOAD:-0}"
ARCHIVE_INCLUDE_GLOBS="${ARCHIVE_INCLUDE_GLOBS:-}"
ARCHIVE_EXCLUDE_GLOBS="${ARCHIVE_EXCLUDE_GLOBS:-}"
PREPROCESS_OVERWRITE="${PREPROCESS_OVERWRITE:-0}"
TRAIN_GENERATOR_EPOCHS="${TRAIN_GENERATOR_EPOCHS:-}"
TRAIN_SYNCNET_EPOCHS="${TRAIN_SYNCNET_EPOCHS:-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-}"
DETECTOR_BACKEND="${DETECTOR_BACKEND:-sfd}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-cuda}"
DETECTOR_BATCH_SIZE="${DETECTOR_BATCH_SIZE:-4}"
PROCESS_NO_PREVIEW="${PROCESS_NO_PREVIEW:-1}"
PROCESS_FFMPEG_THREADS="${PROCESS_FFMPEG_THREADS:-1}"
PROCESS_VIDEO_ENCODER="${PROCESS_VIDEO_ENCODER:-auto}"
PROCESS_RESIZE_DEVICE="${PROCESS_RESIZE_DEVICE:-cuda}"
HDTF_PROCESS_WORKERS="${HDTF_PROCESS_WORKERS:-1}"
TALKVID_PROCESS_WORKERS="${TALKVID_PROCESS_WORKERS:-4}"
WATCHDOG_SAMPLES="${WATCHDOG_SAMPLES:-8}"
WATCHDOG_PATIENCE="${WATCHDOG_PATIENCE:-2}"
WATCHDOG_DEGRADE_THRESHOLD="${WATCHDOG_DEGRADE_THRESHOLD:-0.03}"
WATCHDOG_MIN_EPOCH="${WATCHDOG_MIN_EPOCH:-2}"
SYNCNET_HOLDOUT_COUNT="${SYNCNET_HOLDOUT_COUNT:-10}"
SYNCNET_COMPARE_LAST_N="${SYNCNET_COMPARE_LAST_N:-3}"
SYNCNET_COMPARE_SAMPLES="${SYNCNET_COMPARE_SAMPLES:-20}"
SYNCNET_COMPARE_SEED="${SYNCNET_COMPARE_SEED:-123}"
SYNCNET_COMPARE_DEVICE="${SYNCNET_COMPARE_DEVICE:-cuda}"
SYNCNET_SELECTED_JSON="${SYNCNET_SELECTED_JSON:-$OUTPUT_DIR/syncnet_selected_teacher.json}"

case "$DEFAULT_TALKVID_PROCESSED" in
  talkvid)
    TALKVID_PROCESSED_ROOT="data/talkvid/processed"
    ;;
  talkvid_soft)
    TALKVID_PROCESSED_ROOT="data/talkvid/processed_soft"
    ;;
  talkvid_medium)
    echo "[pipeline] WARNING: DEFAULT_TALKVID_PROCESSED=talkvid_medium is deprecated; using data/talkvid/processed" >&2
    TALKVID_PROCESSED_ROOT="data/talkvid/processed"
    ;;
  talkvid_strict)
    TALKVID_PROCESSED_ROOT="data/talkvid/processed_strict"
    ;;
  *)
    echo "Unsupported DEFAULT_TALKVID_PROCESSED=$DEFAULT_TALKVID_PROCESSED" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"
export KMP_DUPLICATE_LIB_OK=TRUE
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/lipsync_numba_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$NUMBA_CACHE_DIR"

: > "$LOG_PATH"
exec >> "$LOG_PATH" 2>&1

has_find_match() {
  find "$@" -print -quit | grep -q .
}

echo "[pipeline] CUDA 3090 HDTF+TalkVid run started at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[pipeline] Log: $LOG_PATH"
echo "[pipeline] Python: $PYTHON"
echo "[pipeline] Config: $TRAIN_CONFIG"
echo "[pipeline] Official SyncNet checkpoint: $OFFICIAL_SYNCNET_PATH"
echo "[pipeline] GDrive folder: $GDRIVE_FOLDER_URL"
echo "[pipeline] Import source dir: ${IMPORT_SOURCE_DIR:-<none>}"
echo "[pipeline] Default TalkVid processed target: $DEFAULT_TALKVID_PROCESSED"
echo "[pipeline] TalkVid processed root: $TALKVID_PROCESSED_ROOT"
echo "[pipeline] TalkVid head filter mode: $TALKVID_HEAD_FILTER_MODE"
echo "[pipeline] TalkVid min quality: $TALKVID_MIN_QUALITY"
echo "[pipeline] TalkVid quality output dir: $TALKVID_QUALITY_OUTPUT_DIR"
echo "[pipeline] Archive include globs: ${ARCHIVE_INCLUDE_GLOBS:-<none>}"
echo "[pipeline] Archive exclude globs: ${ARCHIVE_EXCLUDE_GLOBS:-<none>}"
echo "[pipeline] Preprocess overwrite: $PREPROCESS_OVERWRITE"
echo "[pipeline] Train generator epochs override: ${TRAIN_GENERATOR_EPOCHS:-<config>}"
echo "[pipeline] Train syncnet epochs override: ${TRAIN_SYNCNET_EPOCHS:-<config>}"
echo "[pipeline] Train batch size override: ${TRAIN_BATCH_SIZE:-<config>}"
echo "[pipeline] Data num_workers override: ${DATA_NUM_WORKERS:-<config>}"
echo "[pipeline] Detector backend: $DETECTOR_BACKEND"
echo "[pipeline] Detector device: $DETECTOR_DEVICE"
echo "[pipeline] Detector batch size: $DETECTOR_BATCH_SIZE"
echo "[pipeline] Process no preview: $PROCESS_NO_PREVIEW"
echo "[pipeline] Process ffmpeg threads: $PROCESS_FFMPEG_THREADS"
echo "[pipeline] Process video encoder: $PROCESS_VIDEO_ENCODER"
echo "[pipeline] Process resize device: $PROCESS_RESIZE_DEVICE"
echo "[pipeline] HDTF process workers: $HDTF_PROCESS_WORKERS"
echo "[pipeline] TalkVid process workers: $TALKVID_PROCESS_WORKERS"
echo "[pipeline] Watchdog samples: $WATCHDOG_SAMPLES"
echo "[pipeline] Watchdog patience: $WATCHDOG_PATIENCE"
echo "[pipeline] Watchdog degrade threshold: $WATCHDOG_DEGRADE_THRESHOLD"
echo "[pipeline] Watchdog min epoch: $WATCHDOG_MIN_EPOCH"
echo "[pipeline] SyncNet holdout count: $SYNCNET_HOLDOUT_COUNT"
echo "[pipeline] SyncNet compare last N checkpoints: $SYNCNET_COMPARE_LAST_N"
echo "[pipeline] SyncNet compare samples: $SYNCNET_COMPARE_SAMPLES"
echo "[pipeline] SyncNet compare seed: $SYNCNET_COMPARE_SEED"
echo "[pipeline] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[pipeline] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "[pipeline] PYTHONPATH=$PYTHONPATH"
nvidia-smi || true

echo "[pipeline] Bootstrapping system deps at $(date '+%Y-%m-%d %H:%M:%S %Z')"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ffmpeg libsndfile1 rsync

echo "[pipeline] Bootstrapping Python deps at $(date '+%Y-%m-%d %H:%M:%S %Z')"
$PYTHON -m pip install --no-cache-dir --upgrade pip
$PYTHON -m pip install --no-cache-dir \
  gdown \
  opencv-python-headless \
  librosa \
  soundfile \
  scipy \
  PyYAML \
  tqdm

echo "[pipeline] Python/Torch probe at $(date '+%Y-%m-%d %H:%M:%S %Z')"
$PYTHON - <<'PY'
import sys
print(f"[pipeline] python_version={sys.version.split()[0]}", flush=True)
try:
    import torch
    print(f"[pipeline] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
except Exception as exc:
    print(f"[pipeline] torch probe failed: {exc}", flush=True)
    raise
PY

if [[ "$SKIP_DATA_DOWNLOAD" != "1" ]]; then
  echo "[pipeline] Downloading/staging training inputs at $(date '+%Y-%m-%d %H:%M:%S %Z')"
  download_cmd=(
    "$PYTHON" -u scripts/download_training_inputs_public_gdrive.py
    --python "$PYTHON"
    --download-dir "$DOWNLOAD_DIR"
    --extract-dir "$EXTRACT_DIR"
    --default-talkvid-processed "$DEFAULT_TALKVID_PROCESSED"
  )
  if [[ -n "$IMPORT_SOURCE_DIR" ]]; then
    download_cmd+=(--source-dir "$IMPORT_SOURCE_DIR")
  else
    download_cmd+=(--folder-url "$GDRIVE_FOLDER_URL")
  fi
  if [[ "$KEEP_DOWNLOADS" == "1" ]]; then
    download_cmd+=(--keep-downloads)
  fi
  if [[ "$KEEP_EXTRACTED" == "1" ]]; then
    download_cmd+=(--keep-extracted)
  fi
  if [[ -n "$ARCHIVE_INCLUDE_GLOBS" ]]; then
    IFS=',' read -r -a include_patterns <<< "$ARCHIVE_INCLUDE_GLOBS"
    for pattern in "${include_patterns[@]}"; do
      if [[ -n "$pattern" ]]; then
        download_cmd+=(--include-archive-glob "$pattern")
      fi
    done
  fi
  if [[ -n "$ARCHIVE_EXCLUDE_GLOBS" ]]; then
    IFS=',' read -r -a exclude_patterns <<< "$ARCHIVE_EXCLUDE_GLOBS"
    for pattern in "${exclude_patterns[@]}"; do
      if [[ -n "$pattern" ]]; then
        download_cmd+=(--exclude-archive-glob "$pattern")
      fi
    done
  fi
  "${download_cmd[@]}"
else
  echo "[pipeline] SKIP_DATA_DOWNLOAD=1, using existing staged roots"
fi

echo "[pipeline] Staged input inventory at $(date '+%Y-%m-%d %H:%M:%S %Z')"
$PYTHON - <<'PY'
from pathlib import Path

roots = {
    "hdtf_raw": Path("data/hdtf/raw"),
    "hdtf_clips": Path("data/hdtf/clips"),
    "hdtf": Path("data/hdtf/processed"),
    "talkvid_raw": Path("data/talkvid_remote/raw"),
    "talkvid": Path("data/talkvid/processed"),
    "talkvid_soft": Path("data/talkvid/processed_soft"),
    "talkvid_strict": Path("data/talkvid/processed_strict"),
}

for name, root in roots.items():
    if not root.is_dir():
        continue
    if name.endswith("_raw") or name.endswith("_clips"):
        count = sum(1 for _ in root.glob("*.mp4"))
    else:
        count = 0
        for child in root.iterdir():
            if child.is_dir() and (child / "frames.npy").is_file() and (child / "mel.npy").is_file():
                count += 1
    print(f"[pipeline] dataset={name} speakers={count} root={root}", flush=True)
PY
du -sh data/hdtf/raw data/hdtf/clips data/hdtf/processed data/talkvid_remote/raw data/talkvid/processed data/talkvid/processed_soft data/talkvid/processed_strict 2>/dev/null || true

echo "[pipeline] Preparing HDTF at $(date '+%Y-%m-%d %H:%M:%S %Z')"
if [[ -d data/hdtf/clips ]] && has_find_match data/hdtf/clips -maxdepth 1 -name '*.mp4'; then
  echo "[pipeline] Building HDTF processed set from staged clips"
  preprocess_cmd=(
    "$PYTHON" -u scripts/preprocess_dataset.py
    --input data/hdtf/clips
    --output data/hdtf/processed
    --size 256
    --ffmpeg-threads "$PROCESS_FFMPEG_THREADS"
    --resize-device "$PROCESS_RESIZE_DEVICE"
    --detector-backend "$DETECTOR_BACKEND"
    --detector-device "$DETECTOR_DEVICE"
    --detector-batch-size "$DETECTOR_BATCH_SIZE"
  )
  if [[ "$PROCESS_NO_PREVIEW" == "1" ]]; then
    preprocess_cmd+=(--no-preview)
  fi
  if [[ "$PREPROCESS_OVERWRITE" == "1" ]]; then
    preprocess_cmd+=(--overwrite)
  fi
  "${preprocess_cmd[@]}"
elif [[ -d data/hdtf/raw ]] && has_find_match data/hdtf/raw -maxdepth 1 -name '*.mp4'; then
  echo "[pipeline] Building HDTF processed set from staged raw videos"
  "$PYTHON" -u scripts/process_hdtf_incremental.py \
    --raw-dir data/hdtf/raw \
    --clips-dir data/hdtf/clips \
    --processed-dir data/hdtf/processed \
    --ffmpeg-threads "$PROCESS_FFMPEG_THREADS" \
    --resize-device "$PROCESS_RESIZE_DEVICE" \
    --detector-backend "$DETECTOR_BACKEND" \
    --detector-device "$DETECTOR_DEVICE" \
    --detector-batch-size "$DETECTOR_BATCH_SIZE" \
    --workers "$HDTF_PROCESS_WORKERS" \
    --video-encoder "$PROCESS_VIDEO_ENCODER" \
    $([[ "$PROCESS_NO_PREVIEW" == "1" ]] && printf '%s' --no-preview)
elif [[ -d data/hdtf/processed ]] && has_find_match data/hdtf/processed -mindepth 2 -maxdepth 2 -name frames.npy; then
  echo "[pipeline] HDTF processed already present, no new staged clips/raw found"
else
  echo "[pipeline] No staged HDTF data found"
fi

echo "[pipeline] Preparing TalkVid at $(date '+%Y-%m-%d %H:%M:%S %Z')"
talkvid_raw_present=0
if [[ -d data/talkvid_remote/raw ]] && has_find_match data/talkvid_remote/raw -maxdepth 1 -name '*.mp4'; then
  talkvid_raw_present=1
  echo "[pipeline] Building TalkVid processed set from staged raw clips"
  "$PYTHON" -u scripts/process_talkvid_incremental.py \
    --raw-dir data/talkvid_remote/raw \
    --clips-dir data/talkvid_remote/clips_25fps \
    --processed-dir "$TALKVID_PROCESSED_ROOT" \
    --head-filter-mode "$TALKVID_HEAD_FILTER_MODE" \
    --ffmpeg-threads "$PROCESS_FFMPEG_THREADS" \
    --resize-device "$PROCESS_RESIZE_DEVICE" \
    --detector-backend "$DETECTOR_BACKEND" \
    --detector-device "$DETECTOR_DEVICE" \
    --detector-batch-size "$DETECTOR_BATCH_SIZE" \
    --workers "$TALKVID_PROCESS_WORKERS" \
    --video-encoder "$PROCESS_VIDEO_ENCODER" \
    $([[ "$PROCESS_NO_PREVIEW" == "1" ]] && printf '%s' --no-preview)
elif [[ -d "$TALKVID_PROCESSED_ROOT" ]] && has_find_match "$TALKVID_PROCESSED_ROOT" -mindepth 2 -maxdepth 2 -name frames.npy; then
  echo "[pipeline] TalkVid processed set already present at $TALKVID_PROCESSED_ROOT, no new staged raw found"
else
  echo "[pipeline] No staged TalkVid data found"
fi

if [[ "$talkvid_raw_present" == "1" ]] && ! has_find_match "$TALKVID_PROCESSED_ROOT" -mindepth 2 -maxdepth 2 -name frames.npy; then
  echo "[pipeline] ERROR: TalkVid raw clips were present but no processed TalkVid samples were generated"
  exit 1
fi

TALKVID_SELECTED_LIST=""
if [[ -d "$TALKVID_PROCESSED_ROOT" ]] && has_find_match "$TALKVID_PROCESSED_ROOT" -mindepth 2 -maxdepth 2 -name frames.npy; then
  echo "[pipeline] Sorting TalkVid processed quality tiers at $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "$PYTHON" -u scripts/sort_talkvid_processed_by_quality.py \
    --processed-dir "$TALKVID_PROCESSED_ROOT" \
    --raw-dir data/talkvid_remote/raw \
    --output-dir "$TALKVID_QUALITY_OUTPUT_DIR"
  TALKVID_SELECTED_LIST="$TALKVID_QUALITY_OUTPUT_DIR/min_${TALKVID_MIN_QUALITY}.txt"
  if [[ ! -s "$TALKVID_SELECTED_LIST" ]]; then
    echo "[pipeline] ERROR: TalkVid selected speaker list is empty or missing: $TALKVID_SELECTED_LIST"
    exit 1
  fi
  echo "[pipeline] TalkVid selected speaker list: $TALKVID_SELECTED_LIST"
fi

HDTF_ALL_SPEAKERS="$OUTPUT_DIR/hdtf_all_speakers.txt"
COMBINED_SPEAKER_LIST="$OUTPUT_DIR/combined_speakers_${TALKVID_MIN_QUALITY}.txt"
"$PYTHON" - <<PY
import os

def collect_names(root):
    names = []
    if root and os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            speaker_dir = os.path.join(root, name)
            if not os.path.isdir(speaker_dir):
                continue
            if os.path.exists(os.path.join(speaker_dir, "frames.npy")) and os.path.exists(os.path.join(speaker_dir, "mel.npy")):
                names.append(name)
    return names

hdtf_names = collect_names("data/hdtf/processed")
talkvid_names = []
talkvid_list = "$TALKVID_SELECTED_LIST"
if talkvid_list and os.path.exists(talkvid_list):
    with open(talkvid_list) as f:
        talkvid_names = [line.strip() for line in f if line.strip()]

for path, names in [("$HDTF_ALL_SPEAKERS", hdtf_names), ("$COMBINED_SPEAKER_LIST", sorted(set(hdtf_names + talkvid_names)))]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for name in names:
            f.write(name + "\\n")

print(f"[pipeline] hdtf_all_speakers={len(hdtf_names)} path=$HDTF_ALL_SPEAKERS", flush=True)
print(f"[pipeline] combined_speakers={len(set(hdtf_names + talkvid_names))} path=$COMBINED_SPEAKER_LIST", flush=True)
PY

EFFECTIVE_CONFIG="$OUTPUT_DIR/effective_train_config.yaml"
echo "[pipeline] Writing effective training config to $EFFECTIVE_CONFIG"
"$PYTHON" - <<PY
import yaml

with open("$TRAIN_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("data", {})
cfg["data"]["talkvid_root"] = "$TALKVID_PROCESSED_ROOT"
cfg.setdefault("training", {})
cfg["training"]["output_dir"] = "$OUTPUT_DIR"

if "$TRAIN_GENERATOR_EPOCHS":
    cfg.setdefault("generator", {})
    cfg["generator"]["epochs"] = int("$TRAIN_GENERATOR_EPOCHS")
if "$TRAIN_SYNCNET_EPOCHS":
    cfg.setdefault("syncnet", {})
    cfg["syncnet"]["epochs"] = int("$TRAIN_SYNCNET_EPOCHS")
if "$TRAIN_BATCH_SIZE":
    cfg.setdefault("generator", {})
    cfg["generator"]["batch_size"] = int("$TRAIN_BATCH_SIZE")
if "$DATA_NUM_WORKERS":
    cfg.setdefault("data", {})
    cfg["data"]["num_workers"] = int("$DATA_NUM_WORKERS")
    cfg["data"]["persistent_workers"] = int("$DATA_NUM_WORKERS") > 0

with open("$EFFECTIVE_CONFIG", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

processed_roots=()
if [[ -d data/hdtf/processed ]] && has_find_match data/hdtf/processed -mindepth 2 -maxdepth 2 -name frames.npy; then
  processed_roots+=(data/hdtf/processed)
fi
if [[ -d "$TALKVID_PROCESSED_ROOT" ]] && has_find_match "$TALKVID_PROCESSED_ROOT" -mindepth 2 -maxdepth 2 -name frames.npy; then
  processed_roots+=("$TALKVID_PROCESSED_ROOT")
fi

echo "[pipeline] Processed roots for SyncNet/generator: ${processed_roots[*]:-<none>}"
if [[ ${#processed_roots[@]} -eq 0 ]]; then
  echo "[pipeline] No processed roots available for training"
  exit 1
fi

SELECTED_SYNCNET_PATH="$OFFICIAL_SYNCNET_PATH"
if [[ "$TRAIN_SYNCNET_EPOCHS" =~ ^[0-9]+$ ]] && (( TRAIN_SYNCNET_EPOCHS > 0 )); then
  SYNCNET_TRAIN_LIST="$OUTPUT_DIR/syncnet_train_speakers.txt"
  SYNCNET_HOLDOUT_LIST="$OUTPUT_DIR/syncnet_holdout_speakers.txt"
  SYNCNET_SPLIT_SUMMARY="$OUTPUT_DIR/syncnet_split_summary.json"
  SYNCNET_COMPARE_JSON="$OUTPUT_DIR/syncnet_teacher_compare.json"

  echo "[pipeline] Building SyncNet split at $(date '+%Y-%m-%d %H:%M:%S %Z')"
  split_cmd=(
    "$PYTHON" -u scripts/build_syncnet_holdout_split.py
    --holdout-count "$SYNCNET_HOLDOUT_COUNT"
    --speaker-list "$COMBINED_SPEAKER_LIST"
    --train-out "$SYNCNET_TRAIN_LIST"
    --holdout-out "$SYNCNET_HOLDOUT_LIST"
    --summary-out "$SYNCNET_SPLIT_SUMMARY"
  )
  for root in "${processed_roots[@]}"; do
    split_cmd+=(--processed-root "$root")
  done
  "${split_cmd[@]}"

  echo "[pipeline] Training SyncNet at $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "$PYTHON" -u scripts/train_syncnet.py \
    --config "$EFFECTIVE_CONFIG" \
    --speaker-list "$SYNCNET_TRAIN_LIST"

  mapfile -t syncnet_ckpts < <(find "$OUTPUT_DIR/syncnet" -maxdepth 1 -name 'syncnet_epoch*.pth' | sort | tail -n "$SYNCNET_COMPARE_LAST_N")
  echo "[pipeline] SyncNet checkpoints considered: ${#syncnet_ckpts[@]}"
  if [[ ${#syncnet_ckpts[@]} -gt 0 ]]; then
    printf '[pipeline] candidate_syncnet=%s\n' "${syncnet_ckpts[@]}"

    echo "[pipeline] Comparing SyncNet teachers at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    compare_cmd=(
      "$PYTHON" -u scripts/compare_syncnet_teachers.py
      --speaker-snapshot "$SYNCNET_TRAIN_LIST"
      --official-checkpoint "$OFFICIAL_SYNCNET_PATH"
      --checkpoints "${syncnet_ckpts[@]}"
      --output "$SYNCNET_COMPARE_JSON"
      --speaker-list "$COMBINED_SPEAKER_LIST"
      --samples "$SYNCNET_COMPARE_SAMPLES"
      --seed "$SYNCNET_COMPARE_SEED"
      --device "$SYNCNET_COMPARE_DEVICE"
      --fps 25
      --T 5
      --img-size 96
      --cache-size 16
    )
    for root in "${processed_roots[@]}"; do
      compare_cmd+=(--processed-root "$root")
    done
    "${compare_cmd[@]}"

    echo "[pipeline] Selecting best SyncNet teacher at $(date '+%Y-%m-%d %H:%M:%S %Z')"
    "$PYTHON" -u scripts/select_best_syncnet_teacher.py \
      --compare-json "$SYNCNET_COMPARE_JSON" \
      --official-checkpoint "$OFFICIAL_SYNCNET_PATH" \
      --checkpoints "${syncnet_ckpts[@]}" \
      --output "$SYNCNET_SELECTED_JSON"

    SELECTED_SYNCNET_PATH="$("$PYTHON" - <<PY
import json
with open("$SYNCNET_SELECTED_JSON") as f:
    data = json.load(f)
print(data["winner_path"])
PY
)"
    echo "[pipeline] Selected SyncNet teacher: $SELECTED_SYNCNET_PATH"
  else
    echo "[pipeline] No local SyncNet checkpoints found, falling back to official teacher"
  fi
else
  echo "[pipeline] TRAIN_SYNCNET_EPOCHS=$TRAIN_SYNCNET_EPOCHS, skipping local SyncNet training and using official teacher"
fi

echo "[pipeline] Starting training watchdog at $(date '+%Y-%m-%d %H:%M:%S %Z') with teacher=$SELECTED_SYNCNET_PATH"
$PYTHON -u scripts/train_with_watchdog.py \
  --config "$EFFECTIVE_CONFIG" \
  --syncnet "$SELECTED_SYNCNET_PATH" \
  --speaker-list "$COMBINED_SPEAKER_LIST" \
  --samples "$WATCHDOG_SAMPLES" \
  --patience "$WATCHDOG_PATIENCE" \
  --degrade-threshold "$WATCHDOG_DEGRADE_THRESHOLD" \
  --min-watchdog-epoch "$WATCHDOG_MIN_EPOCH"

echo "[pipeline] CUDA 3090 HDTF+TalkVid run finished at $(date '+%Y-%m-%d %H:%M:%S %Z')"
