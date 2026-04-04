#!/bin/bash
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$TRAINING_ROOT"

OUT="${OUT:-output/syncnet_sidecar_snapshot_20260324_medium}"
BASE_CONFIG="${BASE_CONFIG:-configs/syncnet_cuda3090_medium.yaml}"
HDTF_PROCESSED_ROOT="${HDTF_PROCESSED_ROOT:-data/hdtf/processed}"
TALKVID_PROCESSED_ROOT="${TALKVID_PROCESSED_ROOT:-data/talkvid/processed}"
TALKVID_QUALITY_LIST="${TALKVID_QUALITY_LIST:-output/talkvid_quality_20260324/min_medium.txt}"
OFFICIAL_SYNCNET_PATH="${OFFICIAL_SYNCNET_PATH:-../models/official_syncnet/checkpoints/lipsync_expert.pth}"
SYNCNET_EPOCHS="${SYNCNET_EPOCHS:-12}"
SYNCNET_BATCH_SIZE="${SYNCNET_BATCH_SIZE:-64}"
SYNCNET_COMPARE_LAST_N="${SYNCNET_COMPARE_LAST_N:-3}"
SYNCNET_COMPARE_SAMPLES="${SYNCNET_COMPARE_SAMPLES:-20}"
SYNCNET_COMPARE_SEED="${SYNCNET_COMPARE_SEED:-123}"
SYNCNET_COMPARE_DEVICE="${SYNCNET_COMPARE_DEVICE:-cuda}"
SYNCNET_HOLDOUT_COUNT="${SYNCNET_HOLDOUT_COUNT:-10}"

mkdir -p "$OUT"
LOG="$OUT/pipeline.log"
: > "$LOG"
exec >> "$LOG" 2>&1

echo "[sidecar] started at $(date '+%Y-%m-%d %H:%M:%S %Z')"

EFFECTIVE_CONFIG="$OUT/effective_syncnet_config.yaml"
HDTF_LIST="$OUT/hdtf_all_speakers_snapshot.txt"
TALKVID_LIST="$OUT/talkvid_min_medium_snapshot.txt"
COMBINED_LIST="$OUT/combined_speakers_snapshot.txt"
TRAIN_LIST="$OUT/syncnet_train_speakers.txt"
HOLDOUT_LIST="$OUT/syncnet_holdout_speakers.txt"
SPLIT_SUMMARY="$OUT/syncnet_split_summary.json"
COMPARE_JSON="$OUT/syncnet_teacher_compare.json"
SELECTED_JSON="$OUT/syncnet_selected_teacher.json"

python3 - <<PY
import os
import yaml

out_dir = os.path.abspath("$OUT")
base_cfg = yaml.safe_load(open("$BASE_CONFIG"))
base_cfg.setdefault("training", {})
base_cfg["training"]["output_dir"] = out_dir
base_cfg.setdefault("syncnet", {})
base_cfg["syncnet"]["epochs"] = int("$SYNCNET_EPOCHS")
base_cfg["syncnet"]["batch_size"] = int("$SYNCNET_BATCH_SIZE")
base_cfg.setdefault("data", {})
base_cfg["data"]["hdtf_root"] = "$HDTF_PROCESSED_ROOT"
base_cfg["data"]["talkvid_root"] = "$TALKVID_PROCESSED_ROOT"
base_cfg["data"]["num_workers"] = int(base_cfg["data"].get("num_workers", 8))
base_cfg["data"]["persistent_workers"] = True
with open("$EFFECTIVE_CONFIG", "w") as f:
    yaml.safe_dump(base_cfg, f, sort_keys=False)

hdtf_root = "$HDTF_PROCESSED_ROOT"
hdtf = []
for name in sorted(os.listdir(hdtf_root)):
    d = os.path.join(hdtf_root, name)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames.npy")) and os.path.exists(os.path.join(d, "mel.npy")):
        hdtf.append(name)
with open("$HDTF_LIST", "w") as f:
    for name in hdtf:
        f.write(name + "\\n")

with open("$TALKVID_QUALITY_LIST") as src:
    talk = [line.strip() for line in src if line.strip()]
with open("$TALKVID_LIST", "w") as dst:
    for name in talk:
        dst.write(name + "\\n")

combined = sorted(set(hdtf + talk))
with open("$COMBINED_LIST", "w") as f:
    for name in combined:
        f.write(name + "\\n")

print(f"[sidecar] hdtf_snapshot={len(hdtf)}", flush=True)
print(f"[sidecar] talkvid_snapshot={len(talk)}", flush=True)
print(f"[sidecar] combined_snapshot={len(combined)}", flush=True)
print(f"[sidecar] effective_config=$EFFECTIVE_CONFIG", flush=True)
PY

echo "[sidecar] building split at $(date '+%Y-%m-%d %H:%M:%S %Z')"
python3 -u scripts/build_syncnet_holdout_split.py \
  --processed-root "$HDTF_PROCESSED_ROOT" \
  --processed-root "$TALKVID_PROCESSED_ROOT" \
  --speaker-list "$COMBINED_LIST" \
  --holdout-count "$SYNCNET_HOLDOUT_COUNT" \
  --train-out "$TRAIN_LIST" \
  --holdout-out "$HOLDOUT_LIST" \
  --summary-out "$SPLIT_SUMMARY"

echo "[sidecar] training syncnet at $(date '+%Y-%m-%d %H:%M:%S %Z')"
python3 -u scripts/train_syncnet.py \
  --config "$EFFECTIVE_CONFIG" \
  --speaker-list "$TRAIN_LIST"

mapfile -t syncnet_ckpts < <(find "$OUT/syncnet" -maxdepth 1 -name 'syncnet_epoch*.pth' | sort | tail -n "$SYNCNET_COMPARE_LAST_N")
if [[ ${#syncnet_ckpts[@]} -eq 0 ]]; then
  echo "[sidecar] ERROR: no syncnet checkpoints found"
  exit 1
fi
printf '[sidecar] candidate_syncnet=%s\n' "${syncnet_ckpts[@]}"

echo "[sidecar] comparing teachers at $(date '+%Y-%m-%d %H:%M:%S %Z')"
compare_cmd=(
  python3 -u scripts/compare_syncnet_teachers.py
  --speaker-snapshot "$TRAIN_LIST"
  --official-checkpoint "$OFFICIAL_SYNCNET_PATH"
  --output "$COMPARE_JSON"
  --speaker-list "$COMBINED_LIST"
  --samples "$SYNCNET_COMPARE_SAMPLES"
  --seed "$SYNCNET_COMPARE_SEED"
  --device "$SYNCNET_COMPARE_DEVICE"
  --fps 25
  --T 5
  --img-size 96
  --cache-size 16
  --processed-root "$HDTF_PROCESSED_ROOT"
  --processed-root "$TALKVID_PROCESSED_ROOT"
)
for ckpt in "${syncnet_ckpts[@]}"; do
  compare_cmd+=(--checkpoints "$ckpt")
done
"${compare_cmd[@]}"

echo "[sidecar] selecting teacher at $(date '+%Y-%m-%d %H:%M:%S %Z')"
select_cmd=(
  python3 -u scripts/select_best_syncnet_teacher.py
  --compare-json "$COMPARE_JSON"
  --official-checkpoint "$OFFICIAL_SYNCNET_PATH"
  --output "$SELECTED_JSON"
)
for ckpt in "${syncnet_ckpts[@]}"; do
  select_cmd+=(--checkpoints "$ckpt")
done
"${select_cmd[@]}"

echo "[sidecar] finished at $(date '+%Y-%m-%d %H:%M:%S %Z')"
