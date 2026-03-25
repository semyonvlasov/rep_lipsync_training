#!/bin/bash
# Local test: download a small HDTF slice → preprocess → train temporal generator
# → run audio sanity checks → export to CoreML.

set -euo pipefail
TRAINING_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$TRAINING_ROOT"

PYTHON=${PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/musetalk/bin/python}
export KMP_DUPLICATE_LIB_OK=TRUE

SYNCNET_PATH="../../models/wav2lip/checkpoints/lipsync_expert.pth"

echo "=== Step 1: Download 20 HDTF videos ==="
$PYTHON scripts/download_hdtf.py --output data/hdtf --max-videos 20

echo ""
echo "=== Step 2: Preprocess (face crops + mel) ==="
$PYTHON scripts/preprocess_dataset.py \
    --input data/hdtf/clips \
    --output data/hdtf/processed \
    --size 256

echo ""
echo "=== Step 3: Train Generator ==="
$PYTHON -u scripts/train_generator.py \
    --config configs/local_validation.yaml \
    --syncnet "$SYNCNET_PATH"

echo ""
echo "=== Step 4: Audio Sanity Checks ==="
LAST_CKPT=$(ls -t output/training_local/generator/generator_epoch*.pth 2>/dev/null | head -1)
if [ -n "$LAST_CKPT" ]; then
    $PYTHON scripts/check_audio_sensitivity.py \
        --checkpoint "$LAST_CKPT" \
        --config configs/local_validation.yaml \
        --syncnet "$SYNCNET_PATH" \
        --samples 8
else
    echo "No checkpoint found — training may have failed"
    exit 1
fi

echo ""
echo "=== Step 5: Export to CoreML ==="
if [ -n "$LAST_CKPT" ]; then
    $PYTHON scripts/export_coreml.py \
        --checkpoint "$LAST_CKPT" \
        --config configs/local_validation.yaml \
        --output output/training_local/lipsync_192_fp16.mlpackage
    echo "CoreML model exported!"
else
    echo "No checkpoint found — training may have failed"
    exit 1
fi

echo ""
echo "=== Done! ==="
echo "Checkpoint: $LAST_CKPT"
echo "CoreML: output/training_local/lipsync_192_fp16.mlpackage"
