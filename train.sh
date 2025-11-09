#!/bin/zsh
# Combined training script for SER (Speech Emotion Recognition)
# 
# Usage:
#   ./train.sh                                    # Use defaults
#   ./train.sh <train_csv> <val_csv>              # Specify CSV files
#   ./train.sh <train_csv> <val_csv> <num_classes> <cache_dir> <epochs> <batch_size> <lr> <out_dir>
#
# Example:
#   ./train.sh datasets/processed/iemocap_train.csv datasets/processed/iemocap_val.csv 4 feat_cache 30 16 2e-3 checkpoints

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Default parameters
TRAIN_CSV="${1:-datasets/processed/iemocap_train.csv}"
VAL_CSV="${2:-datasets/processed/iemocap_val.csv}"
NUM_CLASSES="${3:-4}"
CACHE_DIR="${4:-feat_cache}"
EPOCHS="${5:-30}"
BATCH_SIZE="${6:-16}"
LR="${7:-2e-3}"
OUT_DIR="${8:-checkpoints}"

# Check if CSV files exist
if [ ! -f "$TRAIN_CSV" ]; then
    echo "Error: Train CSV not found: $TRAIN_CSV"
    exit 1
fi

if [ ! -f "$VAL_CSV" ]; then
    echo "Error: Val CSV not found: $VAL_CSV"
    exit 1
fi

echo "=========================================="
echo "SER Training Script"
echo "=========================================="
echo "Train CSV: $TRAIN_CSV"
echo "Val CSV: $VAL_CSV"
echo "Num Classes: $NUM_CLASSES"
echo "Cache Dir: $CACHE_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Output Dir: $OUT_DIR"
echo "=========================================="

# Run training (from project root, ser/train.py handles paths)
python ser/train.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --num_classes $NUM_CLASSES \
  --cache_dir "$CACHE_DIR" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --out_dir "$OUT_DIR" \
  --amp

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUT_DIR"
echo "=========================================="

