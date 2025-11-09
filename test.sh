#!/bin/zsh
# Test/Evaluation script for SER model
# 
# Usage:
#   ./test.sh <checkpoint_path> <test_csv>
#   ./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv
#
# Optional environment variables:
#   SAVE_PREDS=true  # Save predictions to CSV

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: ./test.sh <checkpoint_path> <test_csv> [batch_size] [cache_dir]"
    echo ""
    echo "Example:"
    echo "  ./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv"
    exit 1
fi

CHECKPOINT="$1"
TEST_CSV="$2"
BATCH_SIZE="${3:-16}"
CACHE_DIR="${4:-feat_cache}"

# Check if files exist
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$TEST_CSV" ]; then
    echo "Error: Test CSV not found: $TEST_CSV"
    exit 1
fi

echo "=========================================="
echo "SER Model Testing"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Test CSV: $TEST_CSV"
echo "Batch Size: $BATCH_SIZE"
echo "Cache Dir: $CACHE_DIR"
echo "=========================================="

# Run testing
python ser/test.py \
  --test_csv "$TEST_CSV" \
  --checkpoint "$CHECKPOINT" \
  --batch_size $BATCH_SIZE \
  --cache_dir "$CACHE_DIR"

echo "=========================================="
echo "Testing completed!"
echo "=========================================="
