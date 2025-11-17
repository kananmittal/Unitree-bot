#!/bin/zsh
# Test/Evaluation script for SER model
# 
# Usage:
#   ./test.sh <checkpoint_path> <test_csv>
#   ./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv
#
# Optional flags:
#   --visualize          # Generate visualizations
#   --save_predictions   # Save predictions to CSV

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
    echo "Usage: ./test.sh <checkpoint_path> <test_csv> [--visualize] [--save_predictions]"
    echo ""
    echo "Example:"
    echo "  ./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv"
    echo "  ./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv --visualize --save_predictions"
    exit 1
fi

CHECKPOINT="$1"
TEST_CSV="$2"
shift 2  # Remove first two arguments

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
echo "=========================================="

# Run testing with remaining arguments
python ser/test.py \
  --test_csv "$TEST_CSV" \
  --checkpoint "$CHECKPOINT" \
  "$@"

echo "=========================================="
echo "Testing completed!"
echo "=========================================="

