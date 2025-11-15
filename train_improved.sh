#!/bin/bash
# Enhanced training script with config support
# 
# Usage:
#   ./train_improved.sh                                    # Use default config
#   ./train_improved.sh --config config.yaml               # Use custom config
#   ./train_improved.sh --train_csv train.csv --epochs 50  # Override specific params

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if Python script exists
if [ ! -f "ser/train_improved.py" ]; then
    echo "Error: train_improved.py not found"
    exit 1
fi

echo "=========================================="
echo "SER Enhanced Training Script"
echo "=========================================="

# Run training with all arguments passed through
python ser/train_improved.py "$@"

echo "=========================================="
echo "Training completed!"
echo "=========================================="