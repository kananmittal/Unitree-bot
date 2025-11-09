# SER Training Guide

## Overview
This project implements a two-tier Speech Emotion Recognition (SER) system:
- **Tier 0**: Audio preprocessing (load → denoise → VAD → MFCC extraction)
- **Tier 1**: Emotion classification model (TsPCA → Conv1D → BiLSTM → FC)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r ser/requirements.txt
```

### 2. Run Training

#### Option A: Using the Shell Script (Recommended)
```bash
# Default parameters
./train.sh

# Custom parameters
./train.sh datasets/processed/iemocap_train.csv datasets/processed/iemocap_val.csv 4 feat_cache 30 16 2e-3 checkpoints
```

#### Option B: Direct Python
```bash
python ser/train.py \
  --train_csv datasets/processed/iemocap_train.csv \
  --val_csv datasets/processed/iemocap_val.csv \
  --num_classes 4 \
  --cache_dir feat_cache \
  --epochs 30 \
  --batch_size 16 \
  --lr 2e-3 \
  --out_dir checkpoints \
  --amp
```

## Dataset Format

The training script supports two CSV formats:

### Format 1: Full Format (with emotion codes)
```csv
file_path,emotion_code,...
datasets/IEMOCAP/Session1/...,happy,...
datasets/IEMOCAP/Session1/...,sad,...
```

### Format 2: Simple Format (with numeric labels)
```csv
path,label
/path/to/audio1.wav,0
/path/to/audio2.wav,1
```

## Emotion Mapping

For 4-class setup:
- `happy`, `surprised`, `excited` → 0
- `neutral` → 1
- `sad`, `fearful`, `fear` → 2
- `angry`, `disgust` → 3

## Features

- **Automatic Feature Caching**: MFCC features are cached to speed up subsequent runs
- **Flexible Dataset Support**: Handles both emotion code and numeric label formats
- **Path Resolution**: Automatically resolves relative paths
- **macOS Compatibility**: Auto-adjusts num_workers for macOS
- **Mixed Precision Training**: Supports AMP for faster training on CUDA

## Output

Training saves checkpoints to the specified output directory:
- `best_uar_<score>.pt`: Best model by UAR (Unweighted Average Recall)
- `last.pt`: Final model after all epochs

## Testing

### Test Full Model on Test Set

#### Option A: Using the Shell Script (Recommended)
```bash
# Test with best checkpoint
./test.sh checkpoints/best_uar_0.7500.pt datasets/processed/iemocap_test.csv

# Save predictions to file
./test.sh checkpoints/best_uar_0.7500.pt datasets/processed/iemocap_test.csv results.json
```

#### Option B: Direct Python
```bash
python ser/test.py \
  --checkpoint checkpoints/best_uar_0.7500.pt \
  --test_csv datasets/processed/iemocap_test.csv \
  --batch_size 16 \
  --output_file results.json
```

### Test Single Audio File
```bash
# Test a single audio file with trained model
python ser/test_single.py \
  datasets/ravdess/Actor_01/03-01-01-01-01-01-01.wav \
  checkpoints/best_uar_0.7500.pt
```

### Testing Individual Components

#### Test Tier 0 (Audio → MFCC)
```bash
python ser/run_tier0.py datasets/ravdess/Actor_01/03-01-01-01-01-01-01.wav
```

#### Test Tier 1 (MFCC → Emotion) - Without trained model
```bash
python ser/run_tier1_demo.py datasets/ravdess/Actor_01/03-01-01-01-01-01-01.wav [optional_model_path]
```

## File Structure

```
ser/
├── tier0_io.py          # Audio preprocessing (load, denoise, VAD, MFCC)
├── tier1_model.py       # SER model architecture
├── datasets.py          # Dataset loading and caching
├── metrics.py           # Evaluation metrics
├── train.py             # Training script
├── run_tier0.py         # Test Tier 0
└── run_tier1_demo.py    # Test Tier 1

train.sh                 # Combined training script
```

## Parameters

- `--train_csv`: Path to training CSV file
- `--val_csv`: Path to validation CSV file (optional, will split train if not provided)
- `--num_classes`: Number of emotion classes (default: 4)
- `--cache_dir`: Directory to cache MFCC features (default: feat_cache)
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 2e-3)
- `--n_mfcc`: Number of MFCC coefficients (default: 40)
- `--out_dir`: Output directory for checkpoints (default: checkpoints)
- `--amp`: Enable Automatic Mixed Precision training

