# Speech Emotion Recognition (SER) System

A comprehensive two-tier Speech Emotion Recognition system that processes raw audio through advanced preprocessing and classifies emotions using a deep learning architecture.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Tier 0: Audio Preprocessing](#tier-0-audio-preprocessing)
  - [Tier 1: Emotion Classification Model](#tier-1-emotion-classification-model)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Format](#dataset-format)
- [Training](#training)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Metrics Explained](#metrics-explained)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Overview

This project implements a two-tier Speech Emotion Recognition (SER) system:

- **Tier 0**: Audio preprocessing pipeline (load → denoise → VAD → MFCC extraction)
- **Tier 1**: Emotion classification model (TsPCA → Conv1D → BiLSTM → FC)

The system is designed to handle real-world audio with noise and silence, automatically extracting meaningful features and classifying emotions into multiple categories (default: 4 classes).

## Architecture

### Tier 0: Audio Preprocessing

Tier 0 transforms raw audio files into MFCC (Mel-Frequency Cepstral Coefficients) features through a multi-stage preprocessing pipeline.

#### 1. Audio Loading (`load_audio`)
- **Input**: Audio file path
- **Process**:
  - Loads audio using `torchaudio.load()` (supports various formats)
  - Converts stereo to mono by averaging channels
  - Resamples to target sample rate (16 kHz) if needed
  - Normalizes amplitude to [-1, 1] range
- **Output**: Mono audio waveform at 16 kHz, shape `(T,)` where T is number of samples

#### 2. Denoising (`denoise_demucs`)
- **Model**: Facebook Research's Demucs DNS64 (Deep Noise Suppression)
- **Process**:
  - Uses pre-trained DNS64 model for speech denoising
  - Removes background noise, static, and other audio artifacts
  - Operates on 16 kHz audio
- **Output**: Denoised audio waveform, shape `(T,)`

#### 3. Voice Activity Detection (`vad_silero_keep_speech`)
- **Model**: Silero VAD (Voice Activity Detection)
- **Process**:
  - Detects speech segments using Silero VAD model
  - Removes silence and non-speech portions
  - Concatenates detected speech chunks
- **Output**: Audio containing only speech segments, shape `(T',)` where T' ≤ T

#### 4. MFCC Extraction (`mfcc_40`)
- **Parameters**:
  - `n_mfcc`: Number of MFCC coefficients (default: 40)
  - `n_fft`: 512 (FFT window size)
  - `hop_length`: 160 samples (10ms at 16 kHz)
  - `win_length`: 400 samples (25ms at 16 kHz)
  - `f_min`: 20 Hz (minimum frequency)
  - `f_max`: 8000 Hz (Nyquist frequency at 16 kHz)
  - `window_fn`: Hamming window
- **Process**:
  - Computes Mel-Frequency Cepstral Coefficients
  - MFCCs capture spectral characteristics of speech
  - Each coefficient represents different aspects of the audio spectrum
- **Output**: MFCC features, shape `(C, T_frames)` where C=40 (coefficients) and T_frames is number of time frames

#### Complete Tier 0 Pipeline
```python
def tier0_to_mfcc(path, device="cpu", n_mfcc=40):
    wav = load_audio(path)                    # Load and normalize
    den = denoise_demucs(wav, device=device)  # Remove noise
    speech = vad_silero_keep_speech(den)      # Keep only speech
    return mfcc_40(speech, n_mfcc=n_mfcc)     # Extract MFCC features
```

**Key Features**:
- Automatic noise removal for robust feature extraction
- Silence removal to focus on meaningful audio content
- Efficient feature representation (40 coefficients vs. raw audio samples)
- Caching support for faster subsequent processing

### Tier 1: Emotion Classification Model

Tier 1 is a deep neural network that classifies MFCC features into emotion categories.

#### Model Architecture: `SER_Tier1` (Improved)

**Input**: MFCC features `(B, C, T)` where:
- `B`: Batch size
- `C`: Number of MFCC coefficients (40)
- `T`: Number of time frames (variable)

**Key Improvements**:
- Multi-head attention (4 heads) for better feature representation
- Residual connections in CNN blocks for improved gradient flow
- Deeper architecture (128→256 channels, 2 LSTM layers)
- Attention-based temporal pooling instead of simple mean pooling

**Architecture Components**:

1. **Multi-Head Channel Attention** (Improved)
   - **Purpose**: Channel-wise attention mechanism to focus on important MFCC coefficients
   - **Process**:
     - Uses 4 attention heads (configurable) for richer feature representation
     - Computes two streams: average pooling and max pooling over time dimension
     - Each head: `(B, C, T)` → `(B, C)` (pooling) → MLP → `(B, C)` (attention weights)
     - Heads are combined via learned linear layer
     - Applies attention: `x * weights.unsqueeze(2)`
   - **Output**: Attention-weighted MFCC features, shape `(B, C, T)`

2. **CNN Stack (Convolutional Neural Network)** (Improved)
   - **Architecture**: Stack of 1D convolutional blocks with residual connections
   - **Default**: 2 blocks with 128→256 channels (deeper network)
   - **ResidualConv1DBlock**:
     - **Depthwise-Separable Convolution** (default):
       - Depthwise conv: `Conv1d(in_ch, in_ch, kernel=5, groups=in_ch)` - processes each channel separately
       - Pointwise conv: `Conv1d(in_ch, out_ch, kernel=1)` - combines channels
       - Benefits: Fewer parameters, faster training, less overfitting
     - **Regular Convolution** (optional):
       - Standard `Conv1d(in_ch, out_ch, kernel=5)`
     - **Normalization**: BatchNorm1d
     - **Activation**: GELU (Gaussian Error Linear Unit)
     - **Regularization**: Dropout (default: 0.1)
   - **Output**: Convolved features, shape `(B, F, T)` where F is number of output channels (128)

3. **BiLSTM (Bidirectional Long Short-Term Memory)** (Improved)
   - **Purpose**: Captures temporal dependencies in speech
   - **Architecture**:
     - Input size: F (from CNN output)
     - Hidden size: 128 (default)
     - Layers: 2 (deeper for better temporal modeling)
     - Bidirectional: Yes (forward + backward)
     - Output size: 2 × hidden_size = 256
   - **Process**:
     - Input: `(B, T, F)` (transposed from CNN output)
     - Output: `(B, T, 2H)` where H=128
     - Temporal pooling: Attention-based pooling (learned weights) → `(B, 2H)`
   - **Output**: Sequence representation, shape `(B, 256)`

4. **Classification Head**
   - **Structure**:
     - LayerNorm: Normalizes features
     - Dropout: 0.1 (regularization)
     - Linear: `256 → num_classes` (default: 4)
   - **Output**: Logits, shape `(B, num_classes)`

5. **Temperature Scaling** (optional)
   - **Purpose**: Calibrates confidence scores
   - **Process**: `logits / T` where T is learned temperature parameter
   - **Benefits**: Better calibrated probabilities for confidence estimation

6. **Output Processing**
   - **Logits**: Raw classification scores
   - **Probabilities**: `softmax(logits)` - probability distribution over classes
   - **Confidence**: `max(probabilities)` - confidence in predicted class

#### Complete Forward Pass

```python
def forward(self, mfcc_CT):
    # mfcc_CT: (B, C, T)
    x = self.attn(mfcc_CT)           # (B, C, T) - Channel attention
    x = self.cnn(x)                   # (B, F, T) - Convolutional features
    x = x.transpose(1, 2)             # (B, T, F) - Prepare for RNN
    y, _ = self.rnn(x)                # (B, T, 2H) - Temporal modeling
    y = y.mean(dim=1)                 # (B, 2H) - Temporal pooling
    logits = self.cls_head(y)        # (B, K) - Classification
    logits = self.temp(logits)        # (B, K) - Temperature scaling
    probs = torch.softmax(logits, dim=-1)  # (B, K) - Probabilities
    confidence = probs.max(dim=-1)[0] # (B,) - Confidence scores
    return {"logits": logits, "probs": probs, "confidence": confidence}
```

#### Model Parameters

- **n_mfcc**: 40 (number of MFCC coefficients)
- **num_classes**: 4 (emotion categories)
- **conv_channels**: (128, 128) (CNN output channels)
- **lstm_hidden**: 128 (LSTM hidden size)
- **dropout**: 0.1 (dropout rate)
- **use_depthwise**: True (use depthwise-separable convolutions)
- **use_temp_scale**: True (enable temperature scaling)

#### Loss Function: `SERLoss`

- **Primary Loss**: CrossEntropyLoss for emotion classification
- **Optional**: SmoothL1Loss for valence-arousal regression (multitask learning)
- **Combined**: `loss = CE_loss + λ * L1_loss` (if multitask enabled)

#### Key Design Decisions

1. **Two-Tier Architecture**: Separates feature extraction (Tier 0) from classification (Tier 1)
2. **Attention Mechanism**: TsPCA focuses on important MFCC coefficients
3. **Depthwise-Separable Convolutions**: Reduces parameters while maintaining performance
4. **Bidirectional LSTM**: Captures both forward and backward temporal dependencies
5. **Temporal Pooling**: Mean pooling aggregates sequence information
6. **Temperature Scaling**: Calibrates confidence for better uncertainty estimation

## Installation

### 1. Install Dependencies

```bash
pip install -r ser/requirements.txt
```

**Required Packages**:
- `torch` and `torchaudio`: Deep learning framework and audio processing
- `soundfile`: Audio I/O operations
- `denoiser` (from GitHub): Facebook Research's denoising model
- `pandas`: Data handling
- `scikit-learn`: Evaluation metrics

**Note**: The denoiser package is installed from GitHub:
```bash
pip install git+https://github.com/facebookresearch/denoiser.git
```

### 2. System Requirements

- **Python**: 3.7+
- **CUDA**: Optional, for GPU acceleration
- **macOS Compatibility**: Automatically adjusts `num_workers` for macOS multiprocessing

## Quick Start

### Training a Model

#### Option A: Using Configuration File (Recommended)
```bash
# Create default config
python -c "from ser.config import create_default_config; create_default_config('config.yaml')"

# Train with config
./train.sh --config config.yaml

# Or override specific parameters
./train.sh --config config.yaml --train_csv train.csv --epochs 50
```

#### Option B: Using Shell Script with Arguments
```bash
# With config file
./train.sh --config config.yaml

# With command-line arguments
./train.sh --train_csv datasets/processed/iemocap_train.csv \
           --val_csv datasets/processed/iemocap_val.csv \
           --epochs 50 \
           --batch_size 16
```

#### Option C: Direct Python
```bash
python ser/train.py \
  --config config.yaml \
  --train_csv datasets/processed/iemocap_train.csv \
  --val_csv datasets/processed/iemocap_val.csv \
  --epochs 50 \
  --batch_size 16 \
  --amp
```

### Testing a Model

#### Test on Full Test Set

**Using Shell Script:**
```bash
./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv

# With visualizations
./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv --visualize --save_predictions
```

**Using Python Directly:**
```bash
python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt \
  --batch_size 16 \
  --cache_dir feat_cache \
  --visualize \
  --save_predictions
```

#### Test on Single Audio File
```bash
# With trained model
python ser/run_tier1_demo.py audio.wav checkpoints/best_uar_0.8500.pt

# Without model (uses random weights for testing)
python ser/run_tier1_demo.py audio.wav
```

## Dataset Format

The training and testing scripts support two CSV formats:

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

**Path Resolution**:
- Supports both absolute and relative paths
- Automatically resolves relative paths relative to project root
- Handles paths relative to CSV file location

## Emotion Mapping

For 4-class setup (default):
- **Class 0 (Happy)**: `happy`, `surprised`, `excited`
- **Class 1 (Neutral)**: `neutral`
- **Class 2 (Sad)**: `sad`, `fearful`, `fear`
- **Class 3 (Angry)**: `angry`, `disgust`

**Dynamic Mapping**: The system automatically creates emotion mappings from datasets. If your CSV contains `emotion_code` column, it will create a mapping to contiguous class labels (0, 1, 2, ...).

## Training

### Features

- **Configuration System**: YAML-based configuration for easy hyperparameter management
- **Data Augmentation**: Time stretching, pitch shifting, noise injection, mixup, SpecAugment
- **Learning Rate Scheduling**: Cosine annealing, plateau reduction, step decay with warmup
- **Early Stopping**: Prevents overfitting with configurable patience
- **Gradient Clipping**: Stabilizes training
- **Label Smoothing**: Improves generalization
- **Class Weighting**: Handles imbalanced datasets
- **Automatic Feature Caching**: MFCC features are cached to speed up subsequent runs
- **Flexible Dataset Support**: Handles both emotion code and numeric label formats
- **Path Resolution**: Automatically resolves relative paths
- **macOS Compatibility**: Auto-adjusts num_workers for macOS
- **Mixed Precision Training**: Supports AMP (Automatic Mixed Precision) for faster training on CUDA
- **Auto-Detection**: Automatically detects number of classes from dataset if not specified
- **Visualization**: Confusion matrices and per-class metrics plots
- **Error Analysis**: Detailed error analysis with most confident errors

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train_csv` | Path to training CSV file | Required |
| `--val_csv` | Path to validation CSV file | Optional (splits train if not provided) |
| `--num_classes` | Number of emotion classes | Auto-detected from dataset |
| `--cache_dir` | Directory to cache MFCC features | `feat_cache` |
| `--epochs` | Number of training epochs | 30 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate | 2e-3 |
| `--n_mfcc` | Number of MFCC coefficients | 40 |
| `--val_split` | Validation split ratio (if val_csv not provided) | 0.1 |
| `--num_workers` | Number of data loader workers | 2 (auto-adjusted for macOS) |
| `--out_dir` | Output directory for checkpoints | `checkpoints` |
| `--amp` | Enable Automatic Mixed Precision training | False |

### Training Output

Training saves checkpoints to the specified output directory:
- `best_uar_<score>.pt`: Best model by UAR (Unweighted Average Recall)
- `last.pt`: Final model after all epochs

**Checkpoint Format**:
```python
{
    "model": model.state_dict(),
    "args": {
        "num_classes": 4,
        "n_mfcc": 40,
        ...
    }
}
```

### Training Process

1. **Data Loading**: Loads audio files and extracts MFCC features (with caching)
2. **Model Initialization**: Creates `SER_Tier1` model with specified parameters
3. **Optimization**: Uses AdamW optimizer with specified learning rate
4. **Training Loop**:
   - Forward pass through model
   - Compute loss (CrossEntropyLoss)
   - Backward pass and optimization
   - Validation after each epoch
5. **Checkpointing**: Saves best model (by UAR) and final model

## Testing

### Test Output

#### Full Test Set Evaluation
The test script provides:
- **Overall Metrics**: Accuracy, UAR (Unweighted Average Recall)
- **Confusion Matrix**: Shows predictions vs true labels
- **Per-Class Metrics**: Precision, Recall, F1-Score for each emotion class

#### Single File Inference
Provides:
- Predicted emotion class
- Confidence score
- Probability distribution over all classes

### Finding Checkpoints

After training, checkpoints are saved in the `checkpoints/` directory (or your specified `--out_dir`):
- `best_uar_<score>.pt`: Best model by UAR metric
- `last.pt`: Final model after all epochs

List available checkpoints:
```bash
ls -lh checkpoints/
```

### Advanced Testing Options

#### Save Predictions to CSV
```bash
SAVE_PREDS=true python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt
```

This creates a file `checkpoints/best_uar_0.8500_predictions.csv` with:
- `true_label`: Ground truth label
- `pred_label`: Predicted label
- `correct`: Whether prediction is correct

#### Custom Batch Size
```bash
python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt \
  --batch_size 32
```

#### Use Specific Device
```bash
python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt \
  --device cuda
```

## Testing Individual Components

### Test Tier 0 (Audio → MFCC)
```bash
python ser/run_tier0.py datasets/ravdess/Actor_01/03-01-01-01-01-01-01.wav
```

This will:
1. Load the audio file
2. Apply denoising
3. Apply VAD
4. Extract MFCC features
5. Display feature statistics

### Test Tier 1 (MFCC → Emotion) - Without trained model
```bash
python ser/run_tier1_demo.py datasets/ravdess/Actor_01/03-01-01-01-01-01-01.wav [optional_model_path]
```

This will:
1. Load audio and extract MFCC (using Tier 0)
2. Run through Tier 1 model
3. Display prediction and confidence

## Project Structure

```
ser/
├── tier0_io.py          # Audio preprocessing (load, denoise, VAD, MFCC)
├── tier1_model.py       # SER model architecture (TsPCA, CNN, BiLSTM, Head)
├── datasets.py          # Dataset loading and caching
├── metrics.py           # Evaluation metrics (accuracy, UAR, precision, recall, F1)
├── train.py             # Training script
├── test.py              # Testing script
├── test_single.py       # Single file testing
├── run_tier0.py         # Test Tier 0 pipeline
└── run_tier1_demo.py    # Test Tier 1 pipeline

train.sh                 # Training shell script
test.sh                  # Testing shell script
checkpoints/             # Saved model checkpoints
feat_cache/              # Cached MFCC features
datasets/                # Audio datasets
```

## Metrics Explained

### Accuracy
- **Definition**: Overall correctness (correct predictions / total predictions)
- **Formula**: `(TP + TN) / (TP + TN + FP + FN)`
- **Use Case**: General performance metric, but can be misleading with imbalanced classes

### UAR (Unweighted Average Recall)
- **Definition**: Average recall across all classes (macro recall)
- **Formula**: `mean(recall_per_class)`
- **Use Case**: Better metric for imbalanced datasets, treats all classes equally

### Precision
- **Definition**: Of all predictions for a class, how many were correct
- **Formula**: `TP / (TP + FP)`
- **Use Case**: Measures how reliable positive predictions are

### Recall
- **Definition**: Of all true instances of a class, how many were found
- **Formula**: `TP / (TP + FN)`
- **Use Case**: Measures how well the model finds all instances of a class

### F1-Score
- **Definition**: Harmonic mean of precision and recall
- **Formula**: `2 * (precision * recall) / (precision + recall)`
- **Use Case**: Balanced metric combining precision and recall

## Troubleshooting

### Checkpoint Not Found
```bash
# Check if checkpoint exists
ls -lh checkpoints/

# Use absolute path if needed
python ser/test.py --test_csv test.csv --checkpoint /absolute/path/to/checkpoint.pt
```

### Test CSV Format
Make sure your test CSV has the same format as training CSV:
- Either `file_path,emotion_code` format
- Or `path,label` format with numeric labels

### Out of Memory
Reduce batch size:
```bash
python ser/train.py --train_csv train.csv --val_csv val.csv --batch_size 8
python ser/test.py --test_csv test.csv --checkpoint model.pt --batch_size 8
```

### Audio Loading Errors
- Check audio file format (supports WAV, MP3, FLAC, etc.)
- Verify file paths are correct
- Check file permissions
- Ensure audio files are not corrupted

### Feature Caching Issues
- Clear cache directory if features are corrupted:
  ```bash
  rm -rf feat_cache/
  ```
- Cache files are automatically regenerated if corrupted

### macOS Multiprocessing Issues
- The system automatically sets `num_workers=0` on macOS
- If you encounter issues, manually set `--num_workers 0`

## Examples

### Complete Training and Testing Workflow

```bash
# 1. Train a model
./train.sh datasets/processed/iemocap_train.csv datasets/processed/iemocap_val.csv 4 feat_cache 30 16 2e-3 checkpoints

# 2. Test on test set
./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv

# 3. Test single audio file
python ser/run_tier1_demo.py datasets/ravdess/Actor_01/03-01-01-01-01-01-01.wav checkpoints/best_uar_0.8500.pt
```

### Test Multiple Files
```bash
for file in datasets/ravdess/Actor_01/*.wav; do
    echo "Testing: $file"
    python ser/run_tier1_demo.py "$file" checkpoints/best_uar_0.8500.pt
    echo "---"
done
```

### Training with Custom Parameters
```bash
python ser/train.py \
  --train_csv datasets/processed/iemocap_train.csv \
  --val_csv datasets/processed/iemocap_val.csv \
  --num_classes 4 \
  --cache_dir feat_cache \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --out_dir checkpoints \
  --amp
```

### Testing with Predictions Saved
```bash
SAVE_PREDS=true python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt \
  --batch_size 32
```

## Additional Notes

### Model Architecture Details

- **TsPCA Attention**: The two-stream pooling (average + max) provides complementary information about feature importance
- **Depthwise-Separable Convolutions**: Reduces parameters by ~70% compared to standard convolutions
- **Bidirectional LSTM**: Captures both past and future context for better emotion recognition
- **Temporal Pooling**: Mean pooling aggregates information across the entire sequence

### Performance Considerations

- **Feature Caching**: MFCC extraction is the slowest step; caching significantly speeds up training
- **Mixed Precision**: AMP can provide 1.5-2x speedup on modern GPUs
- **Batch Size**: Larger batches improve GPU utilization but require more memory
- **Data Loading**: Multiple workers speed up data loading (except on macOS)

### Future Enhancements

- Multitask learning with valence-arousal regression
- Support for more emotion classes
- Real-time inference capabilities
- Model quantization for deployment
- Transfer learning from pre-trained models
