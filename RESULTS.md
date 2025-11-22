# Speech Emotion Recognition - Training Results

This document summarizes the training, validation, and test results across all datasets.

---

## Summary Table

| Dataset | Classes | Train Samples | Val Samples | Test Samples | Best Val UAR | Test UAR | Test Accuracy |
|---------|---------|---------------|-------------|--------------|--------------|----------|---------------|
| **CREMA-D** | 6 | 5,480 | 896 | 1,054 | **65.56%** | **56.69%** | **56.45%** |
| **RAVDESS** | 8 | 1,152 | 288 | - | **72.03%** | - | - |
| **SAVEE** | 7 | 360 | 120 | - | **22.86%** | - | - |
| **IEMOCAP** | - | - | - | - | - | - | - |

> **Note**: IEMOCAP results not available in current workspace. RAVDESS and SAVEE test results pending.

---

## CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

### Dataset Information
- **Classes**: 6 emotions (Anger, Disgust, Fear, Happy, Neutral, Sad)
- **Training Samples**: 5,480
- **Validation Samples**: 896
- **Test Samples**: 1,054
- **Audio Format**: 16kHz, mono

### Training Configuration
- **Model**: Tier-1 SER (Conv + LSTM + Attention)
- **Epochs**: 30 (resumed from epoch 3)
- **Batch Size**: 16
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.0001)
- **Scheduler**: Cosine Annealing
- **Augmentation**: Time stretch, pitch shift, noise injection, mixup
- **Device**: CPU (macOS)
- **Training Time**: ~54 hours

### Validation Results (Best Checkpoint)
- **Checkpoint**: `best_uar_0.6556.pt`
- **Epoch**: 26
- **UAR**: **65.56%**
- **Accuracy**: 64.73%
- **Train Loss**: 0.9792
- **Val Loss**: 1.2011

### Test Results
- **UAR**: **56.69%**
- **Accuracy**: **56.45%**
- **Macro Precision**: 57.67%
- **Macro F1-Score**: 56.48%

#### Per-Class Performance (Test Set)
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Anger (0) | 73.62% | 66.30% | 69.77% | 181 |
| Disgust (1) | 57.14% | 53.04% | 55.01% | 181 |
| Fear (2) | 57.52% | **35.91%** | 44.22% | 181 |
| Happy (3) | 45.02% | 58.10% | 50.73% | 179 |
| Neutral (4) | 62.73% | 65.58% | 64.13% | 154 |
| Sad (5) | 50.00% | 61.24% | 55.05% | 178 |

**Key Observations**:
- Best performing: Anger (69.77% F1)
- Most challenging: Fear (35.91% recall - often confused with Happy/Sad)
- Gap between validation and test: ~9% (suggests some overfitting)

### Training Progress
| Checkpoint | Epoch | Val UAR | Improvement |
|------------|-------|---------|-------------|
| Starting | 3 | 50.74% | - |
| best_uar_0.5365.pt | - | 53.65% | +2.91% |
| best_uar_0.5664.pt | - | 56.64% | +5.90% |
| best_uar_0.6183.pt | - | 61.83% | +11.09% |
| best_uar_0.6403.pt | - | 64.03% | +13.29% |
| **best_uar_0.6556.pt** | 26 | **65.56%** | **+14.82%** |

---

## RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

### Dataset Information
- **Classes**: 8 emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
- **Training Samples**: 1,152
- **Validation Samples**: 288
- **Test Samples**: Not evaluated yet
- **Audio Format**: 16kHz, mono

### Training Configuration
- **Model**: Tier-1 SER (Conv + LSTM + Attention)
- **Epochs**: 30
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing
- **Augmentation**: Enabled

### Validation Results (Best Checkpoint)
- **Checkpoint**: `best_uar_0.7203.pt`
- **Epoch**: 17
- **UAR**: **72.03%**
- **Accuracy**: 72.22%
- **Train Loss**: 1.0115
- **Val Loss**: 1.1318

### Training Progress
| Checkpoint | Val UAR |
|------------|---------|
| best_uar_0.2725.pt | 27.25% |
| best_uar_0.3841.pt | 38.41% |
| best_uar_0.4506.pt | 45.06% |
| best_uar_0.5438.pt | 54.38% |
| best_uar_0.6174.pt | 61.74% |
| best_uar_0.6891.pt | 68.91% |
| **best_uar_0.7203.pt** | **72.03%** |

**Note**: Test evaluation pending.

---

## SAVEE (Surrey Audio-Visual Expressed Emotion)

### Dataset Information
- **Classes**: 7 emotions (Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise)
- **Training Samples**: 360
- **Validation Samples**: 120
- **Test Samples**: Not evaluated yet
- **Audio Format**: 16kHz, mono

### Training Configuration
- **Model**: Tier-1 SER (Conv + LSTM + Attention)
- **Epochs**: 30
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing
- **Augmentation**: Enabled

### Validation Results (Best Checkpoint)
- **Checkpoint**: `best_uar_0.2286.pt`
- **Epoch**: 20+
- **UAR**: **22.86%**
- **Accuracy**: 26.67%
- **Train Loss**: 1.1011
- **Val Loss**: 2.3567

### Training Progress
| Checkpoint | Val UAR |
|------------|---------|
| best_uar_0.1524.pt | 15.24% |
| best_uar_0.1810.pt | 18.10% |
| best_uar_0.2048.pt | 20.48% |
| **best_uar_0.2286.pt** | **22.86%** |

**Key Observations**:
- Very small dataset (360 train samples)
- Low performance suggests severe overfitting or data quality issues
- Learning rate appears to have collapsed (LR: 2.45e-08 at epoch 9)
- Requires investigation and potential retraining with adjusted hyperparameters

**Note**: Test evaluation pending.

---

## IEMOCAP (Interactive Emotional Dyadic Motion Capture)

**Status**: Not available in current workspace. Results to be added when training is completed.

---

## Model Architecture

All models use the **Tier-1 SER architecture**:
- **Feature Extraction**: 40-dimensional MFCC
- **Preprocessing**: Denoising (Demucs) + VAD (Silero)
- **Architecture**:
  - Depthwise separable convolutions (128, 256 channels)
  - Bidirectional LSTM (128 hidden, 2 layers)
  - Multi-head attention (4 heads)
  - Temporal scaling
  - Residual connections
- **Parameters**: ~927K (CREMA/SAVEE), ~3.7M (RAVDESS)
- **Loss**: Cross-entropy with label smoothing (0.1) and class weights

---

## Key Findings

### Best Performing Dataset
**RAVDESS** achieved the highest validation UAR of **72.03%**, likely due to:
- High-quality studio recordings
- Professional actors with clear emotional expressions
- Balanced dataset

### Most Challenging Dataset
**SAVEE** showed the lowest performance (22.86% UAR), possibly due to:
- Very small dataset size (360 training samples)
- Limited speaker diversity (4 male speakers)
- Potential data quality issues
- Learning rate collapse during training

### CREMA-D Performance
- Solid performance with **65.56% validation UAR** and **56.69% test UAR**
- ~9% gap between validation and test suggests some overfitting
- Fear emotion is most challenging (often confused with Happy/Sad)
- Robust error handling successfully managed corrupted audio files

---

## Files and Artifacts

### CREMA-D
- **Best Model**: `checkpoints_crema/best_uar_0.6556.pt`
- **Predictions**: `checkpoints_crema/best_uar_0.6556_predictions.csv`
- **Visualizations**: `checkpoints_crema/visualizations/`
  - `confusion_matrix.png`
  - `per_class_metrics.png`

### RAVDESS
- **Best Model**: `checkpoints_ravdess/best_uar_0.7203.pt`
- **Config**: `config_ravdess.yaml`

### SAVEE
- **Best Model**: `checkpoints_savee/best_uar_0.2286.pt`
- **Config**: `config_savee.yaml`

---

## Next Steps

1. **RAVDESS**: Run test evaluation on best checkpoint
2. **SAVEE**: 
   - Investigate low performance
   - Retrain with adjusted hyperparameters (higher initial LR, different scheduler)
   - Consider data augmentation strategies for small datasets
3. **IEMOCAP**: Complete training and evaluation
4. **Cross-dataset evaluation**: Test models across different datasets to assess generalization

---

*Last Updated: 2025-11-23*
