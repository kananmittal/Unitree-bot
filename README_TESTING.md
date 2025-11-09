# SER Testing Guide

## Overview
After training, you can test your model in two ways:
1. **Full Test Set Evaluation**: Evaluate on a test CSV file with detailed metrics
2. **Single File Inference**: Test on individual audio files

## Quick Start

### 1. Test on Full Test Set

#### Using Shell Script:
```bash
./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv
```

#### Using Python Directly:
```bash
python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt \
  --batch_size 16 \
  --cache_dir feat_cache
```

### 2. Test on Single Audio File

```bash
# With trained model
python ser/run_tier1_demo.py audio.wav checkpoints/best_uar_0.8500.pt

# Without model (uses random weights for testing)
python ser/run_tier1_demo.py audio.wav
```

## Test Output

### Full Test Set Evaluation
The test script provides:
- **Overall Metrics**: Accuracy, UAR (Unweighted Average Recall)
- **Confusion Matrix**: Shows predictions vs true labels
- **Per-Class Metrics**: Precision, Recall, F1-Score for each emotion class

Example output:
```
============================================================
TEST RESULTS
============================================================
Accuracy: 0.8234
UAR (Unweighted Average Recall): 0.8156

Confusion Matrix:
            Happy Neutral     Sad   Angry
     Happy     245      12       8       5
   Neutral      15     230      10       8
        Sad     10       8     220      12
      Angry      8       5      15     225

Per-Class Metrics:
Class        Precision      Recall    F1-Score
------------------------------------------------
Happy           0.8824     0.9224     0.9020
Neutral         0.9016     0.8760     0.8886
Sad             0.8704     0.8800     0.8752
Angry           0.9000     0.8889     0.8944
============================================================
```

### Single File Inference
Provides:
- Predicted emotion class
- Confidence score
- Probability distribution over all classes

Example output:
```
============================================================
PREDICTION RESULTS
============================================================
Predicted class: 0 (Happy)
Confidence: 0.8543

Class probabilities:
  Happy       : 0.8543 <--
  Neutral     : 0.0821
  Sad         : 0.0432
  Angry       : 0.0204
============================================================
```

## Finding Checkpoints

After training, checkpoints are saved in the `checkpoints/` directory (or your specified `--out_dir`):
- `best_uar_<score>.pt`: Best model by UAR metric
- `last.pt`: Final model after all epochs

List available checkpoints:
```bash
ls -lh checkpoints/
```

## Advanced Options

### Save Predictions to CSV
```bash
SAVE_PREDS=true python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt
```

This creates a file `checkpoints/best_uar_0.8500_predictions.csv` with:
- `true_label`: Ground truth label
- `pred_label`: Predicted label
- `correct`: Whether prediction is correct

### Custom Batch Size
```bash
python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt \
  --batch_size 32
```

### Use Specific Device
```bash
python ser/test.py \
  --test_csv datasets/processed/iemocap_test.csv \
  --checkpoint checkpoints/best_uar_0.8500.pt \
  --device cuda
```

## Emotion Classes

For 4-class setup:
- **Class 0**: Happy
- **Class 1**: Neutral
- **Class 2**: Sad
- **Class 3**: Angry

## Metrics Explained

- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **UAR (Unweighted Average Recall)**: Average recall across all classes (macro recall)
- **Precision**: Of all predictions for a class, how many were correct
- **Recall**: Of all true instances of a class, how many were found
- **F1-Score**: Harmonic mean of precision and recall

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
python ser/test.py --test_csv test.csv --checkpoint model.pt --batch_size 8
```

## Examples

### Test with Default Settings
```bash
./test.sh checkpoints/best_uar_0.8500.pt datasets/processed/iemocap_test.csv
```

### Test Single Audio File
```bash
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

