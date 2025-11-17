"""
Improved test script with:
- Better checkpoint loading
- Emotion mapping from checkpoint
- Visualization support
- Detailed error analysis
"""
import os
import argparse
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from pathlib import Path

from datasets import create_dataset, pad_collate
from tier1_model import load_model_from_checkpoint
from config import Config
from metrics import compute_metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Test SER model")
    ap.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    ap.add_argument("--cache_dir", type=str, default="feat_cache", help="Cache directory")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    ap.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    ap.add_argument("--save_predictions", action="store_true", help="Save predictions to CSV")
    ap.add_argument("--visualize", action="store_true", help="Generate visualizations")
    return ap.parse_args()


def load_checkpoint_with_config(checkpoint_path: str, device: torch.device):
    """Load model and configuration from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract components
    if "model" in ckpt:
        state_dict = ckpt["model"]
        config_dict = ckpt.get("config", {})
        emotion_map = ckpt.get("emotion_map", None)
    else:
        state_dict = ckpt
        config_dict = {}
        emotion_map = None
    
    # Try to load emotion map from checkpoint directory
    if emotion_map is None:
        emotion_map_path = os.path.join(os.path.dirname(checkpoint_path), "emotion_map.json")
        if os.path.exists(emotion_map_path):
            with open(emotion_map_path, 'r') as f:
                emotion_map = json.load(f)
            print(f"Loaded emotion map from: {emotion_map_path}")
    
    # Load model using the utility function
    model, num_classes, model_config = load_model_from_checkpoint(checkpoint_path, device)
    
    return model, num_classes, config_dict, emotion_map


def get_class_names(emotion_map, num_classes):
    """Get class names from emotion map"""
    if emotion_map is None:
        return [f"Class {i}" for i in range(num_classes)]
    
    # Create reverse mapping: label -> emotion
    label_to_emotion = {v: k for k, v in emotion_map.items()}
    return [label_to_emotion.get(i, f"Class {i}") for i in range(num_classes)]


def print_confusion_matrix(cm, class_names):
    """Print confusion matrix with class names"""
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print(" " * 12, end="")
    for name in class_names:
        print(f"{name:>12}", end="")
    print()
    print("-" * (12 + 12 * len(class_names)))
    
    for i, name in enumerate(class_names):
        print(f"{name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12}", end="")
        print()
    print("=" * 80)


def print_per_class_metrics(cm, class_names):
    """Print detailed per-class metrics"""
    print("\n" + "=" * 80)
    print("PER-CLASS METRICS")
    print("=" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    total_correct = 0
    total_samples = 0
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{name:<15} {precision:>11.4f} {recall:>11.4f} {f1:>11.4f} {support:>9}")
        
        total_correct += tp
        total_samples += support
    
    print("-" * 80)
    print(f"{'Overall':<15} {' '*11} {' '*11} {' '*11} {total_samples:>9}")
    print(f"{'Accuracy':<15} {total_correct/total_samples:>11.4f}")
    print("=" * 80)


def analyze_errors(logits, labels, class_names, topk=5):
    """Analyze most confident errors"""
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=1)
    confidence, _ = probs.max(dim=-1)
    
    # Find errors
    errors = preds != labels
    error_indices = torch.where(errors)[0]
    
    if len(error_indices) == 0:
        print("\nNo errors found!")
        return
    
    # Get most confident errors
    error_confidences = confidence[error_indices]
    top_error_indices = error_indices[error_confidences.argsort(descending=True)[:topk]]
    
    print("\n" + "=" * 80)
    print(f"TOP {topk} MOST CONFIDENT ERRORS")
    print("=" * 80)
    
    for idx in top_error_indices:
        true_label = labels[idx].item()
        pred_label = preds[idx].item()
        conf = confidence[idx].item()
        
        print(f"\nSample {idx}:")
        print(f"  True label: {class_names[true_label]}")
        print(f"  Predicted:  {class_names[pred_label]} (confidence: {conf:.4f})")
        print(f"  Probabilities: {probs[idx].tolist()}")


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Evaluate model on test set"""
    model.eval()
    all_logits, all_labels = [], []
    
    print("\nEvaluating...")
    for batch_idx, (mfcc, labels) in enumerate(loader):
        mfcc = mfcc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        out = model(mfcc)
        logits = out["logits"]
        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(loader)} batches")
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(logits, labels, num_classes)
    
    # Build confusion matrix
    preds = logits.argmax(dim=1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    cm = cm.numpy()
    
    return metrics, cm, logits, labels, preds


def save_predictions(preds, labels, class_names, output_path):
    """Save predictions to CSV"""
    import pandas as pd
    
    df = pd.DataFrame({
        "true_label": labels.numpy(),
        "true_class": [class_names[l] for l in labels.numpy()],
        "pred_label": preds.numpy(),
        "pred_class": [class_names[p] for p in preds.numpy()],
        "correct": (labels == preds).numpy()
    })
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")


def visualize_results(cm, class_names, metrics, output_dir):
    """Generate visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("\nWarning: matplotlib/seaborn not installed. Skipping visualization.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Per-class metrics bar plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric_name, metric_data in zip(
        axes,
        ['Precision', 'Recall', 'F1-Score'],
        [metrics['precision_per_class'], metrics['recall_per_class'], metrics['f1_per_class']]
    ):
        ax.bar(range(len(class_names)), metric_data)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Per-Class {metric_name}')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300)
    plt.close()
    
    print(f"\nSaved visualizations to: {output_dir}")


def main():
    args = parse_args()
    
    import platform
    if platform.system() == "Darwin" and args.num_workers > 0:
        args.num_workers = 0
        print("Note: Using num_workers=0 on macOS for compatibility")
    
    # Get device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load model and config
    model, num_classes, config_dict, emotion_map = load_checkpoint_with_config(args.checkpoint, device)
    
    # Get class names
    class_names = get_class_names(emotion_map, num_classes)
    print(f"Classes: {class_names}")
    
    # Create configuration for dataset loading
    # NEW - Handle both dict and Config object
if config_dict:
    if isinstance(config_dict, dict):
        # Create a simple namespace to hold config
        class SimpleConfig:
            def __init__(self):
                self.data = type('obj', (object,), {
                    'cache_dir': args.cache_dir,
                    'num_workers': args.num_workers
                })()
        config = SimpleConfig()
    else:
        config = config_dict
        config.data.cache_dir = args.cache_dir
        config.data.num_workers = args.num_workers
else:
    class SimpleConfig:
        def __init__(self):
            self.data = type('obj', (object,), {
                'cache_dir': args.cache_dir,
                'num_workers': args.num_workers
            })()
    config = SimpleConfig()
    
    # Load test dataset (no augmentation)
    print(f"\nLoading test dataset: {args.test_csv}")
    test_ds = create_dataset(
        args.test_csv,
        config,
        audio_augmentation=None,
        mfcc_augmentation=None,
        emotion_map=emotion_map
    )
    
    # Create data loader
    pin_mem = device.type == "cuda"
    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=pin_mem
    )
    
    print(f"Test samples: {len(test_ds)}")
    print(f"Test batches: {len(test_ld)}")
    
    # Evaluate
    metrics, cm, logits, labels, preds = evaluate(model, test_ld, device, num_classes)
    
    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"UAR (Unweighted Average Recall): {metrics['uar']:.4f}")
    print(f"Macro Precision: {metrics['precision']:.4f}")
    print(f"Macro F1-Score: {metrics['f1']:.4f}")
    
    print_confusion_matrix(cm, class_names)
    print_per_class_metrics(cm, class_names)
    
    # Analyze errors
    analyze_errors(logits, labels, class_names)
    
    # Save predictions
    if args.save_predictions:
        pred_path = args.checkpoint.replace(".pt", "_predictions.csv")
        save_predictions(preds, labels, class_names, pred_path)
    
    # Generate visualizations
    if args.visualize:
        viz_dir = os.path.join(os.path.dirname(args.checkpoint), "visualizations")
        visualize_results(cm, class_names, metrics, viz_dir)


if __name__ == "__main__":
    main()