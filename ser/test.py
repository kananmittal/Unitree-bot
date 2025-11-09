"""
Test/Evaluation script for SER model
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import AudioEmotionDataset, pad_collate
from tier1_model import SER_Tier1
from metrics import compute_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    ap.add_argument("--cache_dir", type=str, default="feat_cache", help="Directory for cached features")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    ap.add_argument("--device", type=str, default=None, help="Device (cuda/cpu), auto-detect if None")
    return ap.parse_args()

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model" in ckpt:
        state_dict = ckpt["model"]
        args = ckpt.get("args", {})
    else:
        state_dict = ckpt
        args = {}
    
    # Infer model parameters from state dict or args
    num_classes = args.get("num_classes", None)  # Will auto-detect if None
    n_mfcc = args.get("n_mfcc", 40)
    
    # Try to infer from state_dict if not in args
    # New model structure: cls_head.2.weight (LayerNorm, Dropout, Linear)
    if "cls_head.2.weight" in state_dict:
        num_classes = state_dict["cls_head.2.weight"].shape[0]
    # Old model structure: head.1.weight (LayerNorm, Linear)
    elif "head.1.weight" in state_dict:
        num_classes = state_dict["head.1.weight"].shape[0]
    
    # New model structure: cnn.0.net.0.weight (first conv in first Conv1DBlock)
    if "cnn.0.net.0.weight" in state_dict:
        # For depthwise-separable: groups=in_ch, so weight shape is (in_ch, 1, k)
        # For regular conv: weight shape is (out_ch, in_ch, k)
        weight = state_dict["cnn.0.net.0.weight"]
        if weight.shape[1] == 1:  # depthwise-separable: (in_ch, 1, k)
            n_mfcc = weight.shape[0]  # in_ch is n_mfcc
        else:  # regular conv: (out_ch, in_ch, k)
            n_mfcc = weight.shape[1]  # in_ch is n_mfcc
    # Old model structure: conv.weight
    elif "conv.weight" in state_dict:
        n_mfcc = state_dict["conv.weight"].shape[1]
    # Also check attn.mlp.0.weight as fallback (shape: (hidden, n_mfcc))
    elif "attn.mlp.0.weight" in state_dict:
        # attn.mlp.0.weight shape is (hidden, n_mfcc) where hidden = n_mfcc // reduction
        # We can't directly infer n_mfcc from this, but we can check attn.mlp.2.weight
        pass
    if "attn.mlp.2.weight" in state_dict:
        # attn.mlp.2.weight shape is (n_mfcc, hidden)
        n_mfcc = state_dict["attn.mlp.2.weight"].shape[0]
    
    # Auto-detect num_classes if not provided
    if num_classes is None:
        # Try to infer from cls_head weight shape
        if "cls_head.2.weight" in state_dict:
            num_classes = state_dict["cls_head.2.weight"].shape[0]
        elif "head.1.weight" in state_dict:
            num_classes = state_dict["head.1.weight"].shape[0]
        else:
            # Fallback: use default or raise error
            num_classes = 4
            print("Warning: Could not infer num_classes, using default 4")
    
    print(f"Model parameters: n_mfcc={n_mfcc}, num_classes={num_classes}")
    
    model = SER_Tier1(n_mfcc=n_mfcc, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes

def print_confusion_matrix(cm, class_names=None):
    """Print confusion matrix"""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    print("\nConfusion Matrix:")
    print(" " * 12, end="")
    for name in class_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>8}", end="")
        print()

def print_per_class_metrics(cm, class_names=None):
    """Print per-class metrics"""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 48)
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{name:<12} {precision:>11.4f} {recall:>11.4f} {f1:>11.4f}")

@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_names=None):
    """Evaluate model on test set"""
    model.eval()
    all_logits, all_labels = [], []
    
    print("Evaluating...")
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

def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Adjust num_workers for macOS
    import platform
    if platform.system() == "Darwin" and args.num_workers > 0:
        args.num_workers = 0
        print("Note: Using num_workers=0 on macOS for compatibility")
    
    print(f"Using device: {device}")
    
    # Load model
    model, num_classes = load_model(args.checkpoint, device)
    
    # Load test dataset
    print(f"Loading test dataset: {args.test_csv}")
    test_ds = AudioEmotionDataset(args.test_csv, cache_dir=args.cache_dir, device=device)
    
    # Create data loader
    pin_mem = device.type == "cuda"
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=pin_mem)
    
    print(f"Test samples: {len(test_ds)}")
    print(f"Test batches: {len(test_ld)}")
    
    # Auto-detect class names from dataset if possible
    try:
        from datasets import get_emotion_map_from_dataset
        emotion_map = get_emotion_map_from_dataset(args.test_csv)
        if emotion_map:
            # Create reverse mapping: label -> emotion
            label_to_emotion = {v: k for k, v in emotion_map.items()}
            class_names = [label_to_emotion.get(i, f"Class {i}") for i in range(num_classes)]
        else:
            class_names = [f"Class {i}" for i in range(num_classes)]
    except:
        # Fallback: use generic class names
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Evaluate
    metrics, cm, logits, labels, preds = evaluate(model, test_ld, device, num_classes, class_names)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"UAR (Unweighted Average Recall): {metrics['uar']:.4f}")
    
    print_confusion_matrix(cm, class_names)
    print_per_class_metrics(cm, class_names)
    
    print("\n" + "=" * 60)
    
    # Save predictions (optional)
    save_preds = os.environ.get("SAVE_PREDS", "false").lower() == "true"
    if save_preds:
        try:
            import pandas as pd
            pred_df = pd.DataFrame({
                "true_label": labels.numpy(),
                "pred_label": preds.numpy(),
                "correct": (labels == preds).numpy()
            })
            pred_path = args.checkpoint.replace(".pt", "_predictions.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"Predictions saved to: {pred_path}")
        except ImportError:
            print("Warning: pandas not installed, skipping prediction save. Install with: pip install pandas")

if __name__ == "__main__":
    main()
