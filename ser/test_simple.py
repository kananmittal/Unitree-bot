"""Simple test script without config dependencies"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import AudioEmotionDataset, pad_collate, get_emotion_map_from_dataset
from tier1_model import SER_Tier1
from metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache_dir", default="feat_cache")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if "model" in ckpt:
        state_dict = ckpt["model"]
        checkpoint_args = ckpt.get("args", {})
    else:
        state_dict = ckpt
        checkpoint_args = {}
    
    # Get n_mfcc first
    n_mfcc = checkpoint_args.get("n_mfcc", 40)
    
    # Detect num_classes from the FINAL output layer
    num_classes = checkpoint_args.get("num_classes", None)
    
    # Check different possible layer names for the final classification layer
    if num_classes is None:
        for key in state_dict.keys():
            if 'cls_head' in key and 'weight' in key and key.endswith('.weight'):
                shape = state_dict[key].shape
                if len(shape) == 2 and shape[0] < 20:  # Output layer should have small first dim (num_classes)
                    num_classes = shape[0]
                    print(f"Detected num_classes={num_classes} from {key} with shape {shape}")
                    break
        
        # If still not found, check head layers
        if num_classes is None:
            if "head.1.weight" in state_dict:
                num_classes = state_dict["head.1.weight"].shape[0]
                print(f"Detected num_classes={num_classes} from head.1.weight")
    
    # Fallback: get from dataset
    if num_classes is None:
        emotion_map_test = get_emotion_map_from_dataset(args.test_csv)
        num_classes = len(emotion_map_test) if emotion_map_test else 8
        print(f"Using num_classes={num_classes} from dataset")
    
    print(f"Final: num_classes={num_classes}, n_mfcc={n_mfcc}")
    
    # Create model
    model = SER_Tier1(n_mfcc=n_mfcc, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Get emotion mapping
    emotion_map = get_emotion_map_from_dataset(args.test_csv)
    if emotion_map:
        label_to_emotion = {v: k for k, v in emotion_map.items()}
        class_names = [label_to_emotion.get(i, f"Class{i}") for i in range(num_classes)]
    else:
        class_names = [f"Class{i}" for i in range(num_classes)]
    
    print(f"Classes: {class_names}")
    
    # Load dataset
    test_ds = AudioEmotionDataset(
        args.test_csv,
        cache_dir=args.cache_dir,
        device=device,
        n_mfcc=n_mfcc,
        emotion_map=emotion_map
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate
    )
    
    print(f"Test samples: {len(test_ds)}")
    print("\nEvaluating...")
    
    # Evaluate
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch_idx, (mfcc, labels) in enumerate(test_loader):
            mfcc = mfcc.to(device)
            labels = labels.to(device)
            
            out = model(mfcc)
            logits = out["logits"]
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(logits, labels, num_classes)
    
    # Build confusion matrix
    preds = logits.argmax(dim=1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    
    # Print results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Accuracy:  {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
    print(f"UAR:       {metrics['uar']:.4f} ({metrics['uar']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print(" "*12, end="")
    for name in class_names:
        print(f"{name[:8]:>9}", end="")
    print()
    print("-"*70)
    
    for i, name in enumerate(class_names):
        print(f"{name[:11]:>12}", end="")
        for j in range(num_classes):
            print(f"{cm[i,j]:>9}", end="")
        print()
    
    print("\n" + "="*70)
    print("PER-CLASS METRICS")
    print("="*70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*70)
    
    for i in range(num_classes):
        prec = metrics['precision_per_class'][i]
        rec = metrics['recall_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        support = cm[i, :].sum().item()
        print(f"{class_names[i]:<15} {prec:>11.4f} {rec:>11.4f} {f1:>11.4f} {support:>9}")
    
    print("="*70)
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   Total samples: {len(labels)}")
    print(f"   Correct predictions: {(preds == labels).sum().item()}")
    print(f"   Overall Accuracy: {metrics['acc']*100:.2f}%")
    print(f"   UAR (main metric): {metrics['uar']*100:.2f}%")

if __name__ == "__main__":
    main()