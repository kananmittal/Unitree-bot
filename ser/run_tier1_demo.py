"""
Script to run Tier 1 Model Demo (single file inference)
"""

import torch
import sys
import os
from tier1_model import SER_Tier1, load_model_from_checkpoint
from tier0_io import tier0_to_mfcc

# Emotion class names for 4-class setup
CLASS_NAMES = ["Happy", "Neutral", "Sad", "Angry"]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_tier1_demo.py <wav_path> [checkpoint_path]")
        print("\nExample:")
        print("  python run_tier1_demo.py audio.wav")
        print("  python run_tier1_demo.py audio.wav checkpoints/best_uar_0.8500.pt")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load MFCC features (Tier 0)
    print(f"\nProcessing audio: {wav_path}")
    mfcc = tier0_to_mfcc(wav_path, device=device)
    print(f"MFCC shape: {tuple(mfcc.shape)}")
    
    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model, num_classes, _ = load_model_from_checkpoint(checkpoint_path, device)
        class_names = CLASS_NAMES if num_classes == 4 else [f"Class {i}" for i in range(num_classes)]
    else:
        if checkpoint_path:
            print(f"Warning: Checkpoint not found: {checkpoint_path}, using untrained model")
        print("Using untrained model (random weights)")
        num_classes = 4
        model = SER_Tier1(n_mfcc=40, num_classes=num_classes).to(device)
        class_names = CLASS_NAMES
    
    model.eval()
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        mfcc_batch = mfcc.unsqueeze(0).to(device)  # (1, C, T)
        out = model(mfcc_batch)
        logits = out["logits"]
        probs = out["probs"]
        pred_class = torch.argmax(logits, dim=1).item()
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted class: {pred_class} ({class_names[pred_class]})")
    print(f"Confidence: {probs[0][pred_class]:.4f}")
    print("\nClass probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, probs[0])):
        marker = " <--" if i == pred_class else ""
        print(f"  {name:12s}: {prob:.4f}{marker}")
    print("=" * 60)

