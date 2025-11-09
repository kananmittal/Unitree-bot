"""
Script to run Tier 1 Model Demo (single file inference)
"""

import torch
import sys
import os
from tier1_model import SER_Tier1
from tier0_io import tier0_to_mfcc

# Emotion class names for 4-class setup
CLASS_NAMES = ["Happy", "Neutral", "Sad", "Angry"]

def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint file"""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model" in ckpt:
        state_dict = ckpt["model"]
        args = ckpt.get("args", {})
    else:
        state_dict = ckpt
        args = {}
    
    # Infer model parameters
    num_classes = args.get("num_classes", 4)
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
    # Also check attn.mlp.2.weight as fallback (shape: (n_mfcc, hidden))
    if "attn.mlp.2.weight" in state_dict:
        # attn.mlp.2.weight shape is (n_mfcc, hidden)
        n_mfcc = state_dict["attn.mlp.2.weight"].shape[0]
    
    model = SER_Tier1(n_mfcc=n_mfcc, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes

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
        model, num_classes = load_model_from_checkpoint(checkpoint_path, device)
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

