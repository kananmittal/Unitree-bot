"""
Test a single audio file with a trained model
Usage: python ser/test_single.py <audio_file> <checkpoint_path> [--class_names]
"""

import sys
import torch
import os
from tier1_model import SER_Tier1, load_model_from_checkpoint
from tier0_io import tier0_to_mfcc

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_single.py <audio_file> <checkpoint_path>")
        print("Example: python test_single.py ../datasets/ravdess/Actor_01/03-01-01-01-01-01-01.wav checkpoints/best_uar_0.7500.pt")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    checkpoint_path = sys.argv[2]
    
    # Default class names
    class_names = ["Happy", "Neutral", "Sad", "Angry"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load model using the utility function
    model, num_classes, _ = load_model_from_checkpoint(checkpoint_path, device)
    print(f"Model loaded: {num_classes} classes")
    
    # Process audio (default n_mfcc=40)
    print(f"\nProcessing audio: {audio_file}")
    mfcc = tier0_to_mfcc(audio_file, device=device, n_mfcc=40)
    print(f"MFCC shape: {tuple(mfcc.shape)}")
    
    # Predict
    with torch.no_grad():
        mfcc_batch = mfcc.unsqueeze(0).to(device)  # (1, C, T)
        out = model(mfcc_batch)
        logits = out["logits"]
        probs = out["probs"]
        pred_class = torch.argmax(logits, dim=1).item()
    
    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted class: {pred_class} ({class_names[pred_class] if pred_class < len(class_names) else f'Class_{pred_class}'})")
    print(f"Confidence: {probs[0][pred_class]:.4f}")
    print("\nClass Probabilities:")
    for i, prob in enumerate(probs[0]):
        class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
        marker = " <--" if i == pred_class else ""
        print(f"  {class_name}: {prob:.4f}{marker}")
    print("=" * 50)

if __name__ == "__main__":
    main()

