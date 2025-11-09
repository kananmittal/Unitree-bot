import sys, torch
from tier0_io import tier0_to_mfcc

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_tier0.py <wav_path>")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Processing audio: {wav_path}")
    print(f"Using device: {device}")
    feats = tier0_to_mfcc(wav_path, device=device)
    print("MFCC shape:", tuple(feats.shape))  # (C, T)
