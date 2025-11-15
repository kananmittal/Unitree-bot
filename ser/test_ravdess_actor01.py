"""
Test model on RAVDESS Actor_01 samples
Usage: python ser/test_ravdess_actor01.py [checkpoint_path]
"""

import sys
import os
import glob
import torch
from pathlib import Path
from tier1_model import SER_Tier1
from tier0_io import tier0_to_mfcc

# RAVDESS emotion codes
RAVDESS_EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

# Map RAVDESS emotions to 4-class system
# Class 0 (Happy): happy, surprised, excited
# Class 1 (Neutral): neutral, calm
# Class 2 (Sad): sad, fearful, fear
# Class 3 (Angry): angry, disgust
RAVDESS_TO_CLASS_4 = {
    1: 1,  # neutral -> Neutral
    2: 1,  # calm -> Neutral
    3: 0,  # happy -> Happy
    4: 2,  # sad -> Sad
    5: 3,  # angry -> Angry
    6: 2,  # fearful -> Sad
    7: 3,  # disgust -> Angry
    8: 0,  # surprised -> Happy
}

# Map RAVDESS emotions to 7-class IEMOCAP system
# IEMOCAP 7-class mapping (from training data):
# Class 0: angry
# Class 1: disgusted
# Class 2: fearful
# Class 3: happy
# Class 4: neutral
# Class 5: sad
# Class 6: surprised
RAVDESS_TO_CLASS_7 = {
    1: 4,  # neutral -> neutral (4)
    2: 4,  # calm -> neutral (4) [closest match]
    3: 3,  # happy -> happy (3)
    4: 5,  # sad -> sad (5)
    5: 0,  # angry -> angry (0)
    6: 2,  # fearful -> fearful (2)
    7: 1,  # disgust -> disgusted (1)
    8: 6,  # surprised -> surprised (6)
}

CLASS_NAMES_4 = ["Happy", "Neutral", "Sad", "Angry"]
CLASS_NAMES_7 = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename: MM-CC-EE-II-SS-RR-AA.wav
    Returns: (emotion_code, actor_id)
    """
    basename = os.path.basename(filename)
    parts = basename.replace('.wav', '').split('-')
    if len(parts) >= 7:
        emotion_code = int(parts[2])  # EE
        actor_id = int(parts[6])     # AA
        return emotion_code, actor_id
    return None, None

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
    num_classes = args.get("num_classes", None)
    n_mfcc = args.get("n_mfcc", 40)
    
    # Try to infer from state_dict if not in args
    if "cls_head.2.weight" in state_dict:
        num_classes = state_dict["cls_head.2.weight"].shape[0]
    elif "head.1.weight" in state_dict:
        num_classes = state_dict["head.1.weight"].shape[0]
    
    if "cnn.0.net.0.weight" in state_dict:
        weight = state_dict["cnn.0.net.0.weight"]
        if weight.shape[1] == 1:  # depthwise-separable
            n_mfcc = weight.shape[0]
        else:
            n_mfcc = weight.shape[1]
    elif "attn.mlp.2.weight" in state_dict:
        n_mfcc = state_dict["attn.mlp.2.weight"].shape[0]
    
    if num_classes is None:
        num_classes = 4
        print("Warning: Could not infer num_classes, using default 4")
    
    print(f"Model parameters: n_mfcc={n_mfcc}, num_classes={num_classes}")
    
    model = SER_Tier1(n_mfcc=n_mfcc, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes

def test_actor01(checkpoint_path=None, actor_dir=None, device=None):
    """Test model on all Actor_01 files"""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        checkpoints = sorted(glob.glob(str(checkpoint_dir / "best_uar_*.pt")))
        if not checkpoints:
            print("Error: No checkpoint found. Please specify checkpoint path.")
            return
        checkpoint_path = checkpoints[-1]  # Use best checkpoint
        print(f"Using checkpoint: {checkpoint_path}")
    
    # Load model
    model, num_classes = load_model_from_checkpoint(checkpoint_path, device)
    
    # Select class names and mapping based on model
    if num_classes == 4:
        class_names = CLASS_NAMES_4
        ravdess_to_class = RAVDESS_TO_CLASS_4
    elif num_classes == 7:
        # 7-class IEMOCAP model
        class_names = CLASS_NAMES_7
        ravdess_to_class = RAVDESS_TO_CLASS_7
        print(f"Using 7-class IEMOCAP mapping: {class_names}")
    else:
        class_names = [f"Class_{i}" for i in range(num_classes)]
        ravdess_to_class = RAVDESS_TO_CLASS_4  # Default fallback
        print(f"Warning: Model has {num_classes} classes. Using generic class names and 4-class mapping.")
    
    # Find Actor_01 directory
    if actor_dir is None:
        project_root = Path(__file__).parent.parent
        actor_dir = project_root / "datasets" / "ravdess" / "Actor_01"
    
    if not os.path.exists(actor_dir):
        print(f"Error: Actor_01 directory not found: {actor_dir}")
        return
    
    # Get all wav files
    wav_files = sorted(glob.glob(str(actor_dir / "*.wav")))
    if not wav_files:
        print(f"Error: No WAV files found in {actor_dir}")
        return
    
    print(f"\nFound {len(wav_files)} audio files in Actor_01")
    print("=" * 80)
    
    # Test each file
    results = []
    correct = 0
    total = 0
    
    for wav_file in wav_files:
        # Parse filename
        emotion_code, actor_id = parse_ravdess_filename(wav_file)
        if emotion_code is None:
            print(f"Warning: Could not parse filename: {wav_file}")
            continue
        
        # Get ground truth label using the appropriate mapping
        true_class = ravdess_to_class.get(emotion_code, -1)
        if true_class == -1:
            print(f"Warning: Unknown emotion code {emotion_code} in {wav_file}")
            continue
        
        # Check if true_class is valid for the model
        if true_class >= num_classes:
            print(f"Warning: true_class {true_class} out of range for {num_classes}-class model in {wav_file}")
            continue
        
        emotion_name = RAVDESS_EMOTIONS.get(emotion_code, "unknown")
        
        # Process audio
        try:
            mfcc = tier0_to_mfcc(wav_file, device=device)
            
            # Check if MFCC is valid (has enough time frames)
            if mfcc.shape[1] < 1:  # T dimension too small
                print(f"✗ Skipping {os.path.basename(wav_file)}: MFCC has insufficient time frames ({mfcc.shape})")
                continue
            
            mfcc_batch = mfcc.unsqueeze(0).to(device)  # (1, C, T)
            
            # Predict
            with torch.no_grad():
                out = model(mfcc_batch)
                logits = out["logits"]
                probs = out["probs"]
                pred_class = torch.argmax(logits, dim=1).item()
                
                # Handle case where pred_class might be out of range
                if pred_class >= len(probs[0]):
                    print(f"✗ Error: pred_class {pred_class} out of range for {os.path.basename(wav_file)}")
                    continue
                
                confidence = probs[0][pred_class].item()
            
            # Check if correct
            is_correct = (pred_class == true_class)
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            results.append({
                'file': os.path.basename(wav_file),
                'emotion_code': emotion_code,
                'emotion_name': emotion_name,
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Print result
            status = "✓" if is_correct else "✗"
            true_name = class_names[true_class] if true_class < len(class_names) else f"Class_{true_class}"
            pred_name = class_names[pred_class] if pred_class < len(class_names) else f"Class_{pred_class}"
            print(f"{status} {os.path.basename(wav_file):30} | "
                  f"True: {true_name:8} ({true_class}) | "
                  f"Pred: {pred_name:8} ({pred_class}) | "
                  f"Conf: {confidence:.3f} | "
                  f"Emotion: {emotion_name}")
        
        except Exception as e:
            print(f"✗ Error processing {wav_file}: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model trained on: IEMOCAP ({num_classes} classes)")
    print(f"Test dataset: RAVDESS Actor_01")
    print(f"Note: This is a CROSS-DATASET evaluation (trained on IEMOCAP, tested on RAVDESS)")
    print(f"      Lower accuracy is expected due to domain shift between datasets.")
    print("=" * 80)
    print(f"Total files tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {correct/total*100:.2f}%" if total > 0 else "N/A")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for class_idx in range(num_classes):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
        class_total = sum(1 for r in results if r['true_class'] == class_idx)
        class_correct = sum(1 for r in results if r['true_class'] == class_idx and r['correct'])
        if class_total > 0:
            print(f"  {class_name:12}: {class_correct}/{class_total} = {class_correct/class_total*100:.2f}%")
    
    # Per-emotion accuracy
    print("\nPer-Emotion Accuracy:")
    for emotion_code in sorted(RAVDESS_EMOTIONS.keys()):
        emotion_name = RAVDESS_EMOTIONS[emotion_code]
        emotion_total = sum(1 for r in results if r['emotion_code'] == emotion_code)
        emotion_correct = sum(1 for r in results if r['emotion_code'] == emotion_code and r['correct'])
        if emotion_total > 0:
            print(f"  {emotion_name:12}: {emotion_correct}/{emotion_total} = {emotion_correct/emotion_total*100:.2f}%")
    
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_actor01(checkpoint_path)

