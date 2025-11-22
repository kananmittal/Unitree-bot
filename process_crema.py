"""
Script to process CREMA-D dataset and create train/val/test CSV files.

CREMA-D filename format: ActorID_Sentence_Emotion_Intensity.wav
Example: 1001_DFA_ANG_XX.wav
- ActorID: 1001 (4 digits)
- Sentence: DFA (3 letters - sentence identifier)
- Emotion: ANG, DIS, FEA, HAP, NEU, SAD
- Intensity: HI (High), MD (Medium), LO (Low), XX (Unspecified)
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# Emotion mapping for CREMA-D
EMOTION_MAP = {
    'ANG': 0,  # Anger
    'DIS': 1,  # Disgust
    'FEA': 2,  # Fear
    'HAP': 3,  # Happy
    'NEU': 4,  # Neutral
    'SAD': 5   # Sad
}

# Reverse mapping for readable labels
EMOTION_LABELS = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad'
}


def parse_crema_filename(filename):
    """
    Parse CREMA-D filename to extract metadata.
    
    Args:
        filename: Audio filename (e.g., '1001_DFA_ANG_XX.wav')
    
    Returns:
        dict with actor_id, sentence, emotion, emotion_code, intensity
    """
    parts = filename.replace('.wav', '').split('_')
    
    if len(parts) != 4:
        raise ValueError(f"Invalid filename format: {filename}")
    
    actor_id, sentence, emotion_code, intensity = parts
    
    if emotion_code not in EMOTION_MAP:
        raise ValueError(f"Unknown emotion code '{emotion_code}' in {filename}")
    
    return {
        'actor_id': actor_id,
        'sentence': sentence,
        'emotion': EMOTION_LABELS[EMOTION_MAP[emotion_code]],
        'emotion_code': EMOTION_MAP[emotion_code],
        'intensity': intensity
    }


def process_crema_dataset(audio_dir, output_dir, test_size=0.15, val_size=0.15, seed=42):
    """
    Process CREMA-D dataset and create train/val/test splits.
    
    Args:
        audio_dir: Directory containing CREMA-D audio files
        output_dir: Directory to save processed CSV files
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        seed: Random seed for reproducibility
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all wav files
    audio_files = sorted(list(audio_dir.glob('*.wav')))
    
    if len(audio_files) == 0:
        raise ValueError(f"No .wav files found in {audio_dir}")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Parse all filenames and create dataset
    data = []
    skipped = 0
    
    for audio_file in audio_files:
        try:
            metadata = parse_crema_filename(audio_file.name)
            data.append({
                'path': str(audio_file),
                'filename': audio_file.name,
                'actor_id': metadata['actor_id'],
                'sentence': metadata['sentence'],
                'emotion': metadata['emotion'],
                'emotion_code': metadata['emotion_code'],
                'intensity': metadata['intensity']
            })
        except Exception as e:
            print(f"Warning: Skipping {audio_file.name}: {e}")
            skipped += 1
    
    if skipped > 0:
        print(f"Skipped {skipped} files due to parsing errors")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Number of actors: {df['actor_id'].nunique()}")
    print(f"Number of sentences: {df['sentence'].nunique()}")
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts().sort_index())
    print(f"\nIntensity distribution:")
    print(df['intensity'].value_counts())
    
    # Split by actor to avoid data leakage
    # This ensures the same actor doesn't appear in both train and test
    unique_actors = df['actor_id'].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_actors)
    
    # Calculate split indices
    n_actors = len(unique_actors)
    n_test = int(n_actors * test_size)
    n_val = int((n_actors - n_test) * val_size)
    
    # Split actors
    test_actors = unique_actors[:n_test]
    val_actors = unique_actors[n_test:n_test + n_val]
    train_actors = unique_actors[n_test + n_val:]
    
    # Create splits
    train_df = df[df['actor_id'].isin(train_actors)].copy()
    val_df = df[df['actor_id'].isin(val_actors)].copy()
    test_df = df[df['actor_id'].isin(test_actors)].copy()
    
    print(f"\nSplit Statistics:")
    print(f"Train: {len(train_df)} samples ({len(train_actors)} actors)")
    print(f"Val:   {len(val_df)} samples ({len(val_actors)} actors)")
    print(f"Test:  {len(test_df)} samples ({len(test_actors)} actors)")
    
    # Print emotion distribution per split
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{split_name} emotion distribution:")
        print(split_df['emotion'].value_counts().sort_index())
    
    # Save to CSV files
    train_csv = output_dir / 'crema_train.csv'
    val_csv = output_dir / 'crema_val.csv'
    test_csv = output_dir / 'crema_test.csv'
    all_csv = output_dir / 'crema_all.csv'
    
    # Keep only necessary columns for training
    cols_to_save = ['path', 'emotion', 'emotion_code']
    
    train_df[cols_to_save].to_csv(train_csv, index=False)
    val_df[cols_to_save].to_csv(val_csv, index=False)
    test_df[cols_to_save].to_csv(test_csv, index=False)
    df.to_csv(all_csv, index=False)  # Save full dataset with all metadata
    
    print(f"\nSaved CSV files:")
    print(f"  Train: {train_csv}")
    print(f"  Val:   {val_csv}")
    print(f"  Test:  {test_csv}")
    print(f"  All:   {all_csv}")
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    # Auto-detect if running from root or ser/ directory
    script_dir = Path(__file__).parent
    
    # Try to find the datasets directory - check multiple possible locations
    possible_paths = [
        # If audio files are directly in crema folder (no AudioWAV subfolder)
        script_dir / 'datasets' / 'crema',  # Running from root
        script_dir.parent / 'datasets' / 'crema',  # Running from ser/
        # If audio files are in AudioWAV subfolder
        script_dir / 'datasets' / 'crema' / 'AudioWAV',
        script_dir.parent / 'datasets' / 'crema' / 'AudioWAV',
    ]
    
    AUDIO_DIR = None
    for path in possible_paths:
        if path.exists() and list(path.glob('*.wav')):
            AUDIO_DIR = path
            break
    
    # Set output directory
    if script_dir.name == 'ser':
        OUTPUT_DIR = script_dir.parent / 'datasets' / 'processed'
    else:
        OUTPUT_DIR = script_dir / 'datasets' / 'processed'
    
    if AUDIO_DIR is None:
        print("❌ Could not find CREMA-D audio files!")
        print("\nSearched in:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease ensure .wav files are in one of these locations.")
        exit(1)
    
    print(f"✓ Found audio files in: {AUDIO_DIR}")
    print(f"✓ Will save CSV files to: {OUTPUT_DIR}\n")
    
    # Process the dataset
    try:
        train_df, val_df, test_df = process_crema_dataset(
            audio_dir=AUDIO_DIR,
            output_dir=OUTPUT_DIR,
            test_size=0.15,
            val_size=0.15,
            seed=42
        )
        print("\nProcessing complete!")
        print("\nNext steps:")
        print("1. Verify the CSV files in datasets/processed/")
        print("2. Run training with: python ser/train.py --config config_crema.yaml")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure CREMA-D audio files are in: datasets/crema/AudioWAV/")
        print("2. Check that the directory contains .wav files")
        print("3. Verify the directory structure:")
        print("   Unitree/")
        print("   └── datasets/")
        print("       └── crema/")
        print("           └── AudioWAV/")
        print("               ├── 1001_DFA_ANG_XX.wav")
        print("               └── ...")
        import traceback
        traceback.print_exc()