#!/usr/bin/env python3
"""
Process Savee dataset and create train/val/test CSV files with emotion mapping.

Savee dataset structure:
- Files are named: {SPEAKER}_{EMOTION}{NUMBER}.wav
- Speakers: DC, JE, JK, KL (4 speakers)
- Emotions: 
  - a = angry
  - d = disgust
  - f = fear
  - h = happy
  - n = neutral
  - sa = sad
  - su = surprise
"""

import os
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Emotion mapping from Savee filename codes
EMOTION_MAP = {
    'a': 'angry',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happy',
    'n': 'neutral',
    'sa': 'sad',
    'su': 'surprise'
}

def parse_savee_filename(filename):
    """Parse Savee filename to extract speaker and emotion."""
    # Pattern: {SPEAKER}_{EMOTION}{NUMBER}.wav
    # Examples: DC_a01.wav, JE_sa15.wav, KL_su01.wav
    match = re.match(r'([A-Z]{2})_([a-z]+)(\d+)\.wav', filename)
    if match:
        speaker, emotion_code, number = match.groups()
        emotion = EMOTION_MAP.get(emotion_code)
        if emotion:
            return speaker, emotion
    return None, None

def process_savee_dataset(savee_dir, output_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Process Savee dataset and create train/val/test splits.
    
    Uses speaker-based splitting to avoid data leakage:
    - Train: 2 speakers
    - Val: 1 speaker  
    - Test: 1 speaker
    """
    savee_path = Path(savee_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all audio files
    audio_files = sorted(list(savee_path.glob('*.wav')))
    print(f"Found {len(audio_files)} audio files")
    
    # Parse files and create dataframe
    data = []
    project_root = Path.cwd()
    for audio_file in audio_files:
        speaker, emotion = parse_savee_filename(audio_file.name)
        if speaker and emotion:
            # Relative path from project root
            try:
                rel_path = str(audio_file.relative_to(project_root))
            except ValueError:
                # If not relative, use the path as is but make it relative
                rel_path = str(audio_file)
                # Ensure it starts with datasets/SAVEE
                if not rel_path.startswith('datasets/SAVEE'):
                    rel_path = f"datasets/SAVEE/{audio_file.name}"
            data.append({
                'file_path': rel_path,
                'emotion_code': emotion,
                'speaker': speaker
            })
        else:
            print(f"Warning: Could not parse filename: {audio_file.name}")
    
    df = pd.DataFrame(data)
    print(f"\nTotal samples: {len(df)}")
    print(f"\nEmotion distribution:")
    print(df['emotion_code'].value_counts().sort_index())
    print(f"\nSpeaker distribution:")
    print(df['speaker'].value_counts().sort_index())
    
    # Get unique speakers
    speakers = df['speaker'].unique().tolist()
    print(f"\nSpeakers: {speakers}")
    
    # Split by speaker to avoid data leakage
    # With 4 speakers, we can do: 2 train, 1 val, 1 test
    import random
    random.seed(random_state)
    random.shuffle(speakers)
    
    train_speakers = speakers[:2]
    val_speaker = [speakers[2]]
    test_speaker = [speakers[3]]
    
    print(f"\nSpeaker split:")
    print(f"  Train: {train_speakers}")
    print(f"  Val: {val_speaker}")
    print(f"  Test: {test_speaker}")
    
    # Split dataset
    df_train = df[df['speaker'].isin(train_speakers)].copy()
    df_val = df[df['speaker'].isin(val_speaker)].copy()
    df_test = df[df['speaker'].isin(test_speaker)].copy()
    
    # Drop speaker column (not needed in CSV)
    df_train = df_train[['file_path', 'emotion_code']]
    df_val = df_val[['file_path', 'emotion_code']]
    df_test = df_test[['file_path', 'emotion_code']]
    
    # Shuffle within splits
    df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_val = df_val.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save splits
    train_csv = output_path / 'savee_train.csv'
    val_csv = output_path / 'savee_val.csv'
    test_csv = output_path / 'savee_test.csv'
    all_csv = output_path / 'savee_all.csv'
    
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    df.to_csv(all_csv, index=False)
    
    print(f"\nSaved CSV files:")
    print(f"  Train: {train_csv} ({len(df_train)} samples)")
    print(f"  Val: {val_csv} ({len(df_val)} samples)")
    print(f"  Test: {test_csv} ({len(df_test)} samples)")
    print(f"  All: {all_csv} ({len(df)} samples)")
    
    # Create emotion mapping
    emotions = sorted(df['emotion_code'].unique())
    emotion_map = {emotion: idx for idx, emotion in enumerate(emotions)}
    
    emotion_map_path = output_path / 'savee_emotion_map.json'
    with open(emotion_map_path, 'w') as f:
        json.dump(emotion_map, f, indent=2)
    
    print(f"\nEmotion mapping ({len(emotion_map)} classes):")
    for emotion, idx in sorted(emotion_map.items(), key=lambda x: x[1]):
        print(f"  {idx}: {emotion}")
    print(f"\nSaved emotion mapping to: {emotion_map_path}")
    
    # Print split statistics
    print(f"\nSplit statistics:")
    print(f"Train emotion distribution:")
    print(df_train['emotion_code'].value_counts().sort_index())
    print(f"\nVal emotion distribution:")
    print(df_val['emotion_code'].value_counts().sort_index())
    print(f"\nTest emotion distribution:")
    print(df_test['emotion_code'].value_counts().sort_index())
    
    return df_train, df_val, df_test, emotion_map

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Savee dataset')
    parser.add_argument('--savee_dir', type=str, default='datasets/SAVEE',
                       help='Path to Savee dataset directory')
    parser.add_argument('--output_dir', type=str, default='datasets/processed',
                       help='Output directory for CSV files')
    parser.add_argument('--test_size', type=float, default=0.15,
                       help='Test set size (not used with speaker split)')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Validation set size (not used with speaker split)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    process_savee_dataset(
        args.savee_dir,
        args.output_dir,
        args.test_size,
        args.val_size,
        args.random_state
    )

