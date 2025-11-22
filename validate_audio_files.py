"""
Script to validate audio files and identify corrupted files.
Run this before training to clean your dataset.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa


def validate_audio_file(file_path):
    """
    Validate if an audio file can be loaded.
    Returns: (is_valid, error_message, duration)
    """
    try:
        # Try with soundfile first (faster)
        with sf.SoundFile(file_path) as f:
            duration = len(f) / f.samplerate
            return True, None, duration
    except Exception as e1:
        # Try with librosa as fallback
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            return True, None, duration
        except Exception as e2:
            return False, f"soundfile: {str(e1)}, librosa: {str(e2)}", 0.0


def validate_dataset(csv_path, output_cleaned_csv=None):
    """
    Validate all audio files in a CSV dataset.
    
    Args:
        csv_path: Path to the CSV file with 'path' column
        output_cleaned_csv: If provided, saves cleaned CSV without corrupted files
    
    Returns:
        valid_df, corrupted_files
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'path' not in df.columns:
        raise ValueError(f"CSV must have 'path' column. Found: {df.columns.tolist()}")
    
    print(f"Total files to validate: {len(df)}")
    print("\nValidating audio files...")
    
    valid_files = []
    corrupted_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['path']
        
        # Check if file exists
        if not os.path.exists(file_path):
            corrupted_files.append({
                'path': file_path,
                'error': 'File not found',
                'emotion': row.get('emotion', 'N/A'),
                'emotion_code': row.get('emotion_code', 'N/A')
            })
            continue
        
        # Validate audio
        is_valid, error, duration = validate_audio_file(file_path)
        
        if is_valid:
            valid_files.append(idx)
        else:
            corrupted_files.append({
                'path': file_path,
                'error': error,
                'emotion': row.get('emotion', 'N/A'),
                'emotion_code': row.get('emotion_code', 'N/A'),
                'duration': duration
            })
    
    # Create cleaned dataframe
    valid_df = df.iloc[valid_files].copy()
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total files: {len(df)}")
    print(f"Valid files: {len(valid_files)} ({len(valid_files)/len(df)*100:.2f}%)")
    print(f"Corrupted files: {len(corrupted_files)} ({len(corrupted_files)/len(df)*100:.2f}%)")
    
    if corrupted_files:
        print("\n" + "="*60)
        print("CORRUPTED FILES:")
        print("="*60)
        for i, item in enumerate(corrupted_files[:20], 1):  # Show first 20
            print(f"{i}. {os.path.basename(item['path'])}")
            print(f"   Emotion: {item['emotion']} (code: {item['emotion_code']})")
            print(f"   Error: {item['error'][:100]}...")
            print()
        
        if len(corrupted_files) > 20:
            print(f"... and {len(corrupted_files) - 20} more corrupted files")
        
        # Save corrupted files list
        corrupted_df = pd.DataFrame(corrupted_files)
        corrupted_csv = csv_path.replace('.csv', '_corrupted.csv')
        corrupted_df.to_csv(corrupted_csv, index=False)
        print(f"\nSaved list of corrupted files to: {corrupted_csv}")
    
    # Check emotion distribution in cleaned dataset
    if 'emotion' in valid_df.columns:
        print("\n" + "="*60)
        print("EMOTION DISTRIBUTION (CLEANED DATASET):")
        print("="*60)
        print(valid_df['emotion'].value_counts().sort_index())
    
    # Save cleaned dataset
    if output_cleaned_csv:
        valid_df.to_csv(output_cleaned_csv, index=False)
        print(f"\n✓ Saved cleaned dataset to: {output_cleaned_csv}")
    
    return valid_df, corrupted_files


def validate_all_splits(processed_dir='datasets/processed', dataset_name='crema'):
    """
    Validate train, val, and test splits.
    """
    splits = ['train', 'val', 'test']
    
    print("="*60)
    print(f"VALIDATING {dataset_name.upper()} DATASET")
    print("="*60)
    
    all_corrupted = []
    
    for split in splits:
        csv_path = os.path.join(processed_dir, f'{dataset_name}_{split}.csv')
        
        if not os.path.exists(csv_path):
            print(f"\nSkipping {split}: {csv_path} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"VALIDATING {split.upper()} SPLIT")
        print(f"{'='*60}")
        
        output_path = os.path.join(processed_dir, f'{dataset_name}_{split}_cleaned.csv')
        valid_df, corrupted = validate_dataset(csv_path, output_path)
        
        all_corrupted.extend(corrupted)
    
    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Total corrupted files across all splits: {len(all_corrupted)}")
    
    if all_corrupted:
        print("\nRecommendations:")
        print("1. Use the *_cleaned.csv files for training")
        print("2. Or delete/redownload the corrupted files listed above")
        print("3. Update your config to point to the cleaned CSV files:")
        print(f"   train_csv: datasets/processed/{dataset_name}_train_cleaned.csv")
        print(f"   val_csv: datasets/processed/{dataset_name}_val_cleaned.csv")
        print(f"   test_csv: datasets/processed/{dataset_name}_test_cleaned.csv")
    else:
        print("\n✓ All files are valid! No cleaning needed.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate audio dataset')
    parser.add_argument('--dataset', type=str, default='crema', 
                       help='Dataset name (crema, ravdess, savee)')
    parser.add_argument('--processed-dir', type=str, default='datasets/processed',
                       help='Directory containing processed CSV files')
    parser.add_argument('--csv', type=str, default=None,
                       help='Single CSV file to validate (overrides --dataset)')
    
    args = parser.parse_args()
    
    if args.csv:
        # Validate single CSV
        output_path = args.csv.replace('.csv', '_cleaned.csv')
        validate_dataset(args.csv, output_path)
    else:
        # Validate all splits
        validate_all_splits(args.processed_dir, args.dataset)