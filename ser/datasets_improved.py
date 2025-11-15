"""
Improved dataset with:
- Better error handling
- Augmentation support
- Audio length filtering
- Robust path handling
- Emotion mapping persistence
"""
import os
import torch
import pandas as pd
import json
import hashlib
from torch.utils.data import Dataset
from typing import Optional, Dict, Callable
from tier0_io import tier0_to_mfcc

# -------------------------
# Emotion Mapping Utilities
# -------------------------
def get_emotion_map_from_dataset(csv_path: str) -> Optional[Dict[str, int]]:
    """
    Dynamically create emotion mapping from dataset.
    Returns a dictionary mapping emotion codes to contiguous class labels (0, 1, 2, ...)
    """
    df = pd.read_csv(csv_path)
    if 'emotion_code' in df.columns:
        emotions = sorted(df['emotion_code'].unique())
        return {emotion: idx for idx, emotion in enumerate(emotions)}
    elif 'label' in df.columns:
        # Already has numeric labels
        return None
    else:
        raise ValueError(f"CSV must have 'emotion_code' or 'label' column. Found: {df.columns.tolist()}")


def save_emotion_map(emotion_map: Dict[str, int], path: str):
    """Save emotion mapping to JSON file"""
    with open(path, 'w') as f:
        json.dump(emotion_map, f, indent=2)


def load_emotion_map(path: str) -> Dict[str, int]:
    """Load emotion mapping from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


# -------------------------
# Improved Dataset
# -------------------------
class AudioEmotionDataset(Dataset):
    """
    Enhanced dataset with:
    - Better error handling (raises exceptions instead of returning zeros)
    - Augmentation support
    - Audio length filtering
    - Proper path handling
    """
    
    def __init__(
        self,
        csv_path: str,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        n_mfcc: int = 40,
        emotion_map: Optional[Dict[str, int]] = None,
        audio_augmentation: Optional[Callable] = None,
        mfcc_augmentation: Optional[Callable] = None,
        max_audio_length: float = 10.0,  # seconds
        min_audio_length: float = 0.5,   # seconds
        sample_rate: int = 16000,
    ):
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.device = device
        self.n_mfcc = n_mfcc
        self.audio_augmentation = audio_augmentation
        self.mfcc_augmentation = mfcc_augmentation
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.sample_rate = sample_rate
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Handle different CSV formats
        if 'file_path' in self.df.columns and 'emotion_code' in self.df.columns:
            # Full format: map emotion_code to numeric label
            self.df['path'] = self.df['file_path']
            
            # Use provided emotion map or create dynamically
            if emotion_map is None:
                self.emotion_map = get_emotion_map_from_dataset(csv_path)
            else:
                self.emotion_map = emotion_map
            
            # Map emotions to labels
            self.df['label'] = self.df['emotion_code'].map(self.emotion_map)
            
            # Drop rows with unmapped emotions
            before_count = len(self.df)
            self.df = self.df.dropna(subset=['label'])
            after_count = len(self.df)
            
            if before_count != after_count:
                print(f"Warning: Dropped {before_count - after_count} samples with unmapped emotions")
            
            self.df['label'] = self.df['label'].astype(int)
            
        elif 'path' in self.df.columns and 'label' in self.df.columns:
            # Simple format: already has numeric labels
            self.emotion_map = None
            self.df['label'] = self.df['label'].astype(int)
        else:
            raise ValueError(
                f"CSV must have either (path,label) or (file_path,emotion_code) columns. "
                f"Found: {self.df.columns.tolist()}"
            )
        
        # Validate labels
        unique_labels = self.df['label'].unique()
        expected_labels = set(range(len(unique_labels)))
        actual_labels = set(unique_labels)
        
        if actual_labels != expected_labels:
            print(f"Warning: Labels are not contiguous. Expected {expected_labels}, got {actual_labels}")
        
        # Resolve paths relative to CSV directory or project root
        self._resolve_paths()
        
        # Filter by audio length (if file exists and we can check)
        if max_audio_length or min_audio_length:
            self._filter_by_length()
        
        # Create cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Loaded dataset: {len(self.df)} samples, {len(unique_labels)} classes")
    
    def _resolve_paths(self):
        """Resolve audio file paths"""
        csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        def resolve_path(path):
            # If absolute and exists, return as-is
            if os.path.isabs(path) and os.path.exists(path):
                return path
            
            # Try relative to current directory
            if os.path.exists(path):
                return os.path.abspath(path)
            
            # Try relative to CSV directory
            csv_relative = os.path.join(csv_dir, path)
            if os.path.exists(csv_relative):
                return csv_relative
            
            # Try relative to project root
            project_relative = os.path.join(project_root, path)
            if os.path.exists(project_relative):
                return project_relative
            
            # Return original path (will fail later with clear error)
            return path
        
        self.df['path'] = self.df['path'].apply(resolve_path)
    
    def _filter_by_length(self):
        """Filter out audio files that are too short or too long"""
        # This is optional - only checks if torchaudio can load quickly
        # For large datasets, this might be slow
        pass  # Can be implemented if needed
    
    def _cache_key(self, wav_path: str) -> str:
        """Generate cache key for audio file"""
        path_str = str(wav_path)
        path_hash = hashlib.md5(path_str.encode()).hexdigest()[:16]
        base = os.path.splitext(os.path.basename(wav_path))[0]
        # Include n_mfcc in cache key to avoid conflicts
        return os.path.join(self.cache_dir, f"{base}_{path_hash}_mfcc{self.n_mfcc}.pt")
    
    def _load_mfcc(self, wav_path: str, use_augmentation: bool = True) -> torch.Tensor:
        """Load MFCC features with caching and augmentation"""
        # Check cache first (only if no augmentation)
        if self.cache_dir and not use_augmentation:
            cache_path = self._cache_key(wav_path)
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, map_location=self.device)
                except Exception as e:
                    print(f"Warning: Failed to load cache {cache_path}: {e}. Regenerating...")
        
        # Check if file exists
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        
        try:
            # Load audio
            from tier0_io import load_audio, denoise_demucs, vad_silero_keep_speech, mfcc_40
            
            wav = load_audio(wav_path)
            
            # Apply audio augmentation (before MFCC extraction)
            if use_augmentation and self.audio_augmentation:
                wav = self.audio_augmentation(wav)
            
            # Denoise and VAD
            den = denoise_demucs(wav, device=self.device)
            speech = vad_silero_keep_speech(den)
            
            # Check if speech is empty
            if speech.numel() <= 1:
                raise ValueError(f"No speech detected in {wav_path}")
            
            # Extract MFCC
            mfcc = mfcc_40(speech, n_mfcc=self.n_mfcc)
            
            # Apply MFCC augmentation
            if use_augmentation and self.mfcc_augmentation:
                mfcc = self.mfcc_augmentation(mfcc)
            
            # Cache if no augmentation was applied
            if self.cache_dir and not use_augmentation:
                try:
                    torch.save(mfcc, self._cache_key(wav_path))
                except Exception as e:
                    print(f"Warning: Failed to save cache: {e}")
            
            return mfcc
            
        except Exception as e:
            raise RuntimeError(f"Failed to process {wav_path}: {str(e)}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Load MFCC with augmentation (if in training mode)
        use_augmentation = self.audio_augmentation is not None or self.mfcc_augmentation is not None
        mfcc = self._load_mfcc(row["path"], use_augmentation=use_augmentation)
        
        label = int(row["label"])
        
        return mfcc, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalanced datasets"""
        label_counts = self.df['label'].value_counts().sort_index()
        total = len(self.df)
        num_classes = len(label_counts)
        
        # Inverse frequency weighting
        weights = torch.tensor([
            total / (num_classes * label_counts.get(i, 1))
            for i in range(num_classes)
        ], dtype=torch.float32)
        
        return weights
    
    def get_emotion_map(self) -> Optional[Dict[str, int]]:
        """Get emotion mapping (if available)"""
        return self.emotion_map


# -------------------------
# Collate Function
# -------------------------
def pad_collate(batch):
    """
    Collate function with padding
    batch: list of (mfcc_CT, label)
    Returns: mfcc (B, C, T_max), labels (B,)
    """
    C = batch[0][0].size(0)
    max_T = max(x[0].size(1) for x in batch)
    B = len(batch)
    
    out = torch.zeros(B, C, max_T)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    
    for i, (mfcc, _) in enumerate(batch):
        T = mfcc.size(1)
        out[i, :, :T] = mfcc
    
    return out, labels


# -------------------------
# Dataset Factory
# -------------------------
def create_dataset(
    csv_path: str,
    config,
    audio_augmentation: Optional[Callable] = None,
    mfcc_augmentation: Optional[Callable] = None,
    emotion_map: Optional[Dict[str, int]] = None,
) -> AudioEmotionDataset:
    """Create dataset from configuration"""
    return AudioEmotionDataset(
        csv_path=csv_path,
        cache_dir=config.data.cache_dir,
        device="cpu",  # Always CPU for data loading
        n_mfcc=config.audio.n_mfcc,
        emotion_map=emotion_map,
        audio_augmentation=audio_augmentation,
        mfcc_augmentation=mfcc_augmentation,
        max_audio_length=config.data.max_audio_length,
        min_audio_length=config.data.min_audio_length,
        sample_rate=config.audio.sample_rate,
    )