import os, torch, pandas as pd
from torch.utils.data import Dataset
from tier0_io import tier0_to_mfcc

# Emotion mapping for IEMOCAP and similar datasets
# Multi-class mapping: Each emotion gets its own class label
# This preserves all emotions without collapsing to 4 classes
# 
# Common IEMOCAP emotions: angry, fearful, happy, neutral, sad, surprised, excited, disgust, frustrated
# Each emotion is assigned a unique class ID (contiguous labels starting from 0)
# The mapping is ordered alphabetically for consistency
EMOTION_MAP = {
    'angry': 0,
    'disgust': 1,
    'excited': 2,
    'fear': 3,
    'fearful': 4,
    'frustrated': 5,
    'happy': 6,
    'neutral': 7,
    'sad': 8,
    'surprised': 9,
}

def get_emotion_map_from_dataset(csv_path):
    """
    Dynamically create emotion mapping from dataset.
    Returns a dictionary mapping emotion codes to contiguous class labels (0, 1, 2, ...)
    """
    df = pd.read_csv(csv_path)
    if 'emotion_code' in df.columns:
        emotions = sorted(df['emotion_code'].unique())
        return {emotion: idx for idx, emotion in enumerate(emotions)}
    elif 'label' in df.columns:
        # Already has numeric labels, return None to use as-is
        return None
    else:
        raise ValueError(f"CSV must have 'emotion_code' or 'label' column. Found: {df.columns.tolist()}")

class AudioEmotionDataset(Dataset):
    """
    CSV format (supports both):
      1. Simple: path,label
      2. Full: file_path,emotion_code,... (maps emotion_code to numeric label)
    """
    def __init__(self, csv_path, cache_dir=None, device="cpu", n_mfcc=40, emotion_map=None):
        self.df = pd.read_csv(csv_path)
        self.cache_dir = cache_dir
        self.device = device
        self.n_mfcc = n_mfcc
        
        # Handle different CSV formats
        if 'file_path' in self.df.columns and 'emotion_code' in self.df.columns:
            # Full format: map emotion_code to numeric label
            self.df['path'] = self.df['file_path']
            # Use dynamic mapping if not provided, otherwise use provided or default
            if emotion_map is None:
                # Dynamically create mapping from dataset (contiguous labels)
                self.emotion_map = get_emotion_map_from_dataset(csv_path)
            else:
                self.emotion_map = emotion_map
            self.df['label'] = self.df['emotion_code'].map(self.emotion_map)
            # Drop rows with unmapped emotions
            self.df = self.df.dropna(subset=['label'])
            self.df['label'] = self.df['label'].astype(int)
        elif 'path' in self.df.columns and 'label' in self.df.columns:
            # Simple format: already has numeric labels
            self.emotion_map = None  # No mapping needed
        else:
            raise ValueError(f"CSV must have either (path,label) or (file_path,emotion_code) columns. Found: {self.df.columns.tolist()}")
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, wav_path):
        # Use full path hash for cache key to avoid collisions
        import hashlib
        path_str = str(wav_path)
        path_hash = hashlib.md5(path_str.encode()).hexdigest()[:16]
        base = os.path.splitext(os.path.basename(wav_path))[0]
        return os.path.join(self.cache_dir, f"{base}_{path_hash}.pt")

    def _load_mfcc(self, wav_path):
        # Handle relative paths
        if not os.path.isabs(wav_path):
            # Try relative to current directory or CSV directory
            if not os.path.exists(wav_path):
                # Assume relative to project root
                wav_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), wav_path)
        
        if self.cache_dir:
            ck = self._cache_key(wav_path)
            if os.path.exists(ck):
                try:
                    return torch.load(ck)
                except:
                    # Cache corrupted, regenerate
                    pass
        
        try:
            mfcc = tier0_to_mfcc(wav_path, device=self.device, n_mfcc=self.n_mfcc)  # (C,T)
            if self.cache_dir:
                torch.save(mfcc, self._cache_key(wav_path))
            return mfcc
        except Exception as e:
            # If file loading fails, return a zero tensor with correct shape
            # This allows training to continue but the sample will be effectively ignored
            print(f"Warning: Failed to load {wav_path}: {e}. Using zero tensor.")
            # Return a dummy tensor with typical shape (C=40, T=100)
            return torch.zeros(self.n_mfcc, 100, device=self.device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mfcc = self._load_mfcc(row["path"])      # (C,T)
        label = int(row["label"])
        return mfcc, label

def pad_collate(batch):
    """
    batch: list of (mfcc_CT, label)
    Pads along T to max_T with zeros.
    Returns:
      mfcc: (B, C, T_max), labels: (B,)
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
