"""
Data augmentation for audio/MFCC features
Includes time stretch, pitch shift, noise injection, and mixup
"""
import torch
import torchaudio
import random
import numpy as np
from typing import Tuple, Optional

class AudioAugmentation:
    """Audio-level augmentation (before MFCC extraction)"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        time_stretch_prob: float = 0.3,
        time_stretch_rate: Tuple[float, float] = (0.8, 1.2),
        pitch_shift_prob: float = 0.3,
        pitch_shift_steps: Tuple[int, int] = (-2, 2),
        noise_injection_prob: float = 0.3,
        noise_level: Tuple[float, float] = (0.001, 0.01)
    ):
        self.sample_rate = sample_rate
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_rate = time_stretch_rate
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_steps = pitch_shift_steps
        self.noise_injection_prob = noise_injection_prob
        self.noise_level = noise_level
    
    def time_stretch(self, wav: torch.Tensor, rate: Optional[float] = None) -> torch.Tensor:
        """
        Time stretch without changing pitch
        wav: (T,)
        rate: stretch rate (0.8 = slower, 1.2 = faster)
        """
        if rate is None:
            rate = random.uniform(*self.time_stretch_rate)
        
        # Use phase vocoder for time stretching
        try:
            # Convert to spectrogram
            n_fft = 512
            hop_length = 160
            spec = torch.stft(
                wav, n_fft=n_fft, hop_length=hop_length,
                window=torch.hann_window(n_fft),
                return_complex=True
            )
            
            # Time stretch in spectrogram domain
            stretched_spec = torchaudio.functional.phase_vocoder(
                spec, rate=rate, phase_advance=hop_length
            )
            
            # Convert back to waveform
            stretched = torch.istft(
                stretched_spec, n_fft=n_fft, hop_length=hop_length,
                window=torch.hann_window(n_fft)
            )
            
            return stretched
        except:
            # Fallback: simple resampling
            target_length = int(len(wav) / rate)
            return torch.nn.functional.interpolate(
                wav.unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode='linear',
                align_corners=False
            ).squeeze()
    
    def pitch_shift(self, wav: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """
        Pitch shift (changes pitch, not speed)
        wav: (T,)
        steps: semitones to shift (-2 to +2)
        """
        if steps is None:
            steps = random.randint(*self.pitch_shift_steps)
        
        if steps == 0:
            return wav
        
        # Pitch shift by resampling
        rate = 2 ** (steps / 12.0)
        return torchaudio.functional.resample(
            wav, self.sample_rate, int(self.sample_rate * rate)
        )
    
    def add_noise(self, wav: torch.Tensor, noise_level: Optional[float] = None) -> torch.Tensor:
        """
        Add Gaussian noise
        wav: (T,)
        noise_level: standard deviation of noise
        """
        if noise_level is None:
            noise_level = random.uniform(*self.noise_level)
        
        noise = torch.randn_like(wav) * noise_level
        return wav + noise
    
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations"""
        # Time stretch
        if random.random() < self.time_stretch_prob:
            wav = self.time_stretch(wav)
        
        # Pitch shift
        if random.random() < self.pitch_shift_prob:
            wav = self.pitch_shift(wav)
        
        # Noise injection
        if random.random() < self.noise_injection_prob:
            wav = self.add_noise(wav)
        
        return wav


class MFCCAugmentation:
    """MFCC-level augmentation (after MFCC extraction)"""
    
    def __init__(
        self,
        freq_mask_prob: float = 0.3,
        freq_mask_param: int = 8,
        time_mask_prob: float = 0.3,
        time_mask_param: int = 25
    ):
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_param = freq_mask_param
        self.time_mask_prob = time_mask_prob
        self.time_mask_param = time_mask_param
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
    
    def __call__(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment-style masking to MFCC
        mfcc: (C, T)
        """
        # Frequency masking (mask MFCC coefficients)
        if random.random() < self.freq_mask_prob:
            mfcc = self.freq_masking(mfcc.unsqueeze(0)).squeeze(0)
        
        # Time masking
        if random.random() < self.time_mask_prob:
            mfcc = self.time_masking(mfcc.unsqueeze(0)).squeeze(0)
        
        return mfcc


class MixupCollate:
    """
    Mixup augmentation at batch level
    Mixes two samples with random weight
    """
    
    def __init__(self, alpha: float = 0.4, prob: float = 0.2):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch, base_collate_fn):
        """
        Apply mixup to batch
        batch: list of (mfcc, label) tuples
        base_collate_fn: original collate function (pad_collate)
        """
        # First apply base collation
        mfcc, labels = base_collate_fn(batch)
        
        # Apply mixup with probability
        if random.random() < self.prob:
            # Sample lambda from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Random permutation
            batch_size = mfcc.size(0)
            index = torch.randperm(batch_size)
            
            # Mix inputs and labels
            mixed_mfcc = lam * mfcc + (1 - lam) * mfcc[index]
            
            # For labels, return both labels and mixing weight
            # (Will need to modify loss function to handle this)
            labels_a, labels_b = labels, labels[index]
            
            return mixed_mfcc, (labels_a, labels_b, lam)
        
        return mfcc, (labels, None, 1.0)


def mixup_criterion(criterion, pred, labels_info):
    """
    Loss function for mixup
    labels_info: (labels_a, labels_b, lam) or just labels
    """
    if isinstance(labels_info, tuple) and len(labels_info) == 3:
        labels_a, labels_b, lam = labels_info
        if labels_b is not None:
            # Mixup: combine losses from both labels
            return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)
    
    # No mixup: standard loss
    labels = labels_info[0] if isinstance(labels_info, tuple) else labels_info
    return criterion(pred, labels)


# Utility functions for augmentation configuration
def create_train_augmentation(config):
    """Create augmentation pipeline for training"""
    if not config.augmentation.enabled:
        return None, None
    
    audio_aug = AudioAugmentation(
        sample_rate=config.audio.sample_rate,
        time_stretch_prob=config.augmentation.time_stretch_prob,
        time_stretch_rate=config.augmentation.time_stretch_rate,
        pitch_shift_prob=config.augmentation.pitch_shift_prob,
        pitch_shift_steps=config.augmentation.pitch_shift_steps,
        noise_injection_prob=config.augmentation.noise_injection_prob,
        noise_level=config.augmentation.noise_level
    )
    
    mfcc_aug = MFCCAugmentation(
        freq_mask_prob=0.3,
        freq_mask_param=8,
        time_mask_prob=0.3,
        time_mask_param=25
    )
    
    return audio_aug, mfcc_aug


def create_mixup_collate(config, base_collate_fn):
    """Create mixup collate function"""
    if not config.augmentation.enabled or config.augmentation.mixup_prob == 0:
        return base_collate_fn
    
    mixup = MixupCollate(
        alpha=config.augmentation.mixup_alpha,
        prob=config.augmentation.mixup_prob
    )
    
    return lambda batch: mixup(batch, base_collate_fn)