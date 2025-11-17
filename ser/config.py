"""
Configuration management for SER system
Supports YAML config files and command-line overrides
"""
import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, Any

@dataclass
class AudioConfig:
    """Audio preprocessing configuration"""
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 20.0
    f_max: float = 8000.0
    
@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    enabled: bool = True
    time_stretch_prob: float = 0.3
    time_stretch_rate: Tuple[float, float] = (0.8, 1.2)
    pitch_shift_prob: float = 0.3
    pitch_shift_steps: Tuple[int, int] = (-2, 2)
    noise_injection_prob: float = 0.3
    noise_level: Tuple[float, float] = (0.001, 0.01)
    mixup_prob: float = 0.2
    mixup_alpha: float = 0.4
    
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    n_mfcc: int = 40
    num_classes: int = 4
    conv_channels: Tuple[int, ...] = (128, 256)  # Deeper network
    lstm_hidden: int = 128
    lstm_layers: int = 2  # More LSTM layers
    dropout: float = 0.2  # Increased dropout
    use_depthwise: bool = True
    use_temp_scale: bool = True
    use_residual: bool = True  # New: residual connections
    attention_heads: int = 4  # New: multi-head attention
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 10  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement for early stopping
    grad_clip: float = 1.0  # Gradient clipping
    label_smoothing: float = 0.1  # Label smoothing
    class_weights: bool = True  # Use class weights for imbalanced data
    scheduler: str = "cosine"  # "cosine", "plateau", or "step"
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "cosine": {"T_max": 50, "eta_min": 1e-6},
        "plateau": {"factor": 0.5, "patience": 5, "min_lr": 1e-6},
        "step": {"step_size": 10, "gamma": 0.5}
    })
    
@dataclass
class DataConfig:
    """Dataset configuration"""
    train_csv: str = "datasets/processed/iemocap_train.csv"
    val_csv: Optional[str] = "datasets/processed/iemocap_val.csv"
    test_csv: Optional[str] = "datasets/processed/iemocap_test.csv"
    cache_dir: str = "feat_cache"
    num_workers: int = 2
    val_split: float = 0.1
    pin_memory: bool = True
    max_audio_length: float = 10.0  # Maximum audio length in seconds
    min_audio_length: float = 0.5   # Minimum audio length in seconds
    
@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "auto"  # "auto", "cuda", "cpu"
    seed: int = 42
    amp: bool = True  # Automatic Mixed Precision
    out_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_best_only: bool = False
    save_interval: int = 5  # Save every N epochs
    
@dataclass
class Config:
    """Complete configuration"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        if 'audio' in data:
            config.audio = AudioConfig(**data['audio'])
        if 'augmentation' in data:
            # Convert lists to tuples for augmentation config
            aug_data = data['augmentation'].copy()
            if 'time_stretch_rate' in aug_data and isinstance(aug_data['time_stretch_rate'], list):
                aug_data['time_stretch_rate'] = tuple(aug_data['time_stretch_rate'])
            if 'pitch_shift_steps' in aug_data and isinstance(aug_data['pitch_shift_steps'], list):
                aug_data['pitch_shift_steps'] = tuple(aug_data['pitch_shift_steps'])
            if 'noise_level' in aug_data and isinstance(aug_data['noise_level'], list):
                aug_data['noise_level'] = tuple(aug_data['noise_level'])
            config.augmentation = AugmentationConfig(**aug_data)
        if 'model' in data:
            # Convert list to tuple for conv_channels
            model_data = data['model'].copy()
            if 'conv_channels' in model_data and isinstance(model_data['conv_channels'], list):
                model_data['conv_channels'] = tuple(model_data['conv_channels'])
            config.model = ModelConfig(**model_data)
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'system' in data:
            config.system = SystemConfig(**data['system'])
        
        return config
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        data = {
            'audio': asdict(self.audio),
            'augmentation': asdict(self.augmentation),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'system': asdict(self.system)
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'audio': asdict(self.audio),
            'augmentation': asdict(self.augmentation),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'system': asdict(self.system)
        }
    
    def update_from_args(self, args):
        """Update configuration from command-line arguments"""
        # Update data config
        if hasattr(args, 'train_csv') and args.train_csv:
            self.data.train_csv = args.train_csv
        if hasattr(args, 'val_csv') and args.val_csv:
            self.data.val_csv = args.val_csv
        if hasattr(args, 'test_csv') and args.test_csv:
            self.data.test_csv = args.test_csv
        if hasattr(args, 'cache_dir') and args.cache_dir:
            self.data.cache_dir = args.cache_dir
        if hasattr(args, 'num_workers') and args.num_workers is not None:
            self.data.num_workers = args.num_workers
        
        # Update training config
        if hasattr(args, 'epochs') and args.epochs:
            self.training.epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'lr') and args.lr:
            self.training.lr = args.lr
        if hasattr(args, 'patience') and args.patience:
            self.training.patience = args.patience
        
        # Update model config
        if hasattr(args, 'num_classes') and args.num_classes:
            self.model.num_classes = args.num_classes
        if hasattr(args, 'n_mfcc') and args.n_mfcc:
            self.model.n_mfcc = args.n_mfcc
            self.audio.n_mfcc = args.n_mfcc
        
        # Update system config
        if hasattr(args, 'out_dir') and args.out_dir:
            self.system.out_dir = args.out_dir
        if hasattr(args, 'amp') and args.amp is not None:
            self.system.amp = args.amp
        if hasattr(args, 'device') and args.device:
            self.system.device = args.device
        if hasattr(args, 'seed') and args.seed:
            self.system.seed = args.seed

def create_default_config(path: str = "config.yaml"):
    """Create default configuration file"""
    config = Config()
    config.to_yaml(path)
    print(f"Created default configuration: {path}")

if __name__ == "__main__":
    # Create default config
    create_default_config("config_default.yaml")
    
    # Test loading
    config = Config.from_yaml("config_default.yaml")
    print("Configuration loaded successfully")
    print(f"Model channels: {config.model.conv_channels}")
    print(f"Training epochs: {config.training.epochs}")