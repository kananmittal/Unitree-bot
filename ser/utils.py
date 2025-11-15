"""
Utility functions for SER system
"""
import os
import torch
import json
from pathlib import Path
from typing import Dict, Optional


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Resolve audio file path with multiple fallback strategies
    
    Args:
        path: Original path (absolute or relative)
        base_dir: Base directory for relative paths
    
    Returns:
        Resolved absolute path
    """
    # If absolute and exists, return as-is
    if os.path.isabs(path) and os.path.exists(path):
        return path
    
    # Try relative to current directory
    if os.path.exists(path):
        return os.path.abspath(path)
    
    # Try relative to base_dir
    if base_dir:
        base_relative = os.path.join(base_dir, path)
        if os.path.exists(base_relative):
            return base_relative
    
    # Try relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_relative = os.path.join(project_root, path)
    if os.path.exists(project_relative):
        return project_relative
    
    # Return original path (will fail with clear error message later)
    return path


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
    
    Returns:
        Path to created experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    else:
        exp_dir = os.path.join(base_dir, timestamp)
    
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """Track metrics across epochs"""
    
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_uar": [],
            "val_f1": [],
            "learning_rate": []
        }
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self, path: str):
        """Save metrics to JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from JSON"""
        with open(path, 'r') as f:
            self.history = json.load(f)
    
    def get_best_epoch(self, metric: str = "val_uar") -> int:
        """Get epoch with best metric value"""
        if metric not in self.history or not self.history[metric]:
            return 0
        return self.history[metric].index(max(self.history[metric])) + 1


def plot_training_history(metrics: MetricsTracker, output_path: str):
    """Plot training history"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return
    
    history = metrics.history
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history["val_acc"], label="Val Accuracy", color='green')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # UAR
    axes[1, 0].plot(epochs, history["val_uar"], label="Val UAR", color='orange')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("UAR")
    axes[1, 0].set_title("Validation UAR (Unweighted Average Recall)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history["learning_rate"], label="Learning Rate", color='red')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved training history plot to: {output_path}")


def check_dataset_balance(csv_path: str):
    """Check class balance in dataset"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    if 'emotion_code' in df.columns:
        label_col = 'emotion_code'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        print("No emotion/label column found")
        return
    
    counts = df[label_col].value_counts().sort_index()
    total = len(df)
    
    print("\n" + "=" * 60)
    print(f"Dataset Balance: {csv_path}")
    print("=" * 60)
    print(f"{'Class':<20} {'Count':<10} {'Percentage':<12}")
    print("-" * 60)
    
    for label, count in counts.items():
        pct = count / total * 100
        print(f"{str(label):<20} {count:<10} {pct:>10.2f}%")
    
    print("-" * 60)
    print(f"{'Total':<20} {total:<10} {100.0:>10.2f}%")
    print("=" * 60)
    
    # Check imbalance
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        print("⚠️  Dataset is highly imbalanced. Consider using class weights.")
    elif imbalance_ratio > 1.5:
        print("⚠️  Dataset is moderately imbalanced.")
    else:
        print("✓ Dataset is balanced.")