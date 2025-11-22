"""
Improved training script with:
- Configuration system
- Data augmentation
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Better checkpointing
- Emotion mapping persistence
"""
import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from pathlib import Path
import json
from datetime import datetime

# Import modules
from config import Config
from datasets import create_dataset, pad_collate, get_emotion_map_from_dataset, save_emotion_map
from tier1_model import create_model, SERLoss
from augmentation import create_train_augmentation, create_mixup_collate, mixup_criterion
from metrics import compute_metrics

def parse_args():
    ap = argparse.ArgumentParser(description="Train SER model with improved features")
    
    # Config file
    ap.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    
    # Data arguments (override config)
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--val_csv", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    
    # Training arguments
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    
    # Model arguments
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--n_mfcc", type=int, default=None)
    
    # System arguments
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--amp", action="store_true", default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    
    # Augmentation
    ap.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
    
    # Resume training
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    return ap.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config):
    """Get device for training"""
    if config.system.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.system.device)


def create_optimizer(model, config):
    """Create optimizer with weight decay"""
    return optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )


def create_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler"""
    scheduler_type = config.training.scheduler
    params = config.training.scheduler_params
    
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params["cosine"]["T_max"],
            eta_min=params["cosine"]["eta_min"]
        )
    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize UAR
            factor=params["plateau"]["factor"],
            patience=params["plateau"]["patience"],
            min_lr=params["plateau"]["min_lr"]
        )
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params["step"]["step_size"],
            gamma=params["step"]["gamma"]
        )
    else:
        return None


def create_warmup_scheduler(optimizer, config, steps_per_epoch):
    """Create warmup scheduler"""
    warmup_steps = config.training.warmup_epochs * steps_per_epoch
    
    def warmup_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    return optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)


def make_loaders(config, device):
    """Create data loaders with augmentation"""
    import platform
    
    # Adjust num_workers for macOS
    if platform.system() == "Darwin" and config.data.num_workers > 0:
        config.data.num_workers = 0
        print("Note: Using num_workers=0 on macOS for compatibility")
    
    # Get emotion mapping from training data
    emotion_map = get_emotion_map_from_dataset(config.data.train_csv)
    
    # Save emotion mapping
    if emotion_map:
        os.makedirs(config.system.out_dir, exist_ok=True)
        emotion_map_path = os.path.join(config.system.out_dir, "emotion_map.json")
        save_emotion_map(emotion_map, emotion_map_path)
        print(f"Saved emotion mapping to: {emotion_map_path}")
    
    # Auto-detect num_classes if not set
    if config.model.num_classes is None:
        if emotion_map:
            config.model.num_classes = len(emotion_map)
        else:
            # Will be detected from dataset
            pass
    
    # Create augmentation pipelines
    audio_aug, mfcc_aug = create_train_augmentation(config)
    
    # Create datasets
    if config.data.val_csv is None:
        # Split training data
        full_ds = create_dataset(
            config.data.train_csv,
            config,
            audio_augmentation=audio_aug,
            mfcc_augmentation=mfcc_aug,
            emotion_map=emotion_map
        )
        
        n_total = len(full_ds)
        n_val = int(config.data.val_split * n_total)
        n_train = n_total - n_val
        
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.system.seed)
        )
    else:
        # Separate train/val files
        train_ds = create_dataset(
            config.data.train_csv,
            config,
            audio_augmentation=audio_aug,
            mfcc_augmentation=mfcc_aug,
            emotion_map=emotion_map
        )
        
        # No augmentation for validation
        val_ds = create_dataset(
            config.data.val_csv,
            config,
            audio_augmentation=None,
            mfcc_augmentation=None,
            emotion_map=emotion_map
        )
    
    # Auto-detect num_classes from dataset if still None
    if config.model.num_classes is None:
        if hasattr(train_ds, 'dataset'):
            config.model.num_classes = len(train_ds.dataset.df['label'].unique())
        else:
            config.model.num_classes = len(train_ds.df['label'].unique())
    
    # Create collate function (with mixup if enabled)
    collate_fn = create_mixup_collate(config, pad_collate)
    
    # Create data loaders
    pin_mem = device.type == "cuda" and config.data.pin_memory
    
    train_ld = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        drop_last=True  # For stable batch norm
    )
    
    val_ld = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=pad_collate,  # No mixup for validation
        pin_memory=pin_mem
    )
    
    # Get class weights if enabled
    class_weights = None
    if config.training.class_weights:
        if hasattr(train_ds, 'dataset'):
            class_weights = train_ds.dataset.get_class_weights()
        else:
            class_weights = train_ds.get_class_weights()
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights.tolist()}")
    
    return train_ld, val_ld, class_weights


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_one_epoch(model, loader, opt, crit, scaler, device, config, warmup_scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_n = 0
    
    for batch_idx, (mfcc, labels_info) in enumerate(loader):
        # Handle mixup format: labels_info can be (labels, None, 1.0) or (labels_a, labels_b, lam)
        if isinstance(labels_info, tuple) and len(labels_info) == 3:
            labels, labels_b, lam = labels_info
            labels = labels.to(device, non_blocking=True)
            if labels_b is not None:
                labels_b = labels_b.to(device, non_blocking=True)
        else:
            labels = labels_info.to(device, non_blocking=True)
            labels_b, lam = None, 1.0
        
        mfcc = mfcc.to(device, non_blocking=True)
        
        opt.zero_grad(set_to_none=True)
        
        # Mixed precision training
        if scaler:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out = model(mfcc)
                
                # Handle mixup loss
                if labels_b is not None:
                    loss = lam * crit(out, labels) + (1 - lam) * crit(out, labels_b)
                else:
                    loss = crit(out, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.training.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            
            scaler.step(opt)
            scaler.update()
        else:
            out = model(mfcc)
            
            # Handle mixup loss
            if labels_b is not None:
                loss = lam * crit(out, labels) + (1 - lam) * crit(out, labels_b)
            else:
                loss = crit(out, labels)
            
            loss.backward()
            
            # Gradient clipping
            if config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            
            opt.step()
        
        # Warmup scheduler
        if warmup_scheduler:
            warmup_scheduler.step()
        
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    
    return total_loss / max(1, total_n)


@torch.no_grad()
def evaluate(model, loader, crit, device, config):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_logits, all_labels = [], []
    
    for mfcc, labels in loader:
        mfcc = mfcc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        out = model(mfcc)
        logits = out["logits"]
        loss = crit(out, labels)
        
        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(logits, labels, config.model.num_classes)
    avg_loss = total_loss / max(1, total_n)
    
    return avg_loss, metrics


def save_checkpoint(model, config, metrics, epoch, path, emotion_map=None, optimizer=None, scheduler=None, scaler=None):
    """Save model checkpoint with all metadata"""
    checkpoint = {
        "model": model.state_dict(),
        "config": config.to_dict(),
        "metrics": metrics,
        "epoch": epoch,
        "emotion_map": emotion_map,
        "timestamp": datetime.now().isoformat()
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    torch.save(checkpoint, path)


def main():
    args = parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        config = Config.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = Config()
    
    # Override with command-line arguments
    config.update_from_args(args)
    
    # Disable augmentation if requested
    if args.no_augmentation:
        config.augmentation.enabled = False
    
    # Create output directory
    os.makedirs(config.system.out_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.system.out_dir, "config.yaml")
    config.to_yaml(config_path)
    print(f"Saved configuration to: {config_path}")
    
    # Set seed for reproducibility
    set_seed(config.system.seed)
    
    # Get device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading datasets...")
    train_ld, val_ld, class_weights = make_loaders(config, device)
    print(f"Train batches: {len(train_ld)}, Val batches: {len(val_ld)}")
    print(f"Training samples: {len(train_ld.dataset)}, Val samples: {len(val_ld.dataset)}")
    print(f"Number of classes: {config.model.num_classes}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    crit = SERLoss(
        num_classes=config.model.num_classes,
        label_smoothing=config.training.label_smoothing,
        class_weights=class_weights
    )
    
    # Create optimizer
    opt = create_optimizer(model, config)
    
    # Create schedulers
    scheduler = create_scheduler(opt, config, len(train_ld))
    warmup_scheduler = create_warmup_scheduler(opt, config, len(train_ld)) if config.training.warmup_epochs > 0 else None
    
    # Create scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if (config.system.amp and device.type == "cuda") else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        mode='max'  # Maximize UAR
    )
    
    # Resume from checkpoint if provided
    start_epoch = 1
    best_uar = -1.0
    best_path = None
    
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and checkpoint['optimizer'] and isinstance(checkpoint['optimizer'], dict) and 'param_groups' in checkpoint['optimizer']:
            opt.load_state_dict(checkpoint['optimizer'])
        else:
            print("Warning: Optimizer state not found in checkpoint, starting fresh")
        if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler and 'scaler' in checkpoint and checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        best_uar = checkpoint.get('metrics', {}).get('uar', -1.0)
        print(f"Resumed from epoch {checkpoint.get('epoch', 1)}")
        print(f"Best UAR so far: {best_uar:.4f}")
        # Find best checkpoint path if exists
        if 'config' in checkpoint:
            best_path = os.path.join(config.system.out_dir, f"best_uar_{best_uar:.4f}.pt")
    
    # Load emotion map for checkpoint saving
    emotion_map_path = os.path.join(config.system.out_dir, "emotion_map.json")
    emotion_map = None
    if os.path.exists(emotion_map_path):
        with open(emotion_map_path, 'r') as f:
            emotion_map = json.load(f)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.training.epochs + 1):
        # Train
        tr_loss = train_one_epoch(
            model, train_ld, opt, crit, scaler, device, config,
            warmup_scheduler=warmup_scheduler if epoch <= config.training.warmup_epochs else None
        )
        
        # Evaluate
        va_loss, va_metrics = evaluate(model, val_ld, crit, device, config)
        
        acc, uar = va_metrics["acc"], va_metrics["uar"]
        
        # Update scheduler (after warmup)
        if epoch > config.training.warmup_epochs and scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(uar)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = opt.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch:03d}/{config.training.epochs} | "
              f"LR: {current_lr:.2e} | "
              f"Train Loss: {tr_loss:.4f} | "
              f"Val Loss: {va_loss:.4f} | "
              f"Acc: {acc:.4f} | "
              f"UAR: {uar:.4f}")
        
        # Save best model
        if uar > best_uar:
            best_uar = uar
            best_path = os.path.join(config.system.out_dir, f"best_uar_{best_uar:.4f}.pt")
            save_checkpoint(model, config, va_metrics, epoch, best_path, emotion_map, opt, scheduler, scaler)
            print(f"  → Saved best model: {best_path}")
        
        # Save periodic checkpoints
        if config.system.save_interval > 0 and epoch % config.system.save_interval == 0:
            periodic_path = os.path.join(config.system.out_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(model, config, va_metrics, epoch, periodic_path, emotion_map, opt, scheduler, scaler)
        
        # Early stopping check
        if early_stopping(uar):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best UAR: {best_uar:.4f}")
            break
    
    # Save final model
    final_path = os.path.join(config.system.out_dir, "last.pt")
    save_checkpoint(model, config, va_metrics, epoch, final_path, emotion_map, opt, scheduler, scaler)
    print(f"\nSaved final model: {final_path}")
    
    if best_path:
        print(f"Best checkpoint: {best_path} (UAR: {best_uar:.4f})")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()