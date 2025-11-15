import os, argparse, torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from datasets import AudioEmotionDataset, pad_collate
from tier1_model import SER_Tier1, SERLoss
from metrics import compute_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, default=None)  # if None, will split train
    ap.add_argument("--cache_dir", type=str, default="feat_cache")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--num_classes", type=int, default=None, help="Number of classes (auto-detected from dataset if not provided)")
    ap.add_argument("--n_mfcc", type=int, default=40)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return ap.parse_args()

def make_loaders(args, device):
    # Resolve CSV paths - handle both absolute and relative paths
    # If relative, assume it's relative to project root (parent of ser/)
    if not os.path.isabs(args.train_csv):
        # Try relative to current working directory first
        if not os.path.exists(args.train_csv):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            train_csv = os.path.join(project_root, args.train_csv)
        else:
            train_csv = args.train_csv
    else:
        train_csv = args.train_csv
    
    if args.val_csv is not None:
        if not os.path.isabs(args.val_csv):
            if not os.path.exists(args.val_csv):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                val_csv = os.path.join(project_root, args.val_csv)
            else:
                val_csv = args.val_csv
        else:
            val_csv = args.val_csv
    else:
        val_csv = None
    
    # Create unified emotion mapping from both datasets to ensure consistent labels
    from datasets import get_emotion_map_from_dataset
    import pandas as pd
    
    unified_emotion_map = None
    if args.val_csv is not None:
        # Read both CSVs to get all unique emotions
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        if 'emotion_code' in train_df.columns and 'emotion_code' in val_df.columns:
            # Get all unique emotions from both datasets
            all_emotions = sorted(set(train_df['emotion_code'].unique()) | set(val_df['emotion_code'].unique()))
            # Create unified mapping
            unified_emotion_map = {emotion: idx for idx, emotion in enumerate(all_emotions)}
            print(f"Created unified emotion mapping with {len(unified_emotion_map)} classes: {unified_emotion_map}")
    
    if args.val_csv is None:
        full = AudioEmotionDataset(train_csv, cache_dir=args.cache_dir, device=device, n_mfcc=args.n_mfcc, emotion_map=unified_emotion_map)
        n_total = len(full)
        n_val = int(args.val_split * n_total)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    else:
        train_ds = AudioEmotionDataset(train_csv, cache_dir=args.cache_dir, device=device, n_mfcc=args.n_mfcc, emotion_map=unified_emotion_map)
        val_ds   = AudioEmotionDataset(val_csv,   cache_dir=args.cache_dir, device=device, n_mfcc=args.n_mfcc, emotion_map=unified_emotion_map)

    pin_mem = device.type == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=pin_mem)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=pad_collate, pin_memory=pin_mem)
    return train_ld, val_ld

def train_one_epoch(model, loader, opt, crit, scaler, device, num_classes):
    model.train()
    total_loss, total_n = 0.0, 0
    for mfcc, y in loader:
        mfcc = mfcc.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out = model(mfcc)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(mfcc)
            loss = crit(out, y)
            loss.backward()
            opt.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(1, total_n)

@torch.no_grad()
def evaluate(model, loader, crit, device, num_classes):
    model.eval()
    total_loss, total_n = 0.0, 0
    all_logits, all_labels = [], []
    for mfcc, y in loader:
        mfcc = mfcc.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(mfcc)
        logits = out["logits"]
        loss = crit(out, y)
        total_loss += loss.item() * y.size(0)
        total_n += y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels, num_classes)
    return total_loss / max(1, total_n), metrics

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Adjust num_workers for macOS (multiprocessing issues)
    import platform
    if platform.system() == "Darwin" and args.num_workers > 0:
        args.num_workers = 0
        print("Note: Using num_workers=0 on macOS for compatibility")

    print(f"Using device: {device}")
    
    # Build loaders
    print("Loading datasets...")
    train_ld, val_ld = make_loaders(args, device)
    print(f"Train batches: {len(train_ld)}, Val batches: {len(val_ld)}")

    # Infer n_mfcc from a batch (in case cache has different C)
    mfcc_sample, _ = next(iter(train_ld))
    n_mfcc = mfcc_sample.size(1)  # (B,C,T) -> C
    if n_mfcc != args.n_mfcc:
        args.n_mfcc = n_mfcc
    
    # Auto-detect num_classes from dataset if not provided
    if args.num_classes is None:
        # Get number of classes from dataset's emotion map or labels
        train_ds = train_ld.dataset
        if hasattr(train_ds, 'emotion_map') and train_ds.emotion_map is not None:
            args.num_classes = len(train_ds.emotion_map)
        else:
            # Fallback: count unique labels from a sample batch
            _, label_sample = next(iter(train_ld))
            args.num_classes = int(torch.max(label_sample).item() + 1)
        print(f"Auto-detected {args.num_classes} classes from dataset")
    else:
        print(f"Using {args.num_classes} classes (from argument)")

    # Model/optim
    model = SER_Tier1(n_mfcc=args.n_mfcc, num_classes=args.num_classes).to(device)
    crit  = SERLoss()  # Use SERLoss instead of CrossEntropyLoss
    opt   = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    best_uar, best_path = -1.0, None
    start_epoch = 1

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            # Try to load optimizer state if available
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt and scaler is not None:
                scaler.load_state_dict(ckpt["scaler"])
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
                print(f"Resuming from epoch {start_epoch}")
            if "best_uar" in ckpt:
                best_uar = ckpt["best_uar"]
            if "best_path" in ckpt:
                best_path = ckpt["best_path"]
            print(f"Resumed: best_uar={best_uar:.4f}, starting from epoch {start_epoch}")
        else:
            print(f"Warning: Resume checkpoint not found: {args.resume}, starting from scratch")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_ld, opt, crit, scaler, device, args.num_classes)
        va_loss, va_metrics = evaluate(model, val_ld, crit, device, args.num_classes)
        acc, uar = va_metrics["acc"], va_metrics["uar"]
        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | acc {acc:.4f} | uar {uar:.4f}")

        # Save best by UAR
        if uar > best_uar:
            best_uar = uar
            best_path = os.path.join(args.out_dir, f"best_uar_{best_uar:.4f}.pt")
            save_dict = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_uar": best_uar,
                "best_path": best_path
            }
            if scaler is not None:
                save_dict["scaler"] = scaler.state_dict()
            torch.save(save_dict, best_path)
            print(f"Saved: {best_path}")

    # Final save
    final_path = os.path.join(args.out_dir, "last.pt")
    save_dict = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "args": vars(args),
        "epoch": args.epochs,
        "best_uar": best_uar,
        "best_path": best_path
    }
    if scaler is not None:
        save_dict["scaler"] = scaler.state_dict()
    torch.save(save_dict, final_path)
    print(f"Saved last: {final_path}")
    if best_path:
        print(f"Best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
