import torch

@torch.no_grad()
def compute_metrics(logits, labels, num_classes):
    """
    logits: (N,K), labels: (N,)
    Returns: dict(acc, uar, precision, recall, f1)
    """
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

    # Build confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    
    # UAR (macro recall) - unweighted average recall
    recall_per_class = []
    precision_per_class = []
    f1_per_class = []
    
    for k in range(num_classes):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    uar = sum(recall_per_class) / max(1, num_classes)
    macro_precision = sum(precision_per_class) / max(1, num_classes)
    macro_f1 = sum(f1_per_class) / max(1, num_classes)
    
    return {
        "acc": acc,
        "uar": uar,
        "precision": macro_precision,
        "recall": uar,  # UAR is macro recall
        "f1": macro_f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class
    }
