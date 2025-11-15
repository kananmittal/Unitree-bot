"""
Improved Tier 1 Model with:
- Residual connections
- Multi-head attention
- Deeper architecture
- Better temporal modeling
"""
import torch
from torch import nn
from typing import Tuple, Optional

# -------------------------
# Multi-Head Attention for MFCC channels
# -------------------------
class MultiHeadChannelAttention(nn.Module):
    """Multi-head attention over MFCC channels"""
    
    def __init__(self, channels: int, num_heads: int = 4, reduction: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(4, channels // (num_heads * reduction))
        
        # Multiple attention heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, self.head_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(self.head_dim, channels, bias=False),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])
        
        # Combine heads
        self.combine = nn.Linear(channels * num_heads, channels, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        
        # Average and max pooling
        avg = x.mean(dim=2)  # (B, C)
        mx, _ = x.max(dim=2)  # (B, C)
        
        # Apply each attention head
        head_outputs = []
        for head in self.heads:
            w_avg = head(avg)
            w_max = head(mx)
            w = 0.5 * (w_avg + w_max)  # (B, C)
            head_outputs.append(w)
        
        # Combine heads
        combined = torch.cat(head_outputs, dim=1)  # (B, C * num_heads)
        final_weights = torch.sigmoid(self.combine(combined))  # (B, C)
        
        return x * final_weights.unsqueeze(2)  # (B, C, T)


# -------------------------
# Residual Conv Block
# -------------------------
class ResidualConv1DBlock(nn.Module):
    """Convolutional block with residual connection"""
    
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, 
                 depthwise_separable: bool = True, dropout: float = 0.2):
        super().__init__()
        p = k // 2
        
        if depthwise_separable:
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, bias=False),
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        
        # Residual connection
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.norm = nn.BatchNorm1d(out_ch)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv(x)
        out = self.norm(out + residual)  # Residual connection
        out = self.activation(out)
        return out


# -------------------------
# Attention-based Temporal Pooling
# -------------------------
class AttentionPooling(nn.Module):
    """Learnable attention-based temporal pooling"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        weights = self.attention(x)  # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # (B, H)
        return pooled


# -------------------------
# Temperature Scaling
# -------------------------
class TempScale(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor(init_T)))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T).clamp(min=1e-3)
        return logits / T


# -------------------------
# Improved SER Tier-1 Model
# -------------------------
class SER_Tier1_Improved(nn.Module):
    """
    Improved SER model with:
    - Multi-head attention
    - Residual connections
    - Deeper architecture
    - Attention-based pooling
    """
    
    def __init__(
        self,
        n_mfcc: int = 40,
        num_classes: int = 4,
        conv_channels: Tuple[int, ...] = (128, 256),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        use_depthwise: bool = True,
        use_temp_scale: bool = True,
        use_residual: bool = True,
        attention_heads: int = 4,
    ):
        super().__init__()
        
        # Multi-head channel attention
        self.attn = MultiHeadChannelAttention(n_mfcc, num_heads=attention_heads)
        
        # CNN stack with optional residual connections
        convs = []
        in_ch = n_mfcc
        for ch in conv_channels:
            if use_residual:
                convs.append(ResidualConv1DBlock(in_ch, ch, k=5, 
                                                 depthwise_separable=use_depthwise, 
                                                 dropout=dropout))
            else:
                convs.append(Conv1DBlock(in_ch, ch, k=5, 
                                        depthwise_separable=use_depthwise, 
                                        dropout=dropout))
            in_ch = ch
        self.cnn = nn.Sequential(*convs)
        
        # Deeper BiLSTM
        self.rnn = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        
        # Attention-based temporal pooling
        self.temporal_pool = AttentionPooling(2 * lstm_hidden)
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(2 * lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )
        
        # Temperature scaling
        self.temp = TempScale() if use_temp_scale else nn.Identity()
    
    def forward(self, mfcc_CT: torch.Tensor):
        # mfcc_CT: (B, C, T)
        x = self.attn(mfcc_CT)           # (B, C, T) - Multi-head attention
        x = self.cnn(x)                  # (B, F, T) - Residual CNN
        x = x.transpose(1, 2)            # (B, T, F)
        y, _ = self.rnn(x)               # (B, T, 2H) - Deeper BiLSTM
        y = self.temporal_pool(y)        # (B, 2H) - Attention pooling
        logits = self.cls_head(y)        # (B, K)
        logits = self.temp(logits)
        
        probs = torch.softmax(logits, dim=-1)
        confidence, _ = probs.max(dim=-1)
        
        return {"logits": logits, "probs": probs, "confidence": confidence}


# -------------------------
# Conv Block (for backward compatibility)
# -------------------------
class Conv1DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, 
                 depthwise_separable: bool = True, dropout: float = 0.2):
        super().__init__()
        p = k // 2
        
        if depthwise_separable:
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, bias=False),
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Loss with Label Smoothing
# -------------------------
class SERLoss(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 label_smoothing: float = 0.1,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        
        if label_smoothing > 0:
            self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            self.ce = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, out: dict, labels: torch.Tensor):
        return self.ce(out["logits"], labels)


# Factory function to create model based on config
def create_model(config):
    """Create model from configuration"""
    return SER_Tier1_Improved(
        n_mfcc=config.model.n_mfcc,
        num_classes=config.model.num_classes,
        conv_channels=config.model.conv_channels,
        lstm_hidden=config.model.lstm_hidden,
        lstm_layers=config.model.lstm_layers,
        dropout=config.model.dropout,
        use_depthwise=config.model.use_depthwise,
        use_temp_scale=config.model.use_temp_scale,
        use_residual=config.model.use_residual,
        attention_heads=config.model.attention_heads,
    )


# -------------------------
# Utility for loading models
# -------------------------
def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint with proper parameter detection"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model" in ckpt:
        state_dict = ckpt["model"]
        config_dict = ckpt.get("config", {})
    else:
        state_dict = ckpt
        config_dict = {}
    
    # Infer model parameters from state dict
    num_classes = None
    n_mfcc = 40
    
    # Detect num_classes from classification head
    for key in ["cls_head.5.weight", "cls_head.2.weight", "head.1.weight"]:
        if key in state_dict:
            num_classes = state_dict[key].shape[0]
            break
    
    # Detect n_mfcc from attention or first conv layer
    if "attn.heads.0.0.weight" in state_dict:
        n_mfcc = state_dict["attn.heads.0.0.weight"].shape[1]
    elif "attn.combine.weight" in state_dict:
        # Multi-head: combined output is n_mfcc
        n_mfcc = state_dict["attn.combine.weight"].shape[0]
    elif "cnn.0.conv.0.weight" in state_dict:
        weight = state_dict["cnn.0.conv.0.weight"]
        n_mfcc = weight.shape[0] if weight.shape[1] == 1 else weight.shape[1]
    
    # Get other parameters from config or use defaults
    conv_channels = config_dict.get("conv_channels", (128, 256))
    lstm_hidden = config_dict.get("lstm_hidden", 128)
    lstm_layers = config_dict.get("lstm_layers", 2)
    
    if num_classes is None:
        raise ValueError("Could not infer num_classes from checkpoint")
    
    # Create model
    model = SER_Tier1_Improved(
        n_mfcc=n_mfcc,
        num_classes=num_classes,
        conv_channels=conv_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, num_classes, config_dict


if __name__ == "__main__":
    # Test model
    B, C, T, K = 4, 40, 300, 4
    x = torch.randn(B, C, T)
    
    model = SER_Tier1_Improved(n_mfcc=C, num_classes=K)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Probs shape: {out['probs'].shape}")
    print(f"Confidence shape: {out['confidence'].shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")