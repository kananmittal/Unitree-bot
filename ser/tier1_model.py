# tier1_model.py

# Tier 1 = MFCC -> TsPCA -> CNN stack -> BiLSTM -> Head (logits/probs/confidence)

import torch

from torch import nn

from typing import Tuple, Optional

# -------------------------

# TsPCA (Two-Stream Pooling Channel Attention)

# -------------------------

class TsPCA(nn.Module):

    def __init__(self, channels: int, reduction: int = 8):

        super().__init__()

        hidden = max(4, channels // reduction)

        self.mlp = nn.Sequential(

            nn.Linear(channels, hidden, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(hidden, channels, bias=False),

            nn.Sigmoid()

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (B, C, T)

        avg = x.mean(dim=2)          # (B, C)

        mx, _ = x.max(dim=2)         # (B, C)

        w = 0.5 * (self.mlp(avg) + self.mlp(mx))  # (B, C)

        return x * w.unsqueeze(2)    # (B, C, T)

# -------------------------

# Conv Block (depthwise-separable optional)

# -------------------------

class Conv1DBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k: int = 5, p: Optional[int] = None,

                 depthwise_separable: bool = True, dropout: float = 0.1):

        super().__init__()

        p = p if p is not None else k // 2

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

# Temperature scaling (confidence calibration)

# -------------------------

class TempScale(nn.Module):

    def __init__(self, init_T: float = 1.0):

        super().__init__()

        self.log_T = nn.Parameter(torch.log(torch.tensor(init_T)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:

        T = torch.exp(self.log_T).clamp(min=1e-3)

        return logits / T

# -------------------------

# SER Tier-1 Model

# -------------------------

class SER_Tier1(nn.Module):

    """

    Input:  MFCC (B, C, T)

    Stack:  TsPCA -> Conv x N -> BiLSTM -> Head

    Output: dict(logits, probs, confidence, va=None)  # if multitask enabled

    """

    def __init__(

        self,

        n_mfcc: int = 40,

        num_classes: int = 4,

        conv_channels: Tuple[int, ...] = (128, 128),

        lstm_hidden: int = 128,

        dropout: float = 0.1,

        use_depthwise: bool = True,

        use_temp_scale: bool = True,

        multitask_va: bool = False,           # valence-arousal regression (optional)

    ):

        super().__init__()

        self.attn = TsPCA(n_mfcc, reduction=8)

        # CNN stack

        convs = []

        in_ch = n_mfcc

        for ch in conv_channels:

            convs.append(Conv1DBlock(in_ch, ch, k=5, depthwise_separable=use_depthwise, dropout=dropout))

            in_ch = ch

        self.cnn = nn.Sequential(*convs)

        # BiLSTM

        self.rnn = nn.LSTM(

            input_size=in_ch,

            hidden_size=lstm_hidden,

            num_layers=1,

            batch_first=True,

            bidirectional=True,

            dropout=0.0,

        )

        # Heads

        self.cls_head = nn.Sequential(

            nn.LayerNorm(2 * lstm_hidden),

            nn.Dropout(dropout),

            nn.Linear(2 * lstm_hidden, num_classes),

        )

        self.temp = TempScale() if use_temp_scale else nn.Identity()

        self.multitask_va = multitask_va

        if multitask_va:

            self.va_head = nn.Sequential(

                nn.LayerNorm(2 * lstm_hidden),

                nn.Dropout(dropout),

                nn.Linear(2 * lstm_hidden, 2),   # valence, arousal

            )

    def forward(self, mfcc_CT: torch.Tensor):

        # mfcc_CT: (B, C, T)

        x = self.attn(mfcc_CT)           # (B, C, T)

        x = self.cnn(x)                  # (B, F, T)

        x = x.transpose(1, 2)            # (B, T, F)

        y, _ = self.rnn(x)               # (B, T, 2H)

        y = y.mean(dim=1)                # (B, 2H)

        logits = self.cls_head(y)        # (B, K)

        logits = self.temp(logits)

        probs = torch.softmax(logits, dim=-1)

        confidence, _ = probs.max(dim=-1)  # (B,)

        out = {"logits": logits, "probs": probs, "confidence": confidence}

        if self.multitask_va:

            va = self.va_head(y)         # (B, 2) in [-inf, inf]; apply tanh outside if needed

            out["va"] = va

        return out

# -------------------------

# Loss helpers (classification + optional VA regression)

# -------------------------

class SERLoss(nn.Module):

    def __init__(self, ce_weight: Optional[torch.Tensor] = None, va_lambda: float = 0.0):

        super().__init__()

        self.ce = nn.CrossEntropyLoss(weight=ce_weight)

        self.va_lambda = va_lambda

        self.l1 = nn.SmoothL1Loss(reduction="mean")

    def forward(self, out: dict, labels: torch.Tensor, va_targets: Optional[torch.Tensor] = None):

        loss = self.ce(out["logits"], labels)

        if ("va" in out) and (va_targets is not None) and (self.va_lambda > 0):

            loss = loss + self.va_lambda * self.l1(out["va"], va_targets)

        return loss

# -------------------------

# Example (remove in production)

# -------------------------

if __name__ == "__main__":

    B, C, T, K = 4, 40, 300, 4

    x = torch.randn(B, C, T)

    y = torch.randint(0, K, (B,))

    model = SER_Tier1(n_mfcc=C, num_classes=K, conv_channels=(128, 128), multitask_va=True)

    out = model(x)

    crit = SERLoss(va_lambda=0.1)

    va_targets = torch.randn(B, 2).tanh()  # example range [-1,1]

    loss = crit(out, y, va_targets)

    loss.backward()

    print(out["logits"].shape, out["probs"].shape, out["confidence"].shape, out["va"].shape, float(loss.item()))
