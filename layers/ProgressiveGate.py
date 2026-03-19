import torch
import torch.nn as nn


class ProgressiveGate(nn.Module):
    """
    Simple progressive gate:
    input x and residual are [B, N, D]
    returns fused output and gate values g [B, N, 1]
    """
    def __init__(self, d_model):
        super().__init__()
        mid = max(1, d_model // 4)
        self.gate = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        # x, residual: [B, N, D]
        g = self.gate(x)   # [B, N, 1]
        out = g * x + (1.0 - g) * residual
        return out, g


