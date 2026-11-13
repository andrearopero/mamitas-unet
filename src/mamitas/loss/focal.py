"""Focal loss: cross-entropy scaled by (1 - p_t)^gamma.

Pixels the model already gets right (p_t near 1) contribute almost nothing, so
training concentrates on hard pixels such as foot borders. gamma = 0 gives plain CE.
"""

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)  # probability assigned to the true class
        return ((1 - pt) ** self.gamma * ce).mean()
