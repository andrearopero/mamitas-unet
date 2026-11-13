"""Focal loss: cross-entropy scaled by (1 - p_t)^gamma, optionally class-weighted by alpha.

Pixels the model already gets right (p_t near 1) contribute almost nothing, so
training concentrates on hard pixels such as foot borders. gamma = 0 gives plain CE.
alpha weights the foot class (1) and 1 - alpha every other class; None disables it.
"""

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)  # probability assigned to the true class
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = torch.where(target == 1, self.alpha, 1 - self.alpha) * loss
        return loss.mean()
