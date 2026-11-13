"""Cross-entropy: mean over pixels of -log(p[true class])."""

import torch
import torch.nn.functional as F
from torch import nn


class CELoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits (B, C, H, W) raw scores, target (B, H, W) class ids.
        return F.cross_entropy(logits, target)
