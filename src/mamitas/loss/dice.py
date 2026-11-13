"""Soft Dice loss: 1 - 2|P∩T| / (|P| + |T|), averaged over classes.

Uses the softmax probabilities (not hard labels) so it stays differentiable.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(probs.dtype)
        dims = (0, 2, 3)  # sum over batch and pixels, keep one value per class
        inter = (probs * onehot).sum(dims)
        total = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()
