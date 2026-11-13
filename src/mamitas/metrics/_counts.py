"""Shared TP / FP / FN counting for the foot class."""

import torch

EPS = 1e-7


def confusion_counts(pred: torch.Tensor, target: torch.Tensor, foot: int = 1):
    """Counts over the whole batch. `pred` and `target` are (B, H, W) label maps."""
    p, t = pred == foot, target == foot
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    return tp, fp, fn
