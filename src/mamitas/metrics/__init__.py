import torch

from mamitas.metrics.dice import dice_score
from mamitas.metrics.iou import iou_score
from mamitas.metrics.precision import precision_score
from mamitas.metrics.recall import recall_score


def evaluate(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """All four metrics for the foot class, from (B, H, W) label maps."""
    return {
        "iou": iou_score(pred, target),
        "dice": dice_score(pred, target),
        "recall": recall_score(pred, target),
        "precision": precision_score(pred, target),
    }


__all__ = ["dice_score", "iou_score", "precision_score", "recall_score", "evaluate"]
