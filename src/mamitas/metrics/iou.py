"""IoU (Jaccard) = TP / (TP + FP + FN)."""

from mamitas.metrics._counts import EPS, confusion_counts


def iou_score(pred, target, foot: int = 1) -> float:
    tp, fp, fn = confusion_counts(pred, target, foot)
    return (tp + EPS) / (tp + fp + fn + EPS)
