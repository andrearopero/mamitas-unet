"""Dice (F1) = 2TP / (2TP + FP + FN)."""

from mamitas.metrics._counts import EPS, confusion_counts


def dice_score(pred, target, foot: int = 1) -> float:
    tp, fp, fn = confusion_counts(pred, target, foot)
    return (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
