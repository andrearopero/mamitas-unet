"""Precision = TP / (TP + FP): how much of the predicted foot is really foot."""

from mamitas.metrics._counts import EPS, confusion_counts


def precision_score(pred, target, foot: int = 1) -> float:
    tp, fp, _ = confusion_counts(pred, target, foot)
    return (tp + EPS) / (tp + fp + EPS)
