"""Recall = TP / (TP + FN): how much of the real foot was found."""

from mamitas.metrics._counts import EPS, confusion_counts


def recall_score(pred, target, foot: int = 1) -> float:
    tp, _, fn = confusion_counts(pred, target, foot)
    return (tp + EPS) / (tp + fn + EPS)
