import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from mamitas.metrics.dice import dice_score
from mamitas.metrics.iou import iou_score
from mamitas.metrics.precision import precision_score as prec
from mamitas.metrics.recall import recall_score as rec


def _pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (3, 16, 16), generator=g), torch.randint(0, 2, (3, 16, 16), generator=g)


@pytest.mark.parametrize(
    "ours, ref",
    [(iou_score, jaccard_score), (dice_score, f1_score), (rec, recall_score), (prec, precision_score)],
)
def test_matches_sklearn_for_foot_class(ours, ref):
    pred, target = _pair()
    expected = ref(target.flatten().numpy(), pred.flatten().numpy(), pos_label=1)
    assert ours(pred, target) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("fn", [iou_score, dice_score, rec, prec])
def test_perfect_prediction_scores_one(fn):
    _, target = _pair()
    assert fn(target, target) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [iou_score, dice_score, rec, prec])
def test_total_miss_scores_zero(fn):
    _, target = _pair()
    assert fn(1 - target, target) == pytest.approx(0.0, abs=1e-6)


def test_precision_and_recall_differ_on_over_segmentation():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2] = 1
    pred = torch.ones(1, 4, 4, dtype=torch.long)  # predicts foot everywhere
    assert rec(pred, target) == pytest.approx(1.0)
    assert prec(pred, target) == pytest.approx(0.5)


def test_empty_prediction_and_target_is_defined():
    z = torch.zeros(1, 4, 4, dtype=torch.long)
    assert np.isfinite([iou_score(z, z), dice_score(z, z), rec(z, z), prec(z, z)]).all()
