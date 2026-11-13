import pytest
import torch
import torch.nn.functional as F

from mamitas.loss import build_loss
from mamitas.loss.ce import CELoss
from mamitas.loss.dice import DiceLoss
from mamitas.loss.focal import FocalLoss
from mamitas.schemas import LossConfig


def _perfect(target, c=2, scale=20.0):
    return F.one_hot(target, c).permute(0, 3, 1, 2).float() * scale


def _batch():
    torch.manual_seed(0)
    return torch.randn(2, 2, 8, 8, requires_grad=True), torch.randint(0, 2, (2, 8, 8))


def test_ce_matches_torch():
    x, y = _batch()
    assert torch.allclose(CELoss()(x, y), F.cross_entropy(x, y))


def test_dice_is_zero_for_perfect_prediction():
    y = torch.randint(0, 2, (2, 8, 8))
    assert DiceLoss()(_perfect(y), y) < 1e-3


def test_dice_is_high_for_inverted_prediction():
    y = torch.randint(0, 2, (2, 8, 8))
    assert DiceLoss()(_perfect(1 - y), y) > 0.9


def test_focal_with_gamma_zero_equals_ce():
    x, y = _batch()
    assert torch.allclose(FocalLoss(gamma=0.0)(x, y), F.cross_entropy(x, y), atol=1e-6)


def test_focal_downweights_easy_pixels():
    x, y = _batch()
    assert FocalLoss(gamma=2.0)(x, y) < F.cross_entropy(x, y)


@pytest.mark.parametrize("name", ["ce", "dice", "focal"])
def test_every_loss_is_scalar_and_differentiable(name):
    x, y = _batch()
    loss = build_loss(LossConfig(name=name))(x, y)
    assert loss.ndim == 0
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
