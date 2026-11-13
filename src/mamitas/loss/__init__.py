from torch import nn

from mamitas.loss.ce import CELoss
from mamitas.loss.dice import DiceLoss
from mamitas.loss.focal import FocalLoss
from mamitas.schemas import LossConfig


def build_loss(cfg: LossConfig) -> nn.Module:
    if cfg.name == "ce":
        return CELoss()
    if cfg.name == "dice":
        return DiceLoss()
    return FocalLoss(gamma=cfg.focal_gamma)


__all__ = ["CELoss", "DiceLoss", "FocalLoss", "build_loss"]
