"""Build DataLoaders from a validated Config."""

from __future__ import annotations

from torch.utils.data import DataLoader

from mamitas.data.augmentations import build_train_transforms
from mamitas.data.dataset import FootDataset
from mamitas.schemas import Config


def make_loader(cfg: Config, split: str, train: bool = False) -> DataLoader:
    """`train=True` shuffles and (if cfg.data.augment) augments; otherwise deterministic."""
    transform = build_train_transforms() if (train and cfg.data.augment) else None
    ds = FootDataset(
        cfg.data.dataset_dir,
        split,
        in_channels=cfg.model.in_channels,
        image_size=cfg.data.image_size,
        transform=transform,
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=train,
        num_workers=cfg.data.num_workers,
        drop_last=False,
    )
