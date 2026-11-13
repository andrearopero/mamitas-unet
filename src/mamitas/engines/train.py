"""Training loop: train one epoch, validate, keep the best checkpoint by validation Dice."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mamitas.data.dataloader import make_loader
from mamitas.loss import build_loss
from mamitas.metrics import evaluate
from mamitas.model.unet import build_model
from mamitas.schemas import Config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: Config):
    """Adam with a smaller lr for the pretrained encoder, plus per-epoch exponential decay."""
    encoder = list(model.encoder.parameters())
    ids = {id(p) for p in encoder}
    rest = [p for p in model.parameters() if id(p) not in ids]  # decoder + head
    optimizer = torch.optim.Adam(
        [{"params": encoder, "lr": cfg.train.encoder_lr}, {"params": rest, "lr": cfg.train.lr}],
        weight_decay=cfg.train.weight_decay,
    )
    return optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.train.lr_decay)


def set_encoder_trainable(model: nn.Module, epoch: int, every: int) -> None:
    """Progressive unfreezing (epoch is 0-based). every=0 keeps the whole encoder trainable.

    Otherwise the encoder starts frozen and one more stage opens every `every` epochs,
    from the deepest (high-level features) to the shallowest (low-level details).
    """
    stages = model.encoder.stages
    n_open = len(stages) if every == 0 else min(len(stages), epoch // every)
    for i, stage in enumerate(stages):
        trainable = i >= len(stages) - n_open
        for p in stage.parameters():
            p.requires_grad_(trainable)


def _train_epoch(model, loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def _validate(model, loader: DataLoader, criterion, device) -> dict[str, float]:
    model.eval()
    total, n = 0.0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    metrics = evaluate(torch.cat(preds), torch.cat(targets))
    return {"val_loss": total / n, **{f"val_{k}": v for k, v in metrics.items()}}


def train(cfg: Config) -> list[dict]:
    """Train and return the per-epoch history (also saved as history.json)."""
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(cfg, cfg.data.train_split, train=True)
    valid_loader = make_loader(cfg, cfg.data.valid_split, train=False)
    model: nn.Module = build_model(cfg.model).to(device)
    criterion = build_loss(cfg.loss)
    optimizer, scheduler = build_optimizer(model, cfg)

    history: list[dict] = []
    best_dice = -1.0
    for epoch in range(1, cfg.train.epochs + 1):
        set_encoder_trainable(model, epoch - 1, cfg.train.freeze_encoder_epochs)
        row = {"epoch": epoch, "train_loss": _train_epoch(model, train_loader, criterion, optimizer, device)}
        scheduler.step()
        row |= _validate(model, valid_loader, criterion, device)
        history.append(row)

        ckpt = {"model": model.state_dict(), "config": cfg.model_dump(), "epoch": epoch, "val_dice": row["val_dice"]}
        torch.save(ckpt, out / "last.pt")
        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            torch.save(ckpt, out / "best.pt")
        print(
            f"epoch {epoch:03d}/{cfg.train.epochs} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f} val_iou={row['val_iou']:.4f}"
            + ("  *best*" if best_dice == row["val_dice"] else ""),
            flush=True,
        )

    (out / "history.json").write_text(json.dumps(history, indent=2))
    return history
