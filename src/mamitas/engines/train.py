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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    history: list[dict] = []
    best_dice = -1.0
    for epoch in range(1, cfg.train.epochs + 1):
        row = {"epoch": epoch, "train_loss": _train_epoch(model, train_loader, criterion, optimizer, device)}
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
