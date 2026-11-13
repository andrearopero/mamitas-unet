"""Inference: run the best checkpoint over a split, save masks and report metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mamitas.data.dataloader import make_loader
from mamitas.engines.train import resolve_device
from mamitas.metrics import evaluate
from mamitas.model.unet import build_model
from mamitas.schemas import Config, ModelConfig


def predict(cfg: Config) -> dict[str, float]:
    """Predict cfg.predict.split; masks are saved as PNG (0 = background, 255 = foot)."""
    out_root = Path(cfg.train.output_dir)
    ckpt_path = Path(cfg.predict.checkpoint or out_root / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path} (run scripts/train.py first)")

    device = resolve_device(cfg.train.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Rebuild the architecture the checkpoint was trained with (no pretrained download needed).
    model_cfg = ModelConfig.model_validate(ckpt["config"]["model"] | {"pretrained": False})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = make_loader(cfg, cfg.predict.split, train=False)
    names = [img.stem for img, _ in loader.dataset.pairs]  # loader is not shuffled: same order
    out_dir = Path(cfg.predict.output_dir or out_root / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, targets, i = [], [], 0
    with torch.no_grad():
        for x, y in loader:
            labels = model(x.to(device)).argmax(1).cpu()
            for lab in labels:
                Image.fromarray((lab.numpy() * 255).astype(np.uint8)).save(out_dir / f"{names[i]}.png")
                i += 1
            preds.append(labels)
            targets.append(y)

    metrics = evaluate(torch.cat(preds), torch.cat(targets))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(" ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics
