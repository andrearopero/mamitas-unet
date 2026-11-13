import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mamitas.engines.predict import predict
from mamitas.engines.train import train
from mamitas.schemas import Config

ROOT = Path(__file__).resolve().parents[1]


def _cfg(dataset_dir, out, **train_kw):
    return Config.model_validate(
        {
            "data": {"dataset_dir": str(dataset_dir), "image_size": 64, "augment": False, "num_workers": 0},
            "model": {"decoder_blocks": [32, 16, 16, 8], "pretrained": False},
            "train": {"batch_size": 2, "epochs": 2, "device": "cpu", "output_dir": str(out)} | train_kw,
        }
    )


def test_train_writes_checkpoints_and_history(dataset_dir, tmp_path):
    out = tmp_path / "run"
    history = train(_cfg(dataset_dir, out))
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    assert len(history) == 2
    assert {"epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_recall", "val_precision"} <= history[0].keys()
    assert json.loads((out / "history.json").read_text()) == history


def test_training_reduces_loss_on_tiny_data(dataset_dir, tmp_path):
    history = train(_cfg(dataset_dir, tmp_path / "run", epochs=10, lr=3e-3))
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_best_checkpoint_is_self_describing(dataset_dir, tmp_path):
    out = tmp_path / "run"
    train(_cfg(dataset_dir, out))
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert {"model", "config", "epoch", "val_dice"} <= ckpt.keys()


def test_predict_saves_masks_and_metrics(dataset_dir, tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(dataset_dir, out)
    train(cfg)
    metrics = predict(cfg)
    masks = sorted((out / "predictions").glob("*.png"))
    assert len(masks) == 2  # Test split of the fixture
    assert set(metrics) == {"iou", "dice", "recall", "precision"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert (out / "predictions" / "metrics.json").exists()


def test_predict_without_checkpoint_fails_clearly(dataset_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        predict(_cfg(dataset_dir, tmp_path / "nothing"))


def _run(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_scripts_train_then_predict_with_test_yaml(tmp_path):
    out = str(tmp_path / "run")
    r = _run("train.py", "--config", "configs/test.yaml", "--output-dir", out)
    assert r.returncode == 0, r.stderr
    r = _run("predict.py", "--config", "configs/test.yaml", "--output-dir", out)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "run" / "predictions" / "metrics.json").exists()


def test_script_rejects_invalid_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  dataset_dir: x\n  image_size: 250\n")
    r = _run("train.py", "--config", str(bad))
    assert r.returncode != 0 and "multiple of 32" in r.stderr
