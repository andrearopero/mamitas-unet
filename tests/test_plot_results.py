import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _fake_run(runs, name, dice):
    out = runs / name
    (out / "predictions").mkdir(parents=True)
    history = [{"epoch": e, "val_dice": dice * e / 3, "val_iou": dice * e / 4} for e in (1, 2, 3)]
    (out / "history.json").write_text(json.dumps(history))
    metrics = {"iou": dice - 0.04, "dice": dice, "recall": 0.9, "precision": 0.8}
    (out / "predictions" / "metrics.json").write_text(json.dumps(metrics))
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(out / "predictions" / "sample_0.png")


def test_plot_results_writes_figures_and_prints_table(dataset_dir, tmp_path):
    runs = tmp_path / "runs"
    _fake_run(runs, "ce", 0.9512)
    _fake_run(runs, "dice", 0.9333)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_results.py"), "--dataset-dir", str(dataset_dir),
         "--runs-dir", str(runs), "--losses", "ce", "dice"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (runs / "curves.png").stat().st_size > 0
    assert (runs / "examples.png").stat().st_size > 0
    assert "0.9512" in r.stdout and "0.9333" in r.stdout


def test_plot_results_reports_missing_run(dataset_dir, tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_results.py"), "--dataset-dir", str(dataset_dir),
         "--runs-dir", str(tmp_path / "runs"), "--losses", "ce"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0 and "runs" in r.stderr
