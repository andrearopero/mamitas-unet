"""Compare several training runs (for example one per loss).

    python scripts/plot_results.py --dataset-dir <path> [--runs-dir runs] [--losses ce dice focal]

Reads <runs-dir>/<loss>/history.json and <runs-dir>/<loss>/predictions/, then
  - prints the Test metrics of every run as a table,
  - saves <runs-dir>/curves.png   (validation Dice and IoU per epoch),
  - saves <runs-dir>/examples.png (image, ground truth and one prediction per run).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # use this checkout's code

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from mamitas.data.dataset import find_child, resolve_dataset_dir  # noqa: E402

METRICS = ("iou", "dice", "recall", "precision")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, help="folder with Train/ Valid/ Test/")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--losses", nargs="+", default=["ce", "dice", "focal"], help="run folder names")
    p.add_argument("--sample", help="Test sample name to show, e.g. sample_5 (default: first)")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    missing = [k for k in args.losses if not (runs / k / "predictions" / "metrics.json").exists()]
    if missing:
        sys.exit(f"Missing results in '{runs}' for: {', '.join(missing)} (run train.py and predict.py first)")

    # 1) metrics table
    print(f"{'run':8}" + "".join(f"{m:>11}" for m in METRICS))
    for k in args.losses:
        m = json.loads((runs / k / "predictions" / "metrics.json").read_text())
        print(f"{k:8}" + "".join(f"{m[x]:>11.4f}" for x in METRICS))

    # 2) validation curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in args.losses:
        hist = json.loads((runs / k / "history.json").read_text())
        ax[0].plot([e["val_dice"] for e in hist], label=k)
        ax[1].plot([e["val_iou"] for e in hist], label=k)
    for a, title in zip(ax, ("validation Dice", "validation IoU")):
        a.set_title(title)
        a.set_xlabel("epoch")
        a.legend()
    fig.savefig(runs / "curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3) one Test sample: image, truth and each run's prediction
    test = find_child(resolve_dataset_dir(args.dataset_dir, "Test"), "Test")
    masks = find_child(test, "masks")
    masks = masks / "class_0" if (masks / "class_0").is_dir() else masks
    first = sorted((runs / args.losses[0] / "predictions").glob("*.png"))[0].stem
    name = args.sample or first
    image = next(f for f in find_child(test, "images").iterdir() if f.stem == name)
    panels = [("image", image), ("truth", masks / f"{name}.png")]
    panels += [(k, runs / k / "predictions" / f"{name}.png") for k in args.losses]
    fig, ax = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4))
    for a, (title, path) in zip(ax, panels):
        a.imshow(Image.open(path), cmap="gray")
        a.set_title(title)
        a.axis("off")
    fig.savefig(runs / "examples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {runs / 'curves.png'} and {runs / 'examples.png'} (sample: {name})")


if __name__ == "__main__":
    main()
