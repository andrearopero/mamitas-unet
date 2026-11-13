"""Build a tiny synthetic foot dataset (ellipse on noise) used by tests and test.yaml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = {"Train": 6, "Valid": 2, "Test": 2}


def make_dataset(root: Path, size: int = 64, mask_subdir: str = "class_0") -> Path:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:size, :size]
    for split, n in SPLITS.items():
        img_dir = root / split / "images"
        msk_dir = root / split / "masks" / mask_subdir if mask_subdir else root / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cy, cx = rng.integers(size // 3, 2 * size // 3, 2)
            ry, rx = rng.integers(size // 6, size // 3, 2)
            mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1
            img = rng.integers(0, 80, (size, size, 3)).astype(np.uint8)
            img[mask] = rng.integers(150, 255, (int(mask.sum()), 3))
            Image.fromarray(img).save(img_dir / f"sample_{i}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(msk_dir / f"sample_{i}.png")
    return root


if __name__ == "__main__":
    make_dataset(Path(__file__).parent / "fixtures" / "mini_dataset")
