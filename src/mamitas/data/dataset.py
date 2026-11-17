"""Dataset for binary foot segmentation.

Expected layout (split folder names are matched case-insensitively):

    <root>/<split>/images/sample#.png|jpg
    <root>/<split>/masks/sample#.png            (or masks/class_0/sample#.png)

An image and its mask share the same file name (stem).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
KAGGLE_INPUT = "/kaggle/input"
_announced: set[str] = set()
# ImageNet statistics: the MobileNetV3 encoder was pretrained with them.
_RGB_MEAN, _RGB_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_GRAY_MEAN, _GRAY_STD = (0.449,), (0.226,)


def find_child(parent: Path, name: str) -> Path:
    """Return the child directory of `parent` whose name equals `name` ignoring case."""
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    raise FileNotFoundError(f"No '{name}' folder inside {parent}")


def _has_split(directory: Path, split: str) -> bool:
    try:
        find_child(find_child(directory, split), "images")
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _subdirs(base: Path, max_depth: int):
    """Yield `base` and its sub-directories down to `max_depth` levels, breadth first."""
    level = [base]
    for _ in range(max_depth + 1):
        nxt: list[Path] = []
        for directory in level:
            yield directory
            try:
                nxt.extend(sorted(c for c in directory.iterdir() if c.is_dir()))
            except OSError:
                continue
        level = nxt


def resolve_dataset_dir(root: str | Path, split: str = "Train", max_depth: int = 3) -> Path:
    """Return the folder that contains `<split>/images`.

    Looks in `root` first (itself, then its sub-folders) and, if that fails, under
    /kaggle/input. Kaggle mounts a dataset at different paths depending on the platform
    version (/kaggle/input/<name>, /kaggle/input/datasets/<owner>/<name>, ...), so the path
    written in the YAML may not exist there.
    """
    root = Path(root)
    for base in (b for b in (root, Path(KAGGLE_INPUT)) if b.is_dir()):
        for directory in _subdirs(base, max_depth):
            if _has_split(directory, split):
                if directory != root and str(root) not in _announced:
                    _announced.add(str(root))
                    print(f"[mamitas] dataset_dir '{root}' has no '{split}/images'; using '{directory}'")
                return directory
    raise FileNotFoundError(
        f"Could not find '{split}/images' under '{root}' or under {KAGGLE_INPUT}. "
        "Check data.dataset_dir in the YAML (or pass --dataset-dir)."
    )


def _mask_to_labels(arr: np.ndarray) -> np.ndarray:
    """Turn any supported mask array into an HxW uint8 map with values {0, 1}."""
    if arr.ndim == 3:
        # HxWx2 one-hot (classes 0 and 1) -> argmax; otherwise use the first channel.
        arr = arr.argmax(-1) if arr.shape[-1] == 2 else arr[..., 0]
    return (arr > 127 if arr.max() > 1 else arr > 0).astype(np.uint8)


class FootDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        in_channels: int = 3,
        image_size: int = 256,
        transform=None,
    ) -> None:
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        self.in_channels = in_channels
        self.image_size = image_size
        self.transform = transform

        split_dir = find_child(resolve_dataset_dir(root, split), split)
        images_dir = find_child(split_dir, "images")
        masks_dir = find_child(split_dir, "masks")
        nested = masks_dir / "class_0"
        if nested.is_dir():
            masks_dir = nested

        self.pairs: list[tuple[Path, Path]] = []
        for img in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            mask = masks_dir / f"{img.stem}.png"
            if not mask.exists():
                raise FileNotFoundError(f"Mask not found for {img.name}: expected {mask}")
            self.pairs.append((img, mask))
        if not self.pairs:
            raise FileNotFoundError(f"No images found in {images_dir}")

        mean, std = (_RGB_MEAN, _RGB_STD) if in_channels == 3 else (_GRAY_MEAN, _GRAY_STD)
        self._mean = torch.tensor(mean).view(-1, 1, 1)
        self._std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        size = (self.image_size, self.image_size)

        img = Image.open(img_path).convert("RGB" if self.in_channels == 3 else "L")
        img = img.resize(size, Image.BILINEAR)
        mask = _mask_to_labels(np.array(Image.open(mask_path)))
        mask = np.array(Image.fromarray(mask).resize(size, Image.NEAREST))  # nearest keeps labels

        x = tv_tensors.Image(torch.from_numpy(np.array(img)).reshape(*size, -1).permute(2, 0, 1))
        y = tv_tensors.Mask(torch.from_numpy(mask))
        if self.transform is not None:
            x, y = self.transform(x, y)

        x = (torch.as_tensor(x).float() / 255.0 - self._mean) / self._std
        return x, torch.as_tensor(y).long()
