"""Joint image+mask augmentations (torchvision.transforms.v2).

Follows the paper: random horizontal/vertical flips and rotations within +-15 degrees
(applied to image and mask together), brightness/contrast/saturation within +-20% and
low-probability Gaussian noise (image only). v2 knows a `Mask` must get the same
geometry as the `Image`, never color changes, and is resampled with nearest neighbour.
"""

from __future__ import annotations

from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


class RandomGaussianNoise(v2.Transform):
    """Add N(0, std) noise (in 0-255 intensity units) to the image with probability p."""

    def __init__(self, p: float = 0.2, std: float = 8.0) -> None:
        super().__init__()
        self.p, self.std = p, std

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"apply": bool(torch.rand(1).item() < self.p)}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params["apply"] or not isinstance(inpt, tv_tensors.Image):
            return inpt
        noisy = inpt.float() + torch.randn_like(inpt, dtype=torch.float32) * self.std
        return tv_tensors.Image(noisy.clamp(0, 255).round().to(inpt.dtype))


def build_train_transforms() -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            RandomGaussianNoise(p=0.2),
        ]
    )
