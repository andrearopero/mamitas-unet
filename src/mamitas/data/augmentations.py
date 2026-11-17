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


class RandomGaussianNoise:
    """Add N(0, std) noise (in 0-255 intensity units) to the image with probability p.

    A plain callable on (image, mask) instead of a `v2.Transform` subclass: the hooks of
    that base class were renamed between torchvision releases, and Compose only needs a
    callable, so this works on whatever torchvision the platform ships.
    """

    def __init__(self, p: float = 0.2, std: float = 8.0) -> None:
        self.p, self.std = p, std

    def __call__(self, image: Any, mask: Any) -> tuple[Any, Any]:
        if torch.rand(1).item() >= self.p:
            return image, mask
        noisy = image.float() + torch.randn(image.shape) * self.std
        return tv_tensors.Image(noisy.clamp(0, 255).round().to(image.dtype)), mask


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
