"""Joint image+mask augmentations (torchvision.transforms.v2).

v2 transforms know that a `Mask` must get the same geometry as the `Image` but
never color changes, and that masks are resampled with nearest neighbour.
"""

from __future__ import annotations

from torchvision.transforms import v2


def build_train_transforms() -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )
