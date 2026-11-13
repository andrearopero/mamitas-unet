import numpy as np
import pytest
import torch
from PIL import Image

from make_fixture import make_dataset
from mamitas.data.augmentations import build_train_transforms
from mamitas.data.dataloader import make_loader
from mamitas.data.dataset import FootDataset
from mamitas.schemas import Config


def test_len_and_shapes(dataset_dir):
    ds = FootDataset(dataset_dir, "Train", in_channels=3, image_size=32)
    assert len(ds) == 6
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert y.shape == (32, 32) and y.dtype == torch.long


def test_mask_is_binary_labels(dataset_dir):
    ds = FootDataset(dataset_dir, "Train", image_size=64)
    _, y = ds[0]
    assert set(y.unique().tolist()) <= {0, 1} and y.sum() > 0


def test_gray_gives_one_channel(dataset_dir):
    x, _ = FootDataset(dataset_dir, "Train", in_channels=1, image_size=32)[0]
    assert x.shape == (1, 32, 32)


def test_split_name_is_case_insensitive(dataset_dir):
    assert len(FootDataset(dataset_dir, "train", image_size=32)) == 6


def test_masks_directly_under_masks_dir(tmp_path):
    root = make_dataset(tmp_path / "flat", mask_subdir="")
    assert len(FootDataset(root, "Valid", image_size=32)) == 2


def test_two_channel_mask_is_reduced_to_labels(dataset_dir):
    p = dataset_dir / "Train" / "masks" / "class_0" / "sample_0.png"
    m = np.array(Image.open(p)) > 0
    onehot = np.stack([~m, m], -1).astype(np.uint8) * 255
    Image.fromarray(np.concatenate([onehot, onehot[..., :1]], -1)).save(p)  # 3ch: [bg, fg, bg]
    _, y = FootDataset(dataset_dir, "Train", image_size=64)[0]
    assert set(y.unique().tolist()) <= {0, 1}


def test_missing_mask_raises(dataset_dir):
    (dataset_dir / "Train" / "masks" / "class_0" / "sample_0.png").unlink()
    with pytest.raises(FileNotFoundError):
        FootDataset(dataset_dir, "Train")


def test_augmentation_keeps_shapes_and_binary_mask(dataset_dir):
    ds = FootDataset(dataset_dir, "Train", image_size=64, transform=build_train_transforms())
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (3, 64, 64) and y.shape == (64, 64)
        assert set(y.unique().tolist()) <= {0, 1}


def test_dataloader_batches(dataset_dir):
    cfg = Config.model_validate(
        {"data": {"dataset_dir": str(dataset_dir), "image_size": 32, "num_workers": 0},
         "train": {"batch_size": 4}}
    )
    x, y = next(iter(make_loader(cfg, "Train", train=True)))
    assert x.shape == (4, 3, 32, 32) and y.shape == (4, 32, 32)


def test_mask_with_values_0_and_1_is_kept(dataset_dir):
    # Dataset v3 stores masks as {0, 1} instead of {0, 255}.
    p = dataset_dir / "Train" / "masks" / "class_0" / "sample_0.png"
    m = (np.array(Image.open(p)) > 0).astype(np.uint8)
    Image.fromarray(m).save(p)
    _, y = FootDataset(dataset_dir, "Train", image_size=64)[0]
    assert y.sum() == m.sum()
