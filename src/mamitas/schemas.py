"""Pydantic models that validate the YAML configuration.

A typo in the YAML (wrong key, impossible value) fails at load time with a clear
message instead of crashing in the middle of a training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    # extra="forbid": unknown keys are errors, so "batchsize" cannot silently be ignored.
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    dataset_dir: str
    train_split: str = "Train"
    valid_split: str = "Valid"
    test_split: str = "Test"
    image_size: int = Field(256, gt=0)
    augment: bool = True
    num_workers: int = Field(2, ge=0)

    @field_validator("image_size")
    @classmethod
    def _multiple_of_32(cls, v: int) -> int:
        # MobileNetV3 halves the resolution 5 times (2**5 = 32).
        if v % 32 != 0:
            raise ValueError("image_size must be a multiple of 32")
        return v


class ModelConfig(_Strict):
    num_classes: int = Field(2, ge=2)
    in_channels: Literal[1, 3] = 3
    decoder_blocks: list[int] = [256, 128, 64, 32]
    pretrained: bool = True

    @field_validator("decoder_blocks")
    @classmethod
    def _four_blocks(cls, v: list[int]) -> list[int]:
        if len(v) != 4 or any(c <= 0 for c in v):
            raise ValueError("decoder_blocks needs 4 positive channel counts")
        return v


class LossConfig(_Strict):
    name: Literal["ce", "dice", "focal"] = "ce"
    focal_gamma: float = Field(2.0, ge=0)


class TrainConfig(_Strict):
    batch_size: int = Field(8, gt=0)
    epochs: int = Field(30, gt=0)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    seed: int = 42
    device: str = "auto"  # "auto" -> cuda if available, else cpu
    output_dir: str = "runs/default"


class PredictConfig(_Strict):
    split: str = "Test"
    checkpoint: str | None = None  # defaults to <train.output_dir>/best.pt
    output_dir: str | None = None  # defaults to <train.output_dir>/predictions


class Config(_Strict):
    data: DataConfig
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    predict: PredictConfig = PredictConfig()


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated Config."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
