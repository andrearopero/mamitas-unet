import pytest
from pydantic import ValidationError

from mamitas.schemas import Config, load_config


def _base():
    return {"data": {"dataset_dir": "/tmp/x"}}


def test_defaults_are_filled():
    cfg = Config.model_validate(_base())
    assert cfg.model.num_classes == 2
    assert cfg.model.in_channels == 3
    assert cfg.model.decoder_blocks == [256, 128, 64, 64]
    assert cfg.loss.name == "ce"


def test_in_channels_must_be_1_or_3():
    raw = _base() | {"model": {"in_channels": 2}}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_decoder_blocks_must_have_four_entries():
    raw = _base() | {"model": {"decoder_blocks": [64, 32]}}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_unknown_loss_is_rejected():
    raw = _base() | {"loss": {"name": "mse"}}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_unknown_key_is_rejected():
    raw = _base() | {"train": {"batchsize": 4}}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_image_size_must_be_multiple_of_32():
    raw = {"data": {"dataset_dir": "/tmp/x", "image_size": 250}}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  dataset_dir: /tmp/x\ntrain:\n  epochs: 3\n")
    assert load_config(p).train.epochs == 3


def test_paper_training_defaults():
    # Iturriago-Salas et al., Computers 2025, 14, 466, Section 4.
    cfg = Config.model_validate(_base())
    t = cfg.train
    assert (t.batch_size, t.epochs) == (36, 60)
    assert (t.lr, t.encoder_lr, t.lr_decay) == (1e-4, 1e-5, 0.94)
    assert t.freeze_encoder_epochs == 0
    assert cfg.loss.focal_gamma == 2.0 and cfg.loss.focal_alpha == 0.75
