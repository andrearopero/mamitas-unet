import pytest
import torch
from torch import nn

from mamitas.model.unet import UNet, adapt_first_conv, build_model
from mamitas.schemas import ModelConfig


@pytest.mark.parametrize("in_ch", [1, 3])
@pytest.mark.parametrize("num_classes", [2, 3])
def test_output_shape(in_ch, num_classes):
    m = UNet(in_channels=in_ch, num_classes=num_classes, decoder_blocks=[32, 16, 16, 8], pretrained=False)
    out = m(torch.randn(2, in_ch, 64, 64))
    assert out.shape == (2, num_classes, 64, 64)


def test_decoder_blocks_set_decoder_channels():
    m = UNet(decoder_blocks=[64, 32, 16, 8], pretrained=False)
    assert [b.out_channels for b in m.decoder] == [64, 32, 16, 8]


def test_gradients_reach_the_encoder():
    m = UNet(decoder_blocks=[32, 16, 16, 8], pretrained=False)
    m(torch.randn(2, 3, 64, 64)).sum().backward()
    first = next(m.encoder.parameters())
    assert first.grad is not None and first.grad.abs().sum() > 0


def test_non_multiple_of_32_still_works():
    m = UNet(decoder_blocks=[32, 16, 16, 8], pretrained=False).eval()
    assert m(torch.randn(1, 3, 100, 90)).shape == (1, 2, 100, 90)


def test_adapt_first_conv_sums_rgb_weights_for_gray():
    conv = nn.Conv2d(3, 16, 3, bias=False)
    new = adapt_first_conv(conv, in_channels=1, pretrained=True)
    assert new.in_channels == 1
    assert torch.allclose(new.weight, conv.weight.sum(1, keepdim=True))


def test_adapt_first_conv_keeps_conv_for_rgb():
    conv = nn.Conv2d(3, 16, 3, bias=False)
    assert adapt_first_conv(conv, in_channels=3, pretrained=True) is conv


def test_build_model_from_config():
    cfg = ModelConfig(in_channels=1, decoder_blocks=[32, 16, 16, 8], pretrained=False)
    assert build_model(cfg)(torch.randn(1, 1, 64, 64)).shape == (1, 2, 64, 64)
