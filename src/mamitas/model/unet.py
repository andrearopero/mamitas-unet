"""U-Net with a pretrained MobileNetV3-Large encoder (torchvision).

    input (B, C, H, W)
      |
    encoder  -> skips at 1/2, 1/4, 1/8, 1/16 and a bottleneck at 1/32
      |
    decoder  -> 4 blocks: upsample x2 + concat skip + 2 convs
      |
    upsample to full resolution + head (1x1 conv) -> logits (B, num_classes, H, W)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from mamitas.schemas import ModelConfig

# Slices of mobilenet_v3_large().features, each ending at a new resolution:
#   [0:2]  16 ch @1/2   [2:4]  24 ch @1/4   [4:7]  40 ch @1/8
#   [7:13] 112 ch @1/16 [13:17] 960 ch @1/32
_STAGES = [(0, 2), (2, 4), (4, 7), (7, 13), (13, 17)]
_STAGE_CHANNELS = [16, 24, 40, 112, 960]


def adapt_first_conv(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    """Make the stem conv accept `in_channels` (1 = gray).

    For gray input the RGB filters are summed over the channel axis, which keeps
    the pretrained knowledge: a gray image is R = G = B, so the response is the same.
    """
    if conv.in_channels == in_channels:
        return conv
    new = nn.Conv2d(
        in_channels, conv.out_channels, conv.kernel_size, conv.stride,
        conv.padding, bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(conv.weight.sum(1, keepdim=True))
    return new


class Encoder(nn.Module):
    """MobileNetV3-Large split into 5 stages; forward returns every stage output."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v3_large(weights=weights).features
        features[0][0] = adapt_first_conv(features[0][0], in_channels, pretrained)
        self.stages = nn.ModuleList(nn.Sequential(*features[a:b]) for a, b in _STAGES)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    """Upsample to the skip's size, concatenate it, then two 3x3 convs."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.convs = nn.Sequential(
            _conv_bn_relu(in_channels + skip_channels, out_channels),
            _conv_bn_relu(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Using the skip's size (not scale_factor=2) also works for sizes not divisible by 32.
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convs(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        decoder_blocks: tuple[int, ...] | list[int] = (256, 128, 64, 32),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, pretrained)
        # Decoder walks back up: bottleneck (960) + skip 112, then 40, 24, 16.
        in_chs = [_STAGE_CHANNELS[-1], *decoder_blocks[:-1]]
        skip_chs = _STAGE_CHANNELS[-2::-1]
        self.decoder = nn.ModuleList(
            DecoderBlock(i, s, o) for i, s, o in zip(in_chs, skip_chs, decoder_blocks)
        )
        self.head = nn.Conv2d(decoder_blocks[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        *skips, x = self.encoder(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.head(x)  # raw logits; softmax is applied by the loss / predict


def build_model(cfg: ModelConfig) -> UNet:
    return UNet(cfg.in_channels, cfg.num_classes, cfg.decoder_blocks, cfg.pretrained)
