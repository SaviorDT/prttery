"""A basic UNet model for background removal (binary segmentation).

Input:  320 (W) x 180 (H) x 3 (RGB), normalized to [0, 1]
Output: 320 (W) x 180 (H) binary mask (probability of "foreground")
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.base import ModelBase


class _ConvBlock(nn.Module):
    """Two Conv-BatchNorm-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UNetNetwork(nn.Module):
    """A small 3-level encoder/decoder UNet with skip connections.

    ``180`` is not a power of two, so plain transposed-conv upsampling can
    produce a feature map whose spatial size does not exactly match the
    corresponding encoder output. To keep this robust regardless of the
    input's exact height/width, every decoder stage resizes (via
    ``F.interpolate``) to the exact size of its matching skip connection
    before concatenating.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()

        self.enc1 = _ConvBlock(in_channels, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.enc3 = _ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

        self.bottleneck = _ConvBlock(128, 256)

        self.dec3 = _ConvBlock(256 + 128, 128)
        self.dec2 = _ConvBlock(128 + 64, 64)
        self.dec1 = _ConvBlock(64 + 32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        return self.out_act(out)


class UNet(ModelBase):
    """UNet model for 320x180 background removal."""

    input_shape = (180, 320, 3)  # (H, W, C)
    output_shape = (180, 320)  # (H, W)

    def _build_network(self) -> nn.Module:
        return _UNetNetwork(in_channels=self.input_shape[2], out_channels=1)
