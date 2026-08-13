"""UNet with a pretrained ResNet18 encoder for 5-class pottery-part segmentation.

Same encoder/decoder architecture as ``models.unet_resnet18.UNetResNet18``,
copied independently rather than shared/inherited (a deliberate choice, so
the two can be tuned separately later without risking the proven binary
background-removal model) -- only the head differs: 5 output channels
(background, pottery, teapot, rotate_plate, rotate_medal) with a per-pixel
softmax instead of 1 channel with a sigmoid.

Input:  320 (W) x 180 (H) x 3 (RGB), normalized to [0, 1]
Output: 320 (W) x 180 (H) x 5 per-pixel class probabilities (sum to 1 per pixel)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from models.base import ModelBase

NUM_CLASSES = 5  # background, pottery, teapot, rotate_plate, rotate_medal

# ImageNet stats the pretrained ResNet18 weights were trained with.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _DecoderBlock(nn.Module):
    """Two Conv-BatchNorm-ReLU layers, same shape as the plain UNet's block."""

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


class _UNetResNet18MulNetwork(nn.Module):
    """ResNet18 encoder (pretrained, fully fine-tuned) + UNet-style decoder,
    ending in a 5-channel per-pixel softmax instead of a 1-channel sigmoid.

    See ``models.unet_resnet18._UNetResNet18Network`` for the (identical,
    other than the head) architecture notes on normalization and
    skip-connection resizing.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Split the backbone into stages so we can grab skip connections.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2,   64ch
        self.pool = backbone.maxpool  # /2 (applied before layer1)
        self.layer1 = backbone.layer1  # /4,   64ch
        self.layer2 = backbone.layer2  # /8,   128ch
        self.layer3 = backbone.layer3  # /16,  256ch
        self.layer4 = backbone.layer4  # /32,  512ch

        self.bottleneck = _DecoderBlock(512, 512)

        self.dec4 = _DecoderBlock(512 + 256, 256)
        self.dec3 = _DecoderBlock(256 + 128, 128)
        self.dec2 = _DecoderBlock(128 + 64, 64)
        self.dec1 = _DecoderBlock(64 + 64, 32)

        self.out_conv = nn.Conv2d(32, NUM_CLASSES, kernel_size=1)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._std

        e0 = self.stem(x)  # /2
        e1 = self.layer1(self.pool(e0))  # /4
        e2 = self.layer2(e1)  # /8
        e3 = self.layer3(e2)  # /16
        e4 = self.layer4(e3)  # /32

        b = self.bottleneck(e4)

        d4 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = F.interpolate(d4, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = F.interpolate(d2, size=e0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        out = self.out_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.out_act(out)


class _MulLoss(nn.Module):
    """Multi-class pixelwise NLL loss on an already-softmax'd prediction.

    ``_UNetResNet18MulNetwork`` (like every model in this project) bakes its
    own final activation into the forward pass, so the exported ONNX graph
    matches training exactly -- which means the loss must operate on
    probabilities, not logits, and can't use ``nn.CrossEntropyLoss`` (which
    expects raw logits). This is the probability-input equivalent:
    ``-log(p[target])``, averaged over every pixel.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (N, C, H, W) probabilities. target: (N, H, W) int64 class indices.
        log_p = torch.log(pred.clamp_min(self.eps))
        return F.nll_loss(log_p, target)


class UNetResNet18Mul(ModelBase):
    """UNet with a pretrained ResNet18 encoder for 320x180 5-class segmentation
    (background, pottery, teapot, rotate_plate, rotate_medal)."""

    input_shape = (180, 320, 3)  # (H, W, C)
    output_shape = (NUM_CLASSES, 180, 320)  # (C, H, W)
    num_classes = NUM_CLASSES
    HAS_PRETRAINED_ENCODER = True

    def _build_network(self) -> nn.Module:
        return _UNetResNet18MulNetwork(pretrained=True)

    def encoder_modules(self) -> list[nn.Module]:
        net = self.net
        return [net.stem, net.pool, net.layer1, net.layer2, net.layer3, net.layer4]

    def create(self, lr: float = 1e-3) -> None:
        """Same lifecycle as ModelBase.create(), except the loss is
        multi-class NLL on the network's own softmax output instead of BCE
        on a sigmoid output."""
        self.net = self._build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = _MulLoss()
        self.session = None
