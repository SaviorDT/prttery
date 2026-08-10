"""Registry mapping ``--model`` CLI values to model classes."""

from __future__ import annotations

from models.base import ModelBase
from models.unet import UNet
from models.unet_resnet18 import UNetResNet18
from models.unet_resnet18_mul import UNetResNet18Mul

MODEL_REGISTRY: dict[str, type[ModelBase]] = {
    "unet": UNet,
    "unet_resnet18": UNetResNet18,
    "unet_resnet18_mul": UNetResNet18Mul,
}


def get_model_class(name: str) -> type[ModelBase]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Choices: {sorted(MODEL_REGISTRY)}") from None
