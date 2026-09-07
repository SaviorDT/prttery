"""Abstract base class for all models in this project.

A concrete model (e.g. ``models.unet.UNet``) only has to implement
``_build_network`` which returns a ``torch.nn.Module``. This base class takes
care of the common lifecycle: creating a fresh network, running a training
step, running inference (both on a live torch network and on a network
re-loaded from an exported ONNX file), and saving/loading the model.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import onnxruntime
import torch
from torch import nn


class _ProbabilityExport(nn.Module):
    """Expose a binary logits network as a probability-producing ONNX graph."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))


class _BceDiceLoss(nn.Module):
    """Equal-weight BCE-with-logits and soft Dice loss for binary masks."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum(dim=(1, 2, 3))
        denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + self.eps) / (denominator + self.eps)).mean()
        return 0.5 * bce + 0.5 * dice_loss


class ModelBase(ABC):
    """Common interface shared by every model.

    Attributes
    ----------
    input_shape:
        Shape of a single input sample as ``(height, width, channels)``.
    output_shape:
        Shape of a single output sample as ``(height, width)``.
    """

    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int]

    # True for models built around a pretrained encoder (currently the two
    # ResNet18-based models); drives --freeze-encoder's default and
    # validation in main.py. 'unet' has no pretrained encoder, so this stays
    # False there and encoder_modules() below is never called for it.
    HAS_PRETRAINED_ENCODER: bool = False
    OUTPUT_IS_LOGITS: bool = False

    def __init__(self, height: int = 180, width: int = 320) -> None:
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got {(height, width)}")
        channels = self.input_shape[2]
        self.input_shape = (height, width, channels)
        if len(self.output_shape) == 2:
            self.output_shape = (height, width)
        else:
            self.output_shape = (self.output_shape[0], height, width)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("No CUDA-capable GPU detected; falling back to CPU.")
        self.net: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.session: Optional[onnxruntime.InferenceSession] = None
        # While non-empty, train() re-applies .eval() to these modules after
        # every self.net.train() call, so a frozen encoder's BatchNorm/
        # LayerNorm running stats stay fixed even though the rest of the
        # network is training. Set by train.freezing.freeze_encoder() /
        # cleared by train.freezing.unfreeze_encoder().
        self._frozen_encoder_modules: list[nn.Module] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> dict:
        """Return the expected input/output shapes of this model."""
        return {"input": self.input_shape, "output": self.output_shape}

    # ------------------------------------------------------------------
    # Abstract methods subclasses must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_network(self) -> nn.Module:
        """Build and return a fresh (randomly initialized) ``nn.Module``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Pretrained encoder access (only for HAS_PRETRAINED_ENCODER models)
    # ------------------------------------------------------------------
    def encoder_modules(self) -> list[nn.Module]:
        """Return the ``nn.Module``s making up this model's pretrained
        encoder, for ``train.freezing`` to freeze/unfreeze. Only implemented
        by models with ``HAS_PRETRAINED_ENCODER = True``."""
        raise NotImplementedError(f"{type(self).__name__} has no pretrained encoder to freeze/unfreeze")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def create(self, lr: float = 1e-3, loss_name: str = "BCE") -> None:
        """Initialize a fresh network, optimizer and loss function for training."""
        self.net = self._build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        if not self.OUTPUT_IS_LOGITS:
            raise RuntimeError(f"{type(self).__name__} must provide its own create() for a non-binary loss")
        if loss_name == "BCE":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_name == "BCE_Dice":
            self.criterion = _BceDiceLoss()
        else:
            raise ValueError("loss_name must be 'BCE' or 'BCE_Dice' for a binary model")
        self.session = None

    def train(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, np.ndarray]:
        """Run one training step (forward + backward + optimizer step).

        Parameters
        ----------
        x: input batch, shape ``(N, C, H, W)``.
        y: target batch, shape ``(N, 1, H, W)`` with values in ``{0, 1}``.

        Returns
        -------
        (loss, prediction) where prediction is the model's probability output
        (numpy array in ``[0, 1]``, same shape as ``y``).
        """
        if self.net is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model has not been created. Call create() before train().")

        self.net.train()
        # self.net.train() above recursively sets every submodule (including
        # a frozen encoder's) back to training mode, which would let its
        # BatchNorm/LayerNorm running stats keep drifting despite
        # requires_grad=False. Re-pin any frozen modules to eval mode here so
        # that doesn't happen.
        for module in self._frozen_encoder_modules:
            module.eval()
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        raw_prediction = self.net(x)
        loss = self.criterion(raw_prediction, y)
        loss.backward()
        self.optimizer.step()

        prediction = torch.sigmoid(raw_prediction) if self.OUTPUT_IS_LOGITS else raw_prediction
        return float(loss.item()), prediction.detach().cpu().numpy()

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, np.ndarray]:
        """Return validation loss and probability prediction from the live network."""
        if self.net is None or self.criterion is None:
            raise RuntimeError("Validation requires a live model created with create().")
        self.net.eval()
        with torch.no_grad():
            raw_prediction = self.net(x.to(self.device, non_blocking=True))
            target = y.to(self.device, non_blocking=True)
            loss = self.criterion(raw_prediction, target)
            prediction = torch.sigmoid(raw_prediction) if self.OUTPUT_IS_LOGITS else raw_prediction
        return float(loss.item()), prediction.cpu().numpy()

    def eval(self, x: torch.Tensor) -> np.ndarray:
        """Run inference and return the predicted probability map.

        Works both right after ``create()``/``train()`` (live torch network)
        and after ``load()`` (ONNX runtime session), so the same call site can
        be used for validation during training and for the standalone eval
        pipeline.
        """
        if self.net is not None:
            self.net.eval()
            with torch.no_grad():
                x = x.to(self.device, non_blocking=True)
                pred = self.net(x)
                if self.OUTPUT_IS_LOGITS:
                    pred = torch.sigmoid(pred)
            return pred.detach().cpu().numpy()

        if self.session is not None:
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            input_name = self.session.get_inputs()[0].name
            (pred,) = self.session.run(None, {input_name: x_np.astype(np.float32)})
            return pred

        raise RuntimeError("Model has not been created or loaded. Call create() or load() first.")

    def save(self, path: str) -> None:
        """Export the current torch network to an ONNX file."""
        if self.net is None:
            raise RuntimeError("Model has not been created. Call create() before save().")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        h, w, c = self.input_shape
        dummy_input = torch.zeros(1, c, h, w, device=self.device)

        export_net: nn.Module = _ProbabilityExport(self.net) if self.OUTPUT_IS_LOGITS else self.net
        export_net.eval()
        torch.onnx.export(
            export_net,
            dummy_input,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )

    def load(self, path: str) -> None:
        """Load a model previously saved with ``save()`` for inference only."""
        # CUDAExecutionProvider is tried first; onnxruntime falls back to
        # CPUExecutionProvider automatically when no GPU is available.
        self.session = onnxruntime.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.input_shape = (input_shape[2], input_shape[3], input_shape[1])
        if len(output_shape) == 4 and isinstance(output_shape[2], int) and isinstance(output_shape[3], int):
            self.output_shape = (output_shape[2], output_shape[3]) if output_shape[1] == 1 else (output_shape[1], output_shape[2], output_shape[3])
        self.net = None
        self.optimizer = None
