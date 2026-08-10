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

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("No CUDA-capable GPU detected; falling back to CPU.")
        self.net: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.session: Optional[onnxruntime.InferenceSession] = None

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
    # Lifecycle
    # ------------------------------------------------------------------
    def create(self, lr: float = 1e-3) -> None:
        """Initialize a fresh network, optimizer and loss function for training."""
        self.net = self._build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
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
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        pred = self.net(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return float(loss.item()), pred.detach().cpu().numpy()

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

        self.net.eval()
        torch.onnx.export(
            self.net,
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
        self.net = None
        self.optimizer = None
