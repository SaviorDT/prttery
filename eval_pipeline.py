"""Inference and native-resolution postprocessing shared by eval modes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from mask_formats import get_mask_format, to_binary
from models.base import ModelBase


def load_eval_model(
    model_class: type[ModelBase],
    model_path: str,
    expected_image_size: tuple[int, int],
) -> ModelBase:
    """Load a checkpoint and verify its input size against the run profile."""
    model = model_class(*expected_image_size)
    model.load(model_path)

    actual_image_size = model.input_shape[:2]
    if actual_image_size != expected_image_size:
        raise ValueError(
            f"checkpoint input size {actual_image_size} does not match "
            f"profile image_size {expected_image_size}"
        )
    return model


def predict_eval_batch(
    model: ModelBase,
    tensors: torch.Tensor,
    mask_format: str,
    model_path: str,
) -> np.ndarray:
    """Run inference and reject checkpoints whose channels contradict their format."""
    predictions = model.eval(tensors)
    num_channels = predictions.shape[1]
    is_multiclass = mask_format == "cvat_6"
    expected = "> 1 (cvat_6)" if is_multiclass else "1 (binary)"
    if is_multiclass and num_channels <= 1:
        raise RuntimeError(
            f"--mask-format cvat_6 but '{model_path}' only outputs {num_channels} "
            f"channel(s) (expected {expected}); wrong --model-path, or "
            "--mask-format doesn't match this checkpoint"
        )
    if not is_multiclass and num_channels != 1:
        raise RuntimeError(
            f"--mask-format binary but '{model_path}' outputs {num_channels} channels "
            f"(expected {expected}); wrong --model-path, or --mask-format doesn't "
            "match this checkpoint"
        )
    return predictions


@dataclass(frozen=True)
class NativeEvalPrediction:
    probability: np.ndarray
    mask: np.ndarray


def to_native_eval_prediction(
    prediction: np.ndarray,
    mask_format: str,
    image_size: tuple[int, int],
) -> NativeEvalPrediction:
    """Apply eval's foreground collapse, resize, and binary threshold."""
    if mask_format == "binary":
        foreground_probability = prediction[0].astype(np.float32)
    else:
        class_index = get_mask_format(mask_format).raw_output_to_class_index(
            prediction[np.newaxis, ...]
        )
        foreground_probability = to_binary(class_index, mask_format)[0]

    height, width = image_size
    probability = cv2.resize(
        foreground_probability,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return NativeEvalPrediction(
        probability=probability,
        mask=(probability > 0.5).astype(np.float32),
    )
