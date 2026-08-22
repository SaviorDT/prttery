"""Per-mask-format definitions shared by --mode test's cross-format
conversion (--convert-mask-format / --test-mask-format, see main.py) and
--mode eval's multi-class foreground collapse.

Each mask format owns its own class, mirroring the existing
``data_loader.cvat.CvatLabelMap`` convention of "one class owns its own
class-index semantics" -- adding a new mask format later means adding a new
``MaskFormat`` subclass and registering it in ``MASK_FORMATS`` below, nothing
else in this module (or its callers) needs to change.
"""

from __future__ import annotations

import numpy as np


class MaskFormat:
    """Shared interface every concrete mask format below implements."""

    name: str

    def background_classes(self) -> set[int]:
        """Class indices this format considers background."""
        raise NotImplementedError

    def raw_output_to_class_index(self, raw: np.ndarray) -> np.ndarray:
        """Collapse a model's raw ``(N, C, H, W)`` probability output into a
        discrete ``(N, H, W)`` class-index array."""
        raise NotImplementedError

    def ground_truth_to_class_index(self, target: np.ndarray) -> np.ndarray:
        """Normalize a loaded ground-truth array into the same ``(N, H, W)``
        class-index shape as ``raw_output_to_class_index``."""
        raise NotImplementedError

    def class_names(self) -> list[str] | None:
        """Display name for each class index (``names[i]`` labels class ``i``),
        or ``None`` if this format has no names to offer. Optional -- only
        consumed by --mode eval_mul's legend (see outputer.overlay), which
        degrades gracefully (no legend, one warning) when this is ``None``
        rather than requiring every format to invent names."""
        return None


class BinaryFormat(MaskFormat):
    name = "binary"

    def background_classes(self) -> set[int]:
        return {0}

    def raw_output_to_class_index(self, raw: np.ndarray) -> np.ndarray:
        # (N, 1, H, W) foreground probability in [0, 1] -> (N, H, W) in {0, 1}.
        return (raw[:, 0, :, :] > 0.5).astype(np.int64)

    def ground_truth_to_class_index(self, target: np.ndarray) -> np.ndarray:
        # (N, 1, H, W) in {0, 1} -> (N, H, W) in {0, 1}.
        return target[:, 0, :, :].astype(np.int64)


class Cvat6Format(MaskFormat):
    name = "cvat_6"

    def __init__(self) -> None:
        from data_loader.cvat import CvatLabelMap

        self._label_map = CvatLabelMap()

    def background_classes(self) -> set[int]:
        return {self._label_map.background_index}

    def raw_output_to_class_index(self, raw: np.ndarray) -> np.ndarray:
        # (N, 6, H, W) per-pixel softmax -> (N, H, W) argmax class index.
        return np.argmax(raw, axis=1).astype(np.int64)

    def ground_truth_to_class_index(self, target: np.ndarray) -> np.ndarray:
        # already (N, H, W) class-index.
        return target.astype(np.int64)

    def class_names(self) -> list[str] | None:
        return self._label_map.CLASSES


MASK_FORMATS: dict[str, MaskFormat] = {
    "binary": BinaryFormat(),
    "cvat_6": Cvat6Format(),
}


def get_mask_format(name: str) -> MaskFormat:
    try:
        return MASK_FORMATS[name]
    except KeyError:
        raise ValueError(f"Unknown mask format '{name}'; known formats: {sorted(MASK_FORMATS)}") from None


def to_binary(class_index: np.ndarray, format_name: str) -> np.ndarray:
    """Collapse a ``(N, H, W)`` class-index array into a ``(N, H, W)`` binary
    array using ``format_name``'s own background definition: background
    classes -> 0, everything else -> 1."""
    background = list(get_mask_format(format_name).background_classes())
    return np.isin(class_index, background, invert=True).astype(np.float32)


def binary_confusion_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Pixel accuracy / IoU / precision / recall / F1 between two arrays of
    the same shape, both already in {0, 1}.

    Factored out of train.normal.compute_metrics / train.multiclass's
    foreground-collapse metrics so every binary comparison in this project
    (native binary, multi-class foreground collapse, or a cross-format
    --convert-mask-format conversion) computes it identically.
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    tp = float(np.sum((pred == 1) & (target == 1)))
    tn = float(np.sum((pred == 0) & (target == 0)))
    fp = float(np.sum((pred == 1) & (target == 0)))
    fn = float(np.sum((pred == 0) & (target == 1)))

    eps = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": accuracy,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
