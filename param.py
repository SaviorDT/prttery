"""Typed, reusable execution profiles.

Edit :data:`ACTIVE_RUN` (or :data:`ACTIVE_COPY_MASK`) to select the scenario
that ``main.py`` (or ``copy_mask.py``) executes.  Command-line arguments are
intentionally not supported.
"""

from __future__ import annotations

import glob
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from models import MODEL_REGISTRY, ModelBase, UNet, UNetResNet18, UNetResNet18Mul
from preprocessors import PREPROCESSOR_REGISTRY, CopyPastePreprocessor, PreprocessorBase


class RunMode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    EVAL_MUL = "eval_mul"
    TEST = "test"


class MaskFormat(Enum):
    BINARY = "binary"
    CVAT_6 = "cvat_6"


class LrMode(Enum):
    DEFAULT = "default"
    DECREASING = "decreasing"


class BinaryLoss(Enum):
    BCE = "BCE"
    BCE_DICE = "BCE_Dice"


class MonitoringCriterion(Enum):
    DICE = "dice"
    BOUNDARY_F1 = "boundary_f1"


def _choices(enum_type: type[Enum]) -> str:
    return ", ".join(repr(item.value) for item in enum_type)


def _expand_path_patterns(paths: tuple[Path, ...], field_name: str) -> tuple[Path, ...]:
    """Expand filesystem patterns relative to the current working directory."""
    expanded: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        pattern = str(path)
        has_magic = glob.has_magic(pattern)
        matches = tuple(Path(match) for match in glob.glob(pattern)) if has_magic else (path,)
        if has_magic and not matches:
            raise ValueError(f"{field_name} pattern matched no paths: {path}")
        for match in matches:
            if match not in seen:
                seen.add(match)
                expanded.append(match)

    return tuple(expanded)


@dataclass
class RunParams:
    """One complete train, test, or eval scenario.

    ``None`` seeds are deliberately materialized by :meth:`validate_values`
    into random integers.  The resolved seed is then printed and remains on
    this profile so a run can be reproduced.
    """

    mode: RunMode = RunMode.TRAIN
    model_class: type[ModelBase] = UNetResNet18
    dirs: tuple[Path, ...] = (Path("/videos/0816/*"),)
    height: int = 576
    width: int = 1024
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    output_dir: Path = Path("./result/")
    model_path: Path | None = None
    val_ratio: float = 0.2
    patience: int = 5
    lr_mode: LrMode = LrMode.DECREASING
    lr_decrease_rate: float = 0.5
    lr_patience: int = 3
    split_seed: int | None = None
    mask_format: MaskFormat = MaskFormat.BINARY
    mask_paths: tuple[Path, ...] = (Path("/videos/cvat_masks/*"),)
    test_mask_format: MaskFormat | None = MaskFormat.CVAT_6
    convert_mask_format: MaskFormat | None = MaskFormat.BINARY
    preprocessors: tuple[type[PreprocessorBase], ...] = (PREPROCESSOR_REGISTRY["copy_paste"],)
    copy_paste_count: int | None = 40
    copy_paste_seed: int | None = None
    freeze_encoder: bool | None = True
    unfreeze_patience: int = 4
    encoder_lr_factor: float = 0.1
    loss: BinaryLoss | None = BinaryLoss.BCE_DICE
    early_stop_check: MonitoringCriterion | None = MonitoringCriterion.DICE
    boundary_tolerance_px: int = 2

    @property
    def image_size(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def effective_mask_format(self) -> MaskFormat:
        return self.convert_mask_format or self.mask_format

    @property
    def monitor_name(self) -> str:
        return "loss" if self.early_stop_check is None else self.early_stop_check.value

    def validate_values(self) -> None:
        """Validate independent values and describe each field's legal choices."""
        enum_fields = {
            "mode": RunMode,
            "mask_format": MaskFormat,
            "lr_mode": LrMode,
        }
        for field_name, enum_type in enum_fields.items():
            value = getattr(self, field_name)
            if not isinstance(value, enum_type):
                raise ValueError(f"{field_name} must be one of {_choices(enum_type)}, got {value!r}")
        for field_name, enum_type in (("test_mask_format", MaskFormat), ("convert_mask_format", MaskFormat)):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, enum_type):
                raise ValueError(f"{field_name} must be None or one of {_choices(enum_type)}, got {value!r}")
        if self.loss is not None and not isinstance(self.loss, BinaryLoss):
            raise ValueError(f"loss must be None or one of {_choices(BinaryLoss)}, got {self.loss!r}")
        if self.early_stop_check is not None and not isinstance(self.early_stop_check, MonitoringCriterion):
            raise ValueError(
                f"early_stop_check must be None or one of {_choices(MonitoringCriterion)}, "
                f"got {self.early_stop_check!r}"
            )
        if self.model_class not in MODEL_REGISTRY.values():
            raise ValueError(f"model_class must be one of {list(MODEL_REGISTRY)}, got {self.model_class!r}")
        if not isinstance(self.dirs, tuple) or not all(isinstance(value, Path) for value in self.dirs):
            raise ValueError("dirs must be a tuple of pathlib.Path values")
        if not isinstance(self.output_dir, Path) or (self.model_path is not None and not isinstance(self.model_path, Path)):
            raise ValueError("output_dir and model_path must be pathlib.Path values (model_path may be None)")
        if not isinstance(self.mask_paths, tuple) or not all(isinstance(value, Path) for value in self.mask_paths):
            raise ValueError("mask_paths must be a tuple of pathlib.Path values")
        self.dirs = _expand_path_patterns(self.dirs, "dirs")
        self.mask_paths = _expand_path_patterns(self.mask_paths, "mask_paths")
        if not isinstance(self.preprocessors, tuple) or any(item not in PREPROCESSOR_REGISTRY.values() for item in self.preprocessors):
            raise ValueError(f"preprocessors must contain classes from {list(PREPROCESSOR_REGISTRY)}")
        for name in ("height", "width", "epochs", "batch_size", "patience", "lr_patience", "unfreeze_patience", "boundary_tolerance_px"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("lr", "lr_decrease_rate", "encoder_lr_factor"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if not isinstance(self.val_ratio, (int, float)) or not 0 <= self.val_ratio < 1:
            raise ValueError(f"val_ratio must be a number in [0, 1), got {self.val_ratio!r}")
        for name in ("split_seed", "copy_paste_seed"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer or None, got {value!r}")
        if self.copy_paste_count is not None and (not isinstance(self.copy_paste_count, int) or self.copy_paste_count < 0):
            raise ValueError(f"copy_paste_count must be a non-negative integer or None, got {self.copy_paste_count!r}")
        if self.freeze_encoder is not None and not isinstance(self.freeze_encoder, bool):
            raise ValueError(f"freeze_encoder must be bool or None, got {self.freeze_encoder!r}")

        # A missing seed means the caller requested a fresh, reproducible-by-log
        # random split/augmentation seed; store it immediately for this run.
        if self.split_seed is None:
            self.split_seed = random.randint(0, 2**31 - 1)
            print(f"Using random split seed: {self.split_seed}")
        if CopyPastePreprocessor in self.preprocessors and self.copy_paste_seed is None:
            self.copy_paste_seed = random.randint(0, 2**31 - 1)
            print(f"Using random copy-paste seed: {self.copy_paste_seed}")
        if self.model_path is None:
            self.model_path = self.output_dir / "model.onnx"

    def validate_rules(self) -> None:
        """Validate relationships between otherwise valid parameter values."""
        if not self.dirs:
            raise ValueError("dirs is required: set the source directory paths in the selected profile in param.py")
        is_multiclass = self.model_class is UNetResNet18Mul
        if self.mode is RunMode.TRAIN:
            if is_multiclass and self.effective_mask_format is not MaskFormat.CVAT_6:
                raise ValueError("UNetResNet18Mul requires CVAT_6 ground truth with no conversion")
            if self.effective_mask_format is MaskFormat.CVAT_6 and not is_multiclass:
                raise ValueError("CVAT_6 training requires UNetResNet18Mul unless it is converted to BINARY")
            if self.convert_mask_format is not None and self.mask_format is not MaskFormat.CVAT_6:
                raise ValueError("convert_mask_format is only used with CVAT_6 training masks")
            if is_multiclass and self.loss is not None:
                raise ValueError("loss is binary-only; use None for the multi-class NLL model")
            if is_multiclass and self.early_stop_check is MonitoringCriterion.BOUNDARY_F1:
                raise ValueError("boundary_f1 monitoring is currently binary-only; use None or DICE for the multi-class NLL model")
            if not is_multiclass and self.loss is None:
                raise ValueError("a binary model requires loss=BinaryLoss.BCE or BinaryLoss.BCE_DICE")
        elif self.loss is not None and is_multiclass:
            raise ValueError("loss is binary-only; use None for the multi-class NLL model")

        has_pretrained_encoder = self.model_class.HAS_PRETRAINED_ENCODER
        if self.freeze_encoder is None:
            self.freeze_encoder = has_pretrained_encoder
        elif self.freeze_encoder and not has_pretrained_encoder:
            raise ValueError(f"freeze_encoder requires a pretrained-encoder model, got {self.model_class.__name__}")
        if self.freeze_encoder and self.unfreeze_patience > self.patience:
            raise ValueError("unfreeze_patience must be <= patience so early stopping cannot prevent unfreezing")

        if self.test_mask_format is not None and self.mode is not RunMode.TEST:
            raise ValueError("test_mask_format is only used with mode=TEST")
        if self.test_mask_format is None:
            self.test_mask_format = MaskFormat.BINARY
        if self.convert_mask_format is not None and self.mode not in (RunMode.TRAIN, RunMode.TEST):
            raise ValueError("convert_mask_format is only used with mode=TRAIN or mode=TEST")
        if self.mode is RunMode.TEST and self.test_mask_format is not self.mask_format and self.convert_mask_format is None:
            raise ValueError("test_mask_format differing from mask_format requires convert_mask_format")

        needs_cvat = (self.mode is RunMode.TRAIN and self.mask_format is MaskFormat.CVAT_6) or (
            self.mode is RunMode.TEST and self.test_mask_format is MaskFormat.CVAT_6
        )
        if needs_cvat and not self.mask_paths:
            raise ValueError("CVAT ground truth requires mask_paths")
        if not needs_cvat and self.mask_paths:
            raise ValueError("mask_paths is only used when the active train/test ground truth is CVAT_6")
        if self.preprocessors and self.mode is not RunMode.TRAIN:
            raise ValueError("preprocessors are only used with mode=TRAIN")
        if CopyPastePreprocessor in self.preprocessors and self.copy_paste_count is None:
            raise ValueError("CopyPastePreprocessor requires copy_paste_count")
        if self.lr_mode is LrMode.DECREASING and self.val_ratio == 0:
            raise ValueError("lr_mode=DECREASING requires validation data")
        if self.freeze_encoder and self.val_ratio == 0:
            raise ValueError("freeze_encoder requires validation data")

    def validate(self) -> None:
        self.validate_values()
        self.validate_rules()


@dataclass
class CopyMaskParams:
    src_root: Path = Path("/videos")
    dst_root: Path = Path("./masks")
    dry_run: bool = False

    def validate_values(self) -> None:
        if not isinstance(self.src_root, Path) or not isinstance(self.dst_root, Path):
            raise ValueError("src_root and dst_root must be pathlib.Path values")
        if not isinstance(self.dry_run, bool):
            raise ValueError(f"dry_run must be bool, got {self.dry_run!r}")

    def validate_rules(self) -> None:
        if not self.src_root.is_dir():
            raise ValueError(f"source directory does not exist: {self.src_root}")

    def validate(self) -> None:
        self.validate_values()
        self.validate_rules()


# Fill ``dirs`` with your dataset directory paths before running main.py.
CVAT_MULTICLASS = RunParams(model_class=UNetResNet18Mul, mask_format=MaskFormat.CVAT_6, loss=None)
DEFAULT = RunParams()
BINARY_TRAIN = RunParams(mode=RunMode.TRAIN, test_mask_format=None)
BINARY_TEST = RunParams(mode=RunMode.TEST, preprocessors=())
BINARY_EVAL = RunParams(model_path=Path("./result/model.onnx"), output_dir=Path("/videos/results2/"), mode=RunMode.EVAL, test_mask_format=None, convert_mask_format=None, mask_paths=(), preprocessors=())
ACTIVE_RUN = BINARY_EVAL
ACTIVE_COPY_MASK = CopyMaskParams()
