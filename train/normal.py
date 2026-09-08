"""The most basic training routine: train on labeled data, validate on val
data to detect overfitting (early stopping), print metrics to the console,
and export the best model to ONNX.
"""

from __future__ import annotations

import copy
import os

import cv2

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader.normal import (
    EvalDataset, EvalItem, eval_collate, get_test_dataset,
    get_train_val_datasets,
)
from eval_pipeline import load_eval_model, predict_eval_batch, to_native_eval_prediction
from mask_formats import binary_confusion_metrics, get_mask_format, to_binary
from models.base import ModelBase
from train import freezing


class _BinaryFromCvatDataset(Dataset):
    """Wraps a data_loader.cvat.CvatSegmentationDataset, collapsing its
    6-class (H, W) int64 mask into a (1, H, W) float32 {0, 1} target via
    mask_formats.to_binary -- lets --mask-format cvat_6 --convert-mask-format
    binary train a binary model directly off CVAT-labeled data, with no
    separate binary-mask export and no change to the training loop below."""

    def __init__(self, cvat_dataset) -> None:
        self.cvat_dataset = cvat_dataset

    def __len__(self) -> int:
        return len(self.cvat_dataset)

    def __getitem__(self, index: int):
        image_tensor, class_index_tensor = self.cvat_dataset[index]
        binary = to_binary(class_index_tensor.numpy(), "cvat_6")
        mask_tensor = torch.from_numpy(binary[np.newaxis, :, :])  # (1, H, W)
        return image_tensor, mask_tensor


def compute_metrics(pred_prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute pixel accuracy / IoU / precision / recall / F1 for a batch.

    Both arguments are numpy arrays of the same shape, ``target`` in {0, 1}
    and ``pred_prob`` in [0, 1].
    """
    pred = (pred_prob > threshold).astype(np.float32)
    return binary_confusion_metrics(pred, target)


def _bce_loss(pred_prob: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(pred_prob, eps, 1 - eps)
    return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))


def _boundary_mask_2d(mask: np.ndarray) -> np.ndarray:
    """Normalize a mask with an optional singleton channel to two dimensions."""
    value = np.asarray(mask)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError("boundary metrics require two-dimensional masks")
    return value


def _boundary_f1_items(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    tolerance_px: int,
) -> float:
    """Boundary F1 for binary masks, including differently sized images."""
    if len(predictions) != len(targets):
        raise ValueError("boundary prediction and target batch sizes differ")

    total_precision_hits = total_pred = total_recall_hits = total_true = 0
    for pred_item, target_item in zip(predictions, targets, strict=True):
        pred_item = _boundary_mask_2d(pred_item)
        target_item = _boundary_mask_2d(target_item)
        if pred_item.shape != target_item.shape:
            raise ValueError("boundary prediction and target shapes differ")
        pred = (pred_item > 0.5).astype(np.uint8)
        truth = (target_item > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        pred_edge = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel)
        truth_edge = cv2.morphologyEx(truth, cv2.MORPH_GRADIENT, kernel)
        pred_count, truth_count = int(pred_edge.sum()), int(truth_edge.sum())
        if pred_count == truth_count == 0:
            total_precision_hits += 1; total_pred += 1; total_recall_hits += 1; total_true += 1
            continue
        if pred_count:
            distance_to_truth = cv2.distanceTransform((truth_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
            total_precision_hits += int((distance_to_truth[pred_edge.astype(bool)] <= tolerance_px).sum())
            total_pred += pred_count
        if truth_count:
            distance_to_pred = cv2.distanceTransform((pred_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
            total_recall_hits += int((distance_to_pred[truth_edge.astype(bool)] <= tolerance_px).sum())
            total_true += truth_count
    precision = total_precision_hits / max(total_pred, 1)
    recall = total_recall_hits / max(total_true, 1)
    return 2 * precision * recall / max(precision + recall, 1e-7)


def _boundary_f1(pred_prob: np.ndarray, target: np.ndarray, tolerance_px: int) -> float:
    """Boundary F1 for an equal-size binary batch."""
    return _boundary_f1_items(list(pred_prob[:, 0]), list(target[:, 0]), tolerance_px)


def _native_boundary_f1(predictions: np.ndarray, dataset, tolerance_px: int) -> float:
    """Restore each validation prediction to its annotation's native size."""
    scores = []
    if hasattr(dataset, "pairs"):
        records = dataset.pairs
        for prediction, (_image_path, mask_path) in zip(predictions, records):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Failed to read mask: {mask_path}")
            target = (mask > 127).astype(np.float32)
            probability = cv2.resize(prediction[0], (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
            scores.append(_boundary_f1(probability[None, None], target[None, None], tolerance_px))
    elif hasattr(dataset, "cvat_dataset"):
        from data_loader.cvat import render_mask
        cvat_dataset = dataset.cvat_dataset
        for prediction, (_image_path, shapes, width, height) in zip(predictions, cvat_dataset.triples):
            classes = render_mask(shapes, height, width, cvat_dataset.label_map)
            target = to_binary(classes[None], "cvat_6")[0]
            probability = cv2.resize(prediction[0], (width, height), interpolation=cv2.INTER_LINEAR)
            scores.append(_boundary_f1(probability[None, None], target[None, None], tolerance_px))
    else:
        raise TypeError(f"Unsupported validation dataset for native Boundary F1: {type(dataset).__name__}")
    return float(np.mean(scores))


def _average_metrics(metric_dicts: list[dict]) -> dict:
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}


def _format_metrics(prefix: str, loss: float | None, metrics: dict) -> str:
    parts = ([f"loss={loss:.4f}"] if loss is not None else []) + [f"{k}={v:.4f}" for k, v in metrics.items()]
    return f"[{prefix}] " + " ".join(parts)


class BinaryTestEvaluator:
    """Shared batch evaluation and reporting for TEST and EVAL_TEST."""

    def __init__(
        self,
        test_mask_format: str,
        convert_mask_format: str | None,
        boundary_tolerance_px: int = 2,
    ) -> None:
        self.native_format = "binary"
        self.test_mask_format = test_mask_format
        self.convert_mask_format = convert_mask_format
        self.boundary_tolerance_px = boundary_tolerance_px
        self.losses: list[float] = []
        self.metrics: list[dict] = []

    def add_batch(
        self,
        predictions: np.ndarray | list[np.ndarray],
        targets: np.ndarray | list[np.ndarray],
        binary_predictions: list[np.ndarray] | None = None,
    ) -> None:
        """Evaluate one inference batch, including variable native image sizes."""
        prediction_items = list(predictions)
        target_items = list(targets)
        if len(prediction_items) != len(target_items):
            raise ValueError("prediction and target batch sizes differ")
        if binary_predictions is not None and len(binary_predictions) != len(prediction_items):
            raise ValueError("binary prediction and probability batch sizes differ")

        probability_flat = np.concatenate(
            [np.asarray(item, dtype=np.float32).reshape(-1) for item in prediction_items]
        )

        if self.test_mask_format == self.native_format:
            target_binary = [
                np.asarray(item, dtype=np.float32).reshape(-1)
                for item in target_items
            ]
            target_flat = np.concatenate(target_binary)
            if probability_flat.size != target_flat.size:
                raise ValueError("prediction and target pixel counts differ")
            self.losses.append(_bce_loss(probability_flat, target_flat))

            if binary_predictions is None:
                prediction_flat = (probability_flat > 0.5).astype(np.float32)
                boundary_predictions = prediction_items
            else:
                prediction_flat = np.concatenate(
                    [np.asarray(item, dtype=np.float32).reshape(-1) for item in binary_predictions]
                )
                boundary_predictions = binary_predictions
            boundary_targets = target_items
        else:
            if self.convert_mask_format != "binary":
                raise ValueError(
                    f"Unsupported --convert-mask-format '{self.convert_mask_format}'; "
                    "only 'binary' is supported"
                )
            target_format = get_mask_format(self.test_mask_format)
            converted_targets = []
            for target in target_items:
                target_array = np.asarray(target)
                target_index = target_format.ground_truth_to_class_index(
                    target_array[np.newaxis, ...]
                )
                converted_targets.append(
                    to_binary(target_index, self.test_mask_format)[0]
                )
            target_flat = np.concatenate([item.reshape(-1) for item in converted_targets])

            if binary_predictions is None:
                native_format = get_mask_format(self.native_format)
                converted_predictions = []
                for prediction in prediction_items:
                    prediction_array = np.asarray(prediction)
                    if prediction_array.ndim == 2:
                        prediction_array = prediction_array[np.newaxis, ...]
                    prediction_index = native_format.raw_output_to_class_index(
                        prediction_array[np.newaxis, ...]
                    )
                    converted_predictions.append(
                        to_binary(prediction_index, self.native_format)[0].reshape(-1)
                    )
                prediction_flat = np.concatenate(converted_predictions)
                boundary_predictions = prediction_items
            else:
                prediction_flat = np.concatenate(
                    [np.asarray(item, dtype=np.float32).reshape(-1) for item in binary_predictions]
                )
                boundary_predictions = binary_predictions
            boundary_targets = converted_targets

        if prediction_flat.size != target_flat.size:
            raise ValueError("prediction and target pixel counts differ")
        metrics = binary_confusion_metrics(prediction_flat, target_flat)
        metrics["boundary_f1"] = _boundary_f1_items(
            boundary_predictions,
            boundary_targets,
            self.boundary_tolerance_px,
        )
        self.metrics.append(metrics)

    def print_report(self, title: str, prefix: str) -> None:
        loss = float(np.mean(self.losses)) if self.losses else None
        metrics = _average_metrics(self.metrics)
        print()
        print("=" * 60)
        print(title)
        print(_format_metrics(prefix, loss, metrics))
        print("=" * 60)


def run(
    dirs: list[str],
    model_class: type[ModelBase],
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    model_path: str = "./result/model.onnx",
    val_ratio: float = 0.2,
    patience: int = 5,
    lr_mode: str = "default",
    lr_decrease_rate: float = 0.5,
    lr_patience: int = 3,
    split_seed: int = 42,
    preprocessor: list[str] | None = None,
    copy_paste_count: int | None = None,
    copy_paste_seed: int | None = None,
    freeze_encoder: bool = False,
    unfreeze_patience: int = 4,
    encoder_lr_factor: float = 0.1,
    mask_format: str = "binary",
    mask_paths: list[str] | None = None,
    image_size: tuple[int, int] = (180, 320),
    loss_name: str = "BCE",
    early_stop_check: str | None = None,
    boundary_tolerance_px: int = 2,
) -> None:
    if mask_format == "cvat_6":
        # Ground truth is CVAT-labeled (6-class), but --convert-mask-format binary (enforced by
        # main.py's argument parsing whenever mask_format is cvat_6 here) collapses it to binary
        # before this binary model ever sees it -- see _BinaryFromCvatDataset above.
        from data_loader.cvat import get_train_val_datasets as get_cvat_train_val_datasets

        cvat_train_dataset, cvat_val_dataset = get_cvat_train_val_datasets(
            dirs, mask_paths, val_ratio=val_ratio, seed=split_seed, image_size=image_size
        )
        train_dataset = _BinaryFromCvatDataset(cvat_train_dataset)
        val_dataset = _BinaryFromCvatDataset(cvat_val_dataset) if cvat_val_dataset is not None else None
    else:
        train_dataset, val_dataset = get_train_val_datasets(dirs, val_ratio=val_ratio, seed=split_seed, image_size=image_size)
    has_val = val_dataset is not None
    original_train_count = len(train_dataset)

    if preprocessor:
        from preprocessors import apply_preprocessors

        train_dataset = apply_preprocessors(
            train_dataset, preprocessor, copy_paste_count=copy_paste_count, copy_paste_seed=copy_paste_seed
        )
        print(
            f"Train samples: {len(train_dataset)} ({original_train_count} original + "
            f"{len(train_dataset) - original_train_count} synthetic), Val samples: {len(val_dataset) if has_val else 0}"
        )
    else:
        print(f"Train samples: {original_train_count}, Val samples: {len(val_dataset) if has_val else 0}")

    if not has_val:
        print("No val data available: skipping validation and early stopping.")

    if lr_mode == "decreasing" and not has_val:
        raise SystemExit("Error: --lr-mode decreasing requires validation data, but none is available.")

    if freeze_encoder and not has_val:
        raise SystemExit(
            "Error: --freeze-encoder requires validation data (it unfreezes on a val-loss plateau), "
            "but none is available. Pass --no-freeze-encoder to train directly instead."
        )

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        if has_val
        else None
    )

    model = model_class(*image_size)
    model.create(lr=lr, loss_name=loss_name)

    if freeze_encoder:
        freezing.freeze_encoder(model)
        tqdm.write(f"Encoder frozen; will unfreeze after {unfreeze_patience} epoch(s) without validation-monitor improvement.")
    encoder_frozen = freeze_encoder

    best_monitor_value = float("inf") if early_stop_check is None else float("-inf")
    best_state = None
    no_improve_count = 0
    lr_no_improve_count = 0
    unfreeze_no_improve_count = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        current_lr = model.optimizer.param_groups[0]["lr"]
        tqdm.write(f"Epoch {epoch}/{epochs} lr={current_lr:.6g}")

        # ---- training ----
        train_losses = []
        train_metrics = []
        for x, y in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            loss, pred = model.train(x, y)
            train_losses.append(loss)
            train_metrics.append(compute_metrics(pred, y.numpy()))

        train_loss = float(np.mean(train_losses))
        train_metric_avg = _average_metrics(train_metrics)
        tqdm.write(f"Epoch {epoch}/{epochs} " + _format_metrics("train", train_loss, train_metric_avg))

        if not has_val:
            continue

        # ---- validation ----
        val_losses = []
        val_metrics = []
        boundary_scores = []
        val_predictions = []
        for x, y in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
            batch_loss, pred = model.evaluate(x, y)
            y_np = y.numpy()
            val_losses.append(batch_loss)
            val_metrics.append(compute_metrics(pred, y_np))
            val_predictions.append(pred)

        val_loss = float(np.mean(val_losses))
        val_metric_avg = _average_metrics(val_metrics)
        boundary_f1 = _native_boundary_f1(np.concatenate(val_predictions), val_dataset, boundary_tolerance_px)
        tqdm.write(f"Epoch {epoch}/{epochs} " + _format_metrics("val", val_loss, val_metric_avg) + f" boundary_f1={boundary_f1:.4f}")
        monitor_value = val_loss if early_stop_check is None else (val_metric_avg["f1"] if early_stop_check == "dice" else boundary_f1)
        improved = monitor_value < best_monitor_value if early_stop_check is None else monitor_value > best_monitor_value

        # ---- checkpoint / early stopping / lr decay ----
        if improved:
            best_monitor_value = monitor_value
            best_state = copy.deepcopy(model.net.state_dict())
            no_improve_count = 0
            lr_no_improve_count = 0
            unfreeze_no_improve_count = 0
            tqdm.write(f"Epoch {epoch}: validation {early_stop_check or 'loss'} improved to {monitor_value:.4f}, saving best weights")
        else:
            no_improve_count += 1
            lr_no_improve_count += 1
            unfreeze_no_improve_count += 1
            tqdm.write(f"Epoch {epoch}: no improvement ({no_improve_count}/{patience})")

            # Checked before lr-decay/early-stop below: unfreezing is a bigger, more disruptive
            # change than either of those, so it gets first crack at a plateau, and resets every
            # counter (not just its own) so lr-decay/early-stopping get a fresh window to judge
            # the now-unfrozen model on, instead of carrying over stale frozen-phase counts. This
            # also means unfreeze_patience == patience is safe to allow (validated in main.py as
            # `<=`, not `<`): the first time no_improve_count would hit patience, this branch
            # intercepts it and resets it to 0 before the early-stop check below ever sees it;
            # only a second, post-unfreeze run to patience (with encoder_frozen now False) really
            # stops training.
            if encoder_frozen and unfreeze_no_improve_count >= unfreeze_patience:
                freezing.unfreeze_encoder(model, encoder_lr_factor)
                encoder_frozen = False
                no_improve_count = 0
                lr_no_improve_count = 0
                unfreeze_no_improve_count = 0
                decoder_lr = model.optimizer.param_groups[0]["lr"]
                encoder_lr = model.optimizer.param_groups[-1]["lr"]
                tqdm.write(
                    f"Epoch {epoch}: no improvement for {unfreeze_patience} epochs, unfreezing encoder "
                    f"(encoder lr={encoder_lr:.6g}, decoder lr={decoder_lr:.6g})"
                )

            if lr_mode == "decreasing" and lr_no_improve_count >= lr_patience:
                for param_group in model.optimizer.param_groups:
                    param_group["lr"] *= lr_decrease_rate
                current_lr = model.optimizer.param_groups[0]["lr"]
                tqdm.write(
                    f"Epoch {epoch}: no improvement for {lr_patience} epochs, decreasing lr to {current_lr:.6g}"
                )
                lr_no_improve_count = 0

            if no_improve_count >= patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.net.load_state_dict(best_state)

    # ---- final "test" evaluation (val data doubles as test data; falls
    # back to train data when no val data is available) ----
    final_loader = val_loader if has_val else train_loader
    final_losses = []
    final_metrics = []
    for x, y in final_loader:
        batch_loss, pred = model.evaluate(x, y)
        y_np = y.numpy()
        final_losses.append(batch_loss)
        final_metrics.append(compute_metrics(pred, y_np))

    final_loss = float(np.mean(final_losses))
    final_metric_avg = _average_metrics(final_metrics)
    print()
    print("=" * 60)
    print("Final Test (= Val) Metrics" if has_val else "Final Train Metrics")
    print(_format_metrics("test" if has_val else "train", final_loss, final_metric_avg))
    print("=" * 60)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")


def _build_binary_test_dataset(
    dirs: list[str],
    mask_paths: list[str] | None,
    test_mask_format: str,
    image_size: tuple[int, int],
):
    if test_mask_format == "binary":
        return get_test_dataset(dirs, image_size=image_size)

    from data_loader.cvat import get_test_dataset as get_cvat_test_dataset

    return get_cvat_test_dataset(dirs, mask_paths, image_size=image_size)


def run_test(
    dirs: list[str],
    model_path: str,
    model_class: type[ModelBase],
    batch_size: int = 8,
    mask_paths: list[str] | None = None,
    test_mask_format: str = "binary",
    convert_mask_format: str | None = None,
    image_size: tuple[int, int] = (180, 320),
    boundary_tolerance_px: int = 2,
) -> None:
    """Evaluate a saved model (native format: binary) against all labeled data
    (train and val combined).

    Ground truth is normally binary, read from ``./data/{dir}_mask/`` as
    usual. If ``test_mask_format`` is ``'cvat_6'`` instead -- the test set's
    ground truth is only available as CVAT annotations -- it's loaded via
    ``mask_paths`` instead, and both it and the model's (native binary)
    prediction are collapsed into ``convert_mask_format`` (currently only
    ``'binary'`` is supported) before being compared, since a binary
    prediction can't be compared directly against non-binary ground truth.
    main.py's argument parsing guarantees ``convert_mask_format`` is set
    whenever ``test_mask_format`` differs from this module's native format.
    """
    native_format = "binary"

    test_dataset = _build_binary_test_dataset(dirs, mask_paths, test_mask_format, image_size)
    print(f"Test samples: {len(test_dataset)}")

    num_workers = min(4, os.cpu_count() or 1)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    model = load_eval_model(model_class, model_path, image_size)
    evaluator = BinaryTestEvaluator(
        test_mask_format, convert_mask_format, boundary_tolerance_px
    )
    for x, y in tqdm(test_loader, desc="Test"):
        pred = predict_eval_batch(model, x, native_format, model_path)
        evaluator.add_batch(pred, y.numpy())

    evaluator.print_report("Test Metrics", "test")


def run_eval_test(
    dirs: list[str],
    model_path: str,
    model_class: type[ModelBase],
    batch_size: int = 8,
    mask_paths: list[str] | None = None,
    test_mask_format: str = "binary",
    convert_mask_format: str | None = None,
    image_size: tuple[int, int] = (180, 320),
    boundary_tolerance_px: int = 2,
) -> None:
    """Evaluate the exact EVAL output transformation against native targets."""
    native_format = "binary"
    test_dataset = _build_binary_test_dataset(
        dirs, mask_paths, test_mask_format, image_size
    )
    print(f"Eval test samples: {len(test_dataset)}")

    items = [
        EvalItem(image_path=test_dataset.image_path(index), second=0, frame=index)
        for index in range(len(test_dataset))
    ]
    eval_dataset = EvalDataset(items, image_size=image_size, strict=True)
    num_workers = min(4, os.cpu_count() or 1)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=eval_collate,
    )

    model = load_eval_model(model_class, model_path, image_size)
    evaluator = BinaryTestEvaluator(
        test_mask_format, convert_mask_format, boundary_tolerance_px
    )
    sample_index = 0

    for tensors, batch_items, valid_mask in tqdm(eval_loader, desc="Eval Test"):
        if tensors is None or not all(valid_mask):
            failed = next(
                item.image_path
                for item, is_valid in zip(batch_items, valid_mask)
                if not is_valid
            )
            raise FileNotFoundError(f"Failed to read image: {failed}")

        predictions = predict_eval_batch(model, tensors, native_format, model_path)
        native_probabilities = []
        native_masks = []
        native_targets = []

        for prediction, item in zip(predictions, batch_items, strict=True):
            expected_path = test_dataset.image_path(sample_index)
            if item.image_path != expected_path:
                raise RuntimeError("eval-test dataset order changed during inference")

            target = test_dataset.native_target(sample_index)
            image = cv2.imread(item.image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {item.image_path}")
            image_size_native = image.shape[:2]
            target_size_native = target.shape[-2:]
            if image_size_native != target_size_native:
                raise ValueError(
                    f"Image '{item.image_path}' has size {image_size_native}, "
                    f"but its annotation has size {target_size_native}"
                )

            native_prediction = to_native_eval_prediction(
                prediction, native_format, image_size_native
            )
            native_probabilities.append(native_prediction.probability[np.newaxis, :, :])
            native_masks.append(native_prediction.mask)
            native_targets.append(target)
            sample_index += 1

        evaluator.add_batch(
            native_probabilities,
            native_targets,
            binary_predictions=native_masks,
        )

    if sample_index != len(test_dataset):
        raise RuntimeError(f"evaluated {sample_index} samples, expected {len(test_dataset)}")
    evaluator.print_report("Eval Test Metrics", "eval_test")
