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

from data_loader.normal import get_test_dataset, get_train_val_datasets
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


def _boundary_f1(pred_prob: np.ndarray, target: np.ndarray, tolerance_px: int) -> float:
    """Boundary F1 for binary batches, using a pixel-distance tolerance."""
    total_precision_hits = total_pred = total_recall_hits = total_true = 0
    for pred_item, target_item in zip(pred_prob[:, 0], target[:, 0]):
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


def run_test(
    dirs: list[str],
    model_path: str,
    model_class: type[ModelBase],
    batch_size: int = 8,
    mask_paths: list[str] | None = None,
    test_mask_format: str = "binary",
    convert_mask_format: str | None = None,
    image_size: tuple[int, int] = (180, 320),
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

    if test_mask_format == native_format:
        test_dataset = get_test_dataset(dirs, image_size=image_size)
    else:
        from data_loader.cvat import get_test_dataset as get_cvat_test_dataset

        test_dataset = get_cvat_test_dataset(dirs, mask_paths, image_size=image_size)
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

    model = model_class()
    model.load(model_path)

    same_format = test_mask_format == native_format
    test_losses = []
    test_metrics = []
    for x, y in tqdm(test_loader, desc="Test"):
        pred = model.eval(x)
        y_np = y.numpy()

        if same_format:
            test_losses.append(_bce_loss(pred, y_np))
            test_metrics.append(compute_metrics(pred, y_np))
        else:
            if convert_mask_format != "binary":
                raise ValueError(f"Unsupported --convert-mask-format '{convert_mask_format}'; only 'binary' is supported")
            # Loss isn't meaningful across formats (the ground truth doesn't
            # match what the model was trained to predict), so only metrics
            # are reported in this branch.
            pred_class_idx = get_mask_format(native_format).raw_output_to_class_index(pred)
            gt_class_idx = get_mask_format(test_mask_format).ground_truth_to_class_index(y_np)
            pred_binary = to_binary(pred_class_idx, native_format)
            gt_binary = to_binary(gt_class_idx, test_mask_format)
            test_metrics.append(binary_confusion_metrics(pred_binary, gt_binary))

    test_metric_avg = _average_metrics(test_metrics)
    test_loss = float(np.mean(test_losses)) if test_losses else None
    print()
    print("=" * 60)
    print("Test Metrics")
    print(_format_metrics("test", test_loss, test_metric_avg))
    print("=" * 60)
