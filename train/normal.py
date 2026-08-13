"""The most basic training routine: train on labeled data, validate on val
data to detect overfitting (early stopping), print metrics to the console,
and export the best model to ONNX.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.normal import get_test_dataset, get_train_val_datasets
from mask_formats import binary_confusion_metrics, get_mask_format, to_binary
from models import get_model_class


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


def _average_metrics(metric_dicts: list[dict]) -> dict:
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}


def _format_metrics(prefix: str, loss: float | None, metrics: dict) -> str:
    parts = ([f"loss={loss:.4f}"] if loss is not None else []) + [f"{k}={v:.4f}" for k, v in metrics.items()]
    return f"[{prefix}] " + " ".join(parts)


def run(
    dirs: list[str],
    model_name: str = "unet",
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
) -> None:
    train_dataset, val_dataset = get_train_val_datasets(dirs, val_ratio=val_ratio, seed=split_seed)
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

    model = get_model_class(model_name)()
    model.create(lr=lr)

    best_val_loss = float("inf")
    best_state = None
    no_improve_count = 0
    lr_no_improve_count = 0

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
        for x, y in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
            pred = model.eval(x)
            y_np = y.numpy()
            val_losses.append(_bce_loss(pred, y_np))
            val_metrics.append(compute_metrics(pred, y_np))

        val_loss = float(np.mean(val_losses))
        val_metric_avg = _average_metrics(val_metrics)
        tqdm.write(f"Epoch {epoch}/{epochs} " + _format_metrics("val", val_loss, val_metric_avg))

        # ---- early stopping / lr decay ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.net.state_dict())
            no_improve_count = 0
            lr_no_improve_count = 0
            tqdm.write(f"Epoch {epoch}: val loss improved to {val_loss:.4f}, saving best weights")
        else:
            no_improve_count += 1
            lr_no_improve_count += 1
            tqdm.write(f"Epoch {epoch}: no improvement ({no_improve_count}/{patience})")

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
        pred = model.eval(x)
        y_np = y.numpy()
        final_losses.append(_bce_loss(pred, y_np))
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
    model_name: str = "unet",
    batch_size: int = 8,
    mask_paths: list[str] | None = None,
    test_mask_format: str = "binary",
    convert_mask_format: str | None = None,
) -> None:
    """Evaluate a saved model (native format: binary) against all labeled data
    (train and val combined).

    Ground truth is normally binary, read from ``./data/{dir}_mask/`` as
    usual. If ``test_mask_format`` is ``'cvat_5'`` instead -- the test set's
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
        test_dataset = get_test_dataset(dirs)
    else:
        from data_loader.cvat import get_test_dataset as get_cvat_test_dataset

        test_dataset = get_cvat_test_dataset(dirs, mask_paths)
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

    model = get_model_class(model_name)()
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
