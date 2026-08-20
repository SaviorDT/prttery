"""Training routine for multi-class (6-class) segmentation models, e.g.
``unet_resnet18_mul``.

Deliberately parallel to (not shared with) ``train.normal``: the two paths
diverge on target shape/dtype, loss, and metrics, and per the design
discussion, keeping them separate avoids entangling future changes to one
with the other (the binary path backs a proven production model).

Two metric sets are reported every epoch:

- the "main" multi-class metrics: per-class IoU (all 5 classes, including
  background), their macro average, and overall pixel accuracy;
- a "fg" (foreground/background collapse) metric set: the argmax prediction
  and target collapsed to non-background=positive / background=negative,
  then the same accuracy/IoU/precision/recall/F1 shape as train.normal's
  binary metrics -- directly comparable to the existing background-removal
  numbers, and the same view --mode eval uses for its alpha-matte output.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.cvat import CvatLabelMap, get_test_dataset, get_train_val_datasets
from mask_formats import binary_confusion_metrics, get_mask_format, to_binary
from models import get_model_class
from train import freezing


def compute_metrics_multiclass(pred_prob: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """Per-class IoU (all `num_classes` classes) + their macro average +
    overall multi-class pixel accuracy.

    pred_prob: (N, C, H, W) probabilities. target: (N, H, W) int class indices.
    """
    pred = np.argmax(pred_prob, axis=1)
    target = target.astype(np.int64)

    eps = 1e-7
    per_class_iou = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = float(np.sum(pred_c & target_c))
        union = float(np.sum(pred_c | target_c))
        per_class_iou.append(intersection / (union + eps))

    metrics = {f"iou_class{c}": iou for c, iou in enumerate(per_class_iou)}
    metrics["mean_iou"] = float(np.mean(per_class_iou))
    metrics["accuracy"] = float(np.mean(pred == target))
    return metrics


def compute_metrics_fg(pred_prob: np.ndarray, target: np.ndarray, background_index: int) -> dict:
    """Collapse the multi-class prediction/target to a binary
    foreground(=non-background)/background view, then compute the same
    accuracy/IoU/precision/recall/F1 as train.normal.compute_metrics --
    directly comparable to the existing binary background-removal numbers.
    """
    pred = (np.argmax(pred_prob, axis=1) != background_index).astype(np.float32)
    tgt = (target != background_index).astype(np.float32)
    return binary_confusion_metrics(pred, tgt)


def _nll_loss(pred_prob: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Standalone (model-independent) version of models.unet_resnet18_mul's
    _MulLoss, usable after model.load() where model.criterion is unavailable
    -- mirrors train.normal._bce_loss."""
    n, _c, h, w = pred_prob.shape
    p = np.clip(pred_prob, eps, 1 - eps)
    idx = target.reshape(n, 1, h, w).astype(np.int64)
    gathered = np.take_along_axis(p, idx, axis=1)
    return float(-np.mean(np.log(gathered)))


def _average_metrics(metric_dicts: list[dict]) -> dict:
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}


def _format_metrics(prefix: str, loss: float | None, metrics: dict) -> str:
    parts = ([f"loss={loss:.4f}"] if loss is not None else []) + [f"{k}={v:.4f}" for k, v in metrics.items()]
    return f"[{prefix}] " + " ".join(parts)


def run(
    dirs: list[str],
    mask_paths: list[str],
    model_name: str = "unet_resnet18_mul",
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
) -> None:
    label_map = CvatLabelMap()

    train_dataset, val_dataset = get_train_val_datasets(dirs, mask_paths, val_ratio=val_ratio, seed=split_seed)
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

    model = get_model_class(model_name)()
    model.create(lr=lr)

    if freeze_encoder:
        freezing.freeze_encoder(model)
        tqdm.write(f"Encoder frozen; will unfreeze after {unfreeze_patience} epoch(s) without val-loss improvement.")
    encoder_frozen = freeze_encoder

    best_val_loss = float("inf")
    best_state = None
    no_improve_count = 0
    lr_no_improve_count = 0
    unfreeze_no_improve_count = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        current_lr = model.optimizer.param_groups[0]["lr"]
        tqdm.write(f"Epoch {epoch}/{epochs} lr={current_lr:.6g}")

        # ---- training ----
        train_losses = []
        train_metrics_main = []
        train_metrics_fg = []
        for x, y in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            loss, pred = model.train(x, y)
            train_losses.append(loss)
            y_np = y.numpy()
            train_metrics_main.append(compute_metrics_multiclass(pred, y_np, label_map.num_classes))
            train_metrics_fg.append(compute_metrics_fg(pred, y_np, label_map.background_index))

        train_loss = float(np.mean(train_losses))
        tqdm.write(f"Epoch {epoch}/{epochs} " + _format_metrics("train", train_loss, _average_metrics(train_metrics_main)))
        tqdm.write(f"Epoch {epoch}/{epochs} " + _format_metrics("train-fg", None, _average_metrics(train_metrics_fg)))

        if not has_val:
            continue

        # ---- validation ----
        val_losses = []
        val_metrics_main = []
        val_metrics_fg = []
        for x, y in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
            pred = model.eval(x)
            y_np = y.numpy()
            val_losses.append(_nll_loss(pred, y_np))
            val_metrics_main.append(compute_metrics_multiclass(pred, y_np, label_map.num_classes))
            val_metrics_fg.append(compute_metrics_fg(pred, y_np, label_map.background_index))

        val_loss = float(np.mean(val_losses))
        tqdm.write(f"Epoch {epoch}/{epochs} " + _format_metrics("val", val_loss, _average_metrics(val_metrics_main)))
        tqdm.write(f"Epoch {epoch}/{epochs} " + _format_metrics("val-fg", None, _average_metrics(val_metrics_fg)))

        # ---- early stopping / lr decay ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.net.state_dict())
            no_improve_count = 0
            lr_no_improve_count = 0
            unfreeze_no_improve_count = 0
            tqdm.write(f"Epoch {epoch}: val loss improved to {val_loss:.4f}, saving best weights")
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
    final_metrics_main = []
    final_metrics_fg = []
    for x, y in final_loader:
        pred = model.eval(x)
        y_np = y.numpy()
        final_losses.append(_nll_loss(pred, y_np))
        final_metrics_main.append(compute_metrics_multiclass(pred, y_np, label_map.num_classes))
        final_metrics_fg.append(compute_metrics_fg(pred, y_np, label_map.background_index))

    final_loss = float(np.mean(final_losses))
    print()
    print("=" * 60)
    print("Final Test (= Val) Metrics" if has_val else "Final Train Metrics")
    print(_format_metrics("test" if has_val else "train", final_loss, _average_metrics(final_metrics_main)))
    print(_format_metrics("test-fg" if has_val else "train-fg", None, _average_metrics(final_metrics_fg)))
    print("=" * 60)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")


def run_test(
    dirs: list[str],
    mask_paths: list[str] | None,
    model_path: str,
    model_name: str = "unet_resnet18_mul",
    batch_size: int = 8,
    test_mask_format: str = "cvat_6",
    convert_mask_format: str | None = None,
) -> None:
    """Evaluate a saved model (native format: cvat_6) against all labeled data
    (train and val combined).

    Ground truth is normally cvat_6, parsed from ``mask_paths`` as usual. If
    ``test_mask_format`` is ``'binary'`` instead -- the test set's ground
    truth is only available as binary masks -- it's loaded via
    ``data_loader.normal`` instead, and both it and the model's (native
    cvat_6) prediction are collapsed into ``convert_mask_format`` (currently
    only ``'binary'`` is supported) before being compared, since the usual
    per-class/foreground metrics below assume cvat_6 ground truth.
    main.py's argument parsing guarantees ``convert_mask_format`` is set
    whenever ``test_mask_format`` differs from this module's native format.
    """
    label_map = CvatLabelMap()
    native_format = "cvat_6"

    if test_mask_format == native_format:
        test_dataset = get_test_dataset(dirs, mask_paths)
    else:
        from data_loader.normal import get_test_dataset as get_binary_test_dataset

        test_dataset = get_binary_test_dataset(dirs)
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
    test_metrics_main = []
    test_metrics_fg = []
    test_metrics_converted = []
    for x, y in tqdm(test_loader, desc="Test"):
        pred = model.eval(x)
        y_np = y.numpy()

        if same_format:
            test_losses.append(_nll_loss(pred, y_np))
            test_metrics_main.append(compute_metrics_multiclass(pred, y_np, label_map.num_classes))
            test_metrics_fg.append(compute_metrics_fg(pred, y_np, label_map.background_index))
        else:
            if convert_mask_format != "binary":
                raise ValueError(f"Unsupported --convert-mask-format '{convert_mask_format}'; only 'binary' is supported")
            # Ground truth doesn't carry per-class labels in this branch (it's
            # binary), so the per-class/mean-IoU breakdown above isn't
            # meaningful here and is skipped -- only the converted binary
            # metrics are reported. Loss is skipped too, for the same reason.
            pred_class_idx = get_mask_format(native_format).raw_output_to_class_index(pred)
            gt_class_idx = get_mask_format(test_mask_format).ground_truth_to_class_index(y_np)
            pred_binary = to_binary(pred_class_idx, native_format)
            gt_binary = to_binary(gt_class_idx, test_mask_format)
            test_metrics_converted.append(binary_confusion_metrics(pred_binary, gt_binary))

    print()
    print("=" * 60)
    print("Test Metrics")
    if same_format:
        test_loss = float(np.mean(test_losses))
        print(_format_metrics("test", test_loss, _average_metrics(test_metrics_main)))
        print(_format_metrics("test-fg", None, _average_metrics(test_metrics_fg)))
    else:
        print(_format_metrics("test", None, _average_metrics(test_metrics_converted)))
    print("=" * 60)
