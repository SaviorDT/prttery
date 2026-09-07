"""Basic data loading utilities for the background-removal task.

Each source-frame directory is paired with a sibling whose name ends in
``_mask``. Run Parameter Paths are already expanded before reaching this
module and are interpreted relative to the current working directory.

Only images that have a matching mask are used as labeled training data; all
images (labeled or not) are candidates for the eval pipeline.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
FRAME_NAME_RE = re.compile(r"^(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class EvalItem:
    image_path: str
    second: int
    frame: int


def _list_images(dir_path: str) -> dict:
    """Return {stem: full_path} for every image file directly inside dir_path."""
    result = {}
    if not os.path.isdir(dir_path):
        return result
    for name in os.listdir(dir_path):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTENSIONS:
            result[stem] = os.path.join(dir_path, name)
    return result


def scan_dirs(dir_names: list[str]):
    """Scan source-frame directories already expanded by RunParams validation.

    Entries that resolve to something other than a genuine source-frame
    directory (a stray file, or a directory whose name itself ends with
    ``"_mask"``) are silently skipped rather than raising, since they can
    show up after Run Parameters expand a Dataset Directory Pattern.

    Returns
    -------
    labeled_pairs: list[(image_path, mask_path)]
        Every image that has a matching mask, used for train/val.
    all_images: dict[str, list[EvalItem]]
        For each dir_name, every image found (labeled or not), used for eval.
    """
    labeled_pairs: list[tuple[str, str]] = []
    all_images: dict[str, list[EvalItem]] = {}

    for dir_name in dir_names:
        image_dir = os.path.normpath(dir_name)
        mask_dir = f"{image_dir}_mask"

        # Expanded Run Parameter Paths can include
        # stray non-directory files (e.g. source .mp4 recordings) and the
        # paired "*_mask" directories themselves; skip anything that isn't a
        # genuine source-frame directory instead of failing the whole scan.
        if os.path.basename(image_dir).endswith("_mask") or not os.path.isdir(image_dir):
            continue

        images = _list_images(image_dir)
        masks = _list_images(mask_dir)

        for stem, mask_path in masks.items():
            image_path = images.get(stem)
            if image_path is None:
                raise FileNotFoundError(
                    f"Mask '{mask_path}' has no matching image in '{image_dir}'"
                )
            labeled_pairs.append((image_path, mask_path))

        items = []
        for stem, image_path in images.items():
            match = FRAME_NAME_RE.match(os.path.basename(image_path))
            if not match:
                continue
            second, frame = int(match.group(1)), int(match.group(2))
            items.append(EvalItem(image_path=image_path, second=second, frame=frame))
        items.sort(key=lambda it: (it.second, it.frame))
        all_images[dir_name] = items

    return labeled_pairs, all_images


class SegmentationDataset(Dataset):
    """Loads (image, mask) pairs, resized to the model's (H, W)."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        image_size: tuple[int, int] = (180, 320),  # (H, W)
    ) -> None:
        self.pairs = pairs
        self.height, self.width = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # (C, H, W)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :])  # (1, H, W)

        return image_tensor, mask_tensor


class EvalDataset(Dataset):
    """Loads eval frames, resized to the model's (H, W), for GPU inference.

    Only returns the small resized tensor plus the ``EvalItem`` (a file path
    and lightweight metadata) - never the full-resolution original image, so
    multi-process ``DataLoader`` workers don't have to ship large arrays
    across the process boundary. The caller is responsible for re-reading
    the original image (via ``item.image_path``) when it needs full-res
    pixels, e.g. for compositing the final output frame.
    """

    def __init__(
        self,
        items: list[EvalItem],
        image_size: tuple[int, int] = (180, 320),  # (H, W)
    ) -> None:
        self.items = items
        self.height, self.width = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]

        image = cv2.imread(item.image_path, cv2.IMREAD_COLOR)
        if image is None:
            # Let the caller skip this frame instead of crashing the whole run.
            return None, item

        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # (C, H, W)

        return image_tensor, item


def eval_collate(batch: list) -> tuple[torch.Tensor | None, list[EvalItem], list[bool]]:
    """Collate ``EvalDataset`` samples, keeping items in their original order.

    Returns ``(stacked_tensor, items, valid_mask)``: ``items``/``valid_mask``
    line up 1:1 with the input order (needed to stay in sync with anything
    else - e.g. a background reader - iterating the same item sequence);
    ``stacked_tensor`` only contains rows for items where ``valid_mask`` is
    True (``None`` if every sample in the batch failed to load).
    """
    items = [item for _, item in batch]
    valid_mask = [tensor is not None for tensor, _ in batch]
    tensors = [tensor for tensor, _ in batch if tensor is not None]
    stacked = torch.stack(tensors, dim=0) if tensors else None
    return stacked, items, valid_mask


def get_train_val_datasets(
    dir_names: list[str],
    val_ratio: float = 0.2,
    seed: int = 42,
    image_size: tuple[int, int] = (180, 320),
):
    """Split labeled (mask-available) samples into train/val datasets.

    If ``val_ratio`` is ``0`` (or too small to yield any sample), no data is
    held out for validation: ``val_dataset`` is ``None`` and all labeled
    pairs are used for training.
    """
    labeled_pairs, _ = scan_dirs(dir_names)
    if not labeled_pairs:
        raise ValueError(
            "No labeled (image + mask) pairs found. Make sure "
            "a sibling {dir}_mask directory contains files matching {dir}."
        )

    pairs = list(labeled_pairs)
    random.Random(seed).shuffle(pairs)

    val_count = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    if not train_pairs:
        train_pairs = pairs

    train_dataset = SegmentationDataset(train_pairs, image_size=image_size)
    val_dataset = SegmentationDataset(val_pairs, image_size=image_size) if val_pairs else None
    return train_dataset, val_dataset


def get_test_dataset(
    dir_names: list[str],
    image_size: tuple[int, int] = (180, 320),
):
    """Return a dataset of every labeled (image + mask) pair, train and val combined."""
    labeled_pairs, _ = scan_dirs(dir_names)
    if not labeled_pairs:
        raise ValueError(
            "No labeled (image + mask) pairs found. Make sure "
            "a sibling {dir}_mask directory contains files matching {dir}."
        )
    return SegmentationDataset(labeled_pairs, image_size=image_size)


def get_eval_items(dir_names: list[str]) -> dict:
    """Return {dir_name: [EvalItem, ...]} sorted by (second, frame)."""
    _, all_images = scan_dirs(dir_names)
    return all_images
