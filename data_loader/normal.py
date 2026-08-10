"""Basic data loading utilities for the background-removal task.

Directory convention (relative to ``./data``):

    ./data/{dir_name}/          source frames, named "{second}_{frame}.jpg"
    ./data/{dir_name}_mask/     ground-truth masks (white=foreground=1,
                                 black=background=0), same file stem as the
                                 matching source frame.

Every file present in a ``{dir_name}_mask`` folder is guaranteed (by the task
spec) to have a same-name counterpart in ``{dir_name}``. Only images that
have a mask are used as labeled training data; *all* images (labeled or not)
are candidates for the eval pipeline.
"""

from __future__ import annotations

import glob
import os
import random
import re
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_ROOT = "./data"
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


def _resolve_dir_names(dir_names: list[str], data_root: str) -> list[str]:
    """Expand ``*`` wildcard entries in ``dir_names`` into concrete dir names.

    A wildcard pattern is matched against directories under ``data_root``.
    A match is only kept if its last path component does *not* end with
    ``_mask``, and a sibling directory with the same parent path whose last
    component is ``"{name}_mask"`` also exists. Non-wildcard entries are
    passed through unchanged (their mask directory is not required to
    exist).
    """
    resolved: list[str] = []
    seen: set[str] = set()

    for pattern in dir_names:
        if "*" not in pattern:
            if pattern not in seen:
                seen.add(pattern)
                resolved.append(pattern)
            continue

        for match in sorted(glob.glob(os.path.join(data_root, pattern))):
            if not os.path.isdir(match):
                continue

            rel_dir_name = os.path.relpath(match, data_root)
            basename = os.path.basename(rel_dir_name)
            if basename.endswith("_mask"):
                continue

            mask_dir = os.path.join(os.path.dirname(match), f"{basename}_mask")
            if not os.path.isdir(mask_dir):
                continue

            if rel_dir_name not in seen:
                seen.add(rel_dir_name)
                resolved.append(rel_dir_name)

    return resolved


def scan_dirs(dir_names: list[str], data_root: str = DATA_ROOT):
    """Scan the given directories under ``data_root``.

    ``dir_names`` entries may contain ``*`` wildcards. A wildcard entry is
    expanded to every directory under ``data_root`` matching the pattern
    whose last path component has a sibling ``"{name}_mask"`` directory
    (same parent path); directories without such a pair are skipped.

    Entries that resolve to something other than a genuine source-frame
    directory (a stray file, or a directory whose name itself ends with
    ``"_mask"``) are silently skipped rather than raising, since they can
    show up when the shell expands a ``*`` glob before Python ever sees it.

    Returns
    -------
    labeled_pairs: list[(image_path, mask_path)]
        Every image that has a matching mask, used for train/val.
    all_images: dict[str, list[EvalItem]]
        For each dir_name, every image found (labeled or not), used for eval.
    """
    labeled_pairs: list[tuple[str, str]] = []
    all_images: dict[str, list[EvalItem]] = {}

    for dir_name in _resolve_dir_names(dir_names, data_root):
        image_dir = os.path.join(data_root, dir_name)
        mask_dir = os.path.join(data_root, f"{dir_name}_mask")

        # dir_names may come from a shell-expanded "*" glob, which can include
        # stray non-directory files (e.g. source .mp4 recordings) and the
        # paired "*_mask" directories themselves; skip anything that isn't a
        # genuine source-frame directory instead of failing the whole scan.
        if os.path.basename(dir_name).endswith("_mask") or not os.path.isdir(image_dir):
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
    data_root: str = DATA_ROOT,
    image_size: tuple[int, int] = (180, 320),
):
    """Split labeled (mask-available) samples into train/val datasets.

    If ``val_ratio`` is ``0`` (or too small to yield any sample), no data is
    held out for validation: ``val_dataset`` is ``None`` and all labeled
    pairs are used for training.
    """
    labeled_pairs, _ = scan_dirs(dir_names, data_root=data_root)
    if not labeled_pairs:
        raise ValueError(
            "No labeled (image + mask) pairs found. Make sure "
            "./data/{dir}_mask contains files matching ./data/{dir}."
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
    data_root: str = DATA_ROOT,
    image_size: tuple[int, int] = (180, 320),
):
    """Return a dataset of every labeled (image + mask) pair, train and val combined."""
    labeled_pairs, _ = scan_dirs(dir_names, data_root=data_root)
    if not labeled_pairs:
        raise ValueError(
            "No labeled (image + mask) pairs found. Make sure "
            "./data/{dir}_mask contains files matching ./data/{dir}."
        )
    return SegmentationDataset(labeled_pairs, image_size=image_size)


def get_eval_items(dir_names: list[str], data_root: str = DATA_ROOT) -> dict:
    """Return {dir_name: [EvalItem, ...]} sorted by (second, frame)."""
    _, all_images = scan_dirs(dir_names, data_root=data_root)
    return all_images
