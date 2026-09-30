"""Basic data loading utilities for the background-removal task.

Each source-frame directory is paired with a sibling whose name ends in
``_mask``. Run Parameter Paths are already expanded before reaching this
module and are interpreted relative to the current working directory.

Only images that have a matching mask are used as labeled training data; all
images (labeled or not) are candidates for the eval pipeline.

Image inputs accept JPEG (`.jpg`/`.jpeg`) and PNG (`.png`), including
uppercase extensions. They are decoded as BGR with `cv2.IMREAD_COLOR` so
source alpha channels are not part of the model input or eval output.
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
FRAME_NAME_RE = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)
LEGACY_FRAME_NAME_RE = re.compile(r"^(\d+)_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
FILENAME_ORDERED_FPS = 30.0


@dataclass(frozen=True)
class EvalItem:
    image_path: str
    second: int
    frame: int
    output_stem: str | None = None
    fps: float | None = None


def _list_images(dir_path: str, *, reject_duplicate_stems: bool = False) -> dict[str, str]:
    """Return {stem: full_path} for every image file directly inside dir_path."""
    result: dict[str, str] = {}
    stems_by_casefold: dict[str, str] = {}
    if not os.path.isdir(dir_path):
        return result
    for name in os.listdir(dir_path):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTENSIONS:
            image_path = os.path.join(dir_path, name)
            if reject_duplicate_stems:
                folded_stem = stem.casefold()
                previous_path = stems_by_casefold.get(folded_stem)
                if previous_path is not None:
                    raise ValueError(
                        f"Duplicate image stem in '{dir_path}': "
                        f"'{previous_path}' and '{image_path}'"
                    )
                stems_by_casefold[folded_stem] = image_path
            result[stem] = image_path
    return result


def scan_dirs(dir_names: list[str], *, validate_unique_stems: bool = False):
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
        For each dir_name, images are represented as legacy time-based frames
        when possible, otherwise sorted by filename with a fixed FPS. Only
        non-empty image groups are included.
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

        images = _list_images(image_dir, reject_duplicate_stems=validate_unique_stems)
        masks = _list_images(mask_dir)

        for stem, mask_path in masks.items():
            image_path = images.get(stem)
            if image_path is None:
                raise FileNotFoundError(
                    f"Mask '{mask_path}' has no matching image in '{image_dir}'"
                )
            labeled_pairs.append((image_path, mask_path))

        image_paths = [
            path
            for path in images.values()
            if FRAME_NAME_RE.search(os.path.basename(path))
        ]
        legacy_matches = [
            LEGACY_FRAME_NAME_RE.match(os.path.basename(path))
            for path in image_paths
        ]
        if image_paths and all(match is not None for match in legacy_matches):
            items = [
                EvalItem(
                    image_path=path,
                    second=int(match.group(1)),
                    frame=int(match.group(2)),
                    output_stem=f"{int(match.group(1))}_{int(match.group(2))}",
                )
                for path, match in zip(image_paths, legacy_matches)
            ]
            items.sort(key=lambda it: (it.second, it.frame))
        else:
            image_paths.sort(
                key=lambda path: (os.path.basename(path).casefold(), os.path.basename(path))
            )
            items = [
                EvalItem(
                    image_path=path,
                    second=0,
                    frame=index,
                    output_stem=os.path.splitext(os.path.basename(path))[0],
                    fps=FILENAME_ORDERED_FPS,
                )
                for index, path in enumerate(image_paths)
            ]
        if validate_unique_stems:
            output_stems: dict[str, str] = {}
            for item in items:
                output_stem = item.output_stem or f"{item.second}_{item.frame}"
                folded_stem = output_stem.casefold()
                previous_path = output_stems.get(folded_stem)
                if previous_path is not None:
                    raise ValueError(
                        f"Duplicate eval output stem in '{image_dir}': "
                        f"'{previous_path}' and '{item.image_path}'"
                    )
                output_stems[folded_stem] = item.image_path
        if items:
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

    def image_path(self, index: int) -> str:
        """Return the source image for a labeled sample."""
        return self.pairs[index][0]

    def native_target(self, index: int) -> np.ndarray:
        """Load a binary target without resizing it to the model input."""
        _image_path, mask_path = self.pairs[index]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        return (mask > 127).astype(np.float32)[np.newaxis, :, :]

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
        strict: bool = False,
    ) -> None:
        self.items = items
        self.height, self.width = image_size
        self.strict = strict

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]

        image = cv2.imread(item.image_path, cv2.IMREAD_COLOR)
        if image is None:
            if self.strict:
                raise FileNotFoundError(f"Failed to read image: {item.image_path}")
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
    """Return eval items, validating output stem uniqueness for eval only."""
    _, all_images = scan_dirs(dir_names, validate_unique_stems=True)
    return all_images
