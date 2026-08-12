"""``copy_paste`` preprocessor: randomly copies a non-background part (a
connected component of one class) from one train sample onto another,
pasting only the part's own pixels (not its bounding box), to synthesize
additional train samples.

Operates directly on the tensors a train ``Dataset.__getitem__`` already
returns -- i.e. after resizing/normalization -- so it needs no knowledge of
``--mask-format`` or file layout, just whichever mask convention the
underlying dataset uses:

- binary (``data_loader.normal.SegmentationDataset``): mask is ``(1, H, W)``
  float32 in {0, 1}.
- cvat_5 (``data_loader.cvat.CvatSegmentationDataset``): mask is ``(H, W)``
  int64 class index.

In both formats, pixel value 0 is always background (binary: background=0
by definition; cvat_5: "background" is always ``CvatLabelMap.CLASSES[0]``,
i.e. index 0), so any nonzero mask value is treated as a paste-able "part"
without needing to know which dataset class produced it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessors.base import PreprocessorBase

# A connected component smaller than this fraction of the image's total
# pixel count is skipped as a paste candidate -- too small to be a
# meaningful augmentation, and often just resize/threshold noise.
MIN_COMPONENT_AREA_FRACTION = 0.005

# How many different random source samples to try before giving up on
# generating one synthetic sample (only hit if most/all train images have
# no non-background pixels at all).
MAX_SOURCE_ATTEMPTS = 50


@dataclass(frozen=True)
class _Component:
    class_value: float | int  # source mask's own dtype value for this part
    mask: np.ndarray  # (H, W) bool, True where this component's pixels are


class CopyPastePreprocessor(PreprocessorBase):
    def __init__(self, count: int, seed: int) -> None:
        self.count = count
        self.seed = seed

    def apply(self, train_dataset: Dataset) -> Dataset:
        n = len(train_dataset)
        if n == 0:
            raise ValueError("copy_paste preprocessor requires a non-empty train dataset")

        rng = random.Random(self.seed)
        synthetic_samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for _ in range(self.count):
            source_idx, component = self._pick_source_component(train_dataset, n, rng)

            target_idx = rng.randrange(n)
            if n > 1:
                while target_idx == source_idx:
                    target_idx = rng.randrange(n)

            image_src, _mask_src = train_dataset[source_idx]
            image_tgt, mask_tgt = train_dataset[target_idx]
            image_tgt = image_tgt.clone()
            mask_tgt = mask_tgt.clone()

            _paste_component(image_src, component, image_tgt, mask_tgt, rng)
            synthetic_samples.append((image_tgt, mask_tgt))

        return _ConcatWithSynthetic(train_dataset, synthetic_samples)

    def _pick_source_component(self, train_dataset: Dataset, n: int, rng: random.Random) -> tuple[int, _Component]:
        """Retry random source indices until one has a usable non-background part."""
        for _ in range(MAX_SOURCE_ATTEMPTS):
            source_idx = rng.randrange(n)
            _, mask_src = train_dataset[source_idx]
            component = _pick_component(mask_src, rng)
            if component is not None:
                return source_idx, component
        raise RuntimeError(
            f"copy_paste: couldn't find a train sample with a usable non-background part "
            f"after {MAX_SOURCE_ATTEMPTS} attempts"
        )


def _mask_to_2d(mask: torch.Tensor) -> np.ndarray:
    """(1, H, W) float or (H, W) int64 -> (H, W) numpy array, same values."""
    array = mask.numpy()
    return array[0] if array.ndim == 3 else array


def _pick_component(mask: torch.Tensor, rng: random.Random) -> _Component | None:
    """Connected-component analysis, computed separately per class value present.

    Randomly picks one *class* present in ``mask`` (uniformly among classes
    that have at least one large-enough component), then randomly picks one
    of that class's components -- so a class with many small components
    isn't over-represented relative to a class with few, large ones.
    """
    mask_2d = _mask_to_2d(mask)
    height, width = mask_2d.shape
    min_area = MIN_COMPONENT_AREA_FRACTION * height * width

    components_by_class: dict[float, list[np.ndarray]] = {}
    for class_value in np.unique(mask_2d):
        if class_value == 0:  # background
            continue
        binary = (mask_2d == class_value).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        valid = [labels == label for label in range(1, num_labels) if np.sum(labels == label) >= min_area]
        if valid:
            components_by_class[class_value] = valid

    if not components_by_class:
        return None

    class_value = rng.choice(list(components_by_class))
    component_mask = rng.choice(components_by_class[class_value])
    return _Component(class_value=class_value, mask=component_mask)


def _paste_component(
    image_src: torch.Tensor,
    component: _Component,
    image_tgt: torch.Tensor,
    mask_tgt: torch.Tensor,
    rng: random.Random,
) -> None:
    """Paste ``component``'s pixels from ``image_src`` onto ``image_tgt`` (and
    the matching value into ``mask_tgt``) at a random offset, in place.

    Only the component's own (irregularly shaped) pixels are copied, not its
    bounding box -- so ``image_tgt``'s own background/other objects show
    through everywhere outside the pasted shape. A random translation may
    push part of the shape outside ``image_tgt``'s bounds; that part is
    simply clipped, not resampled, and overlapping existing content in the
    target is overwritten (occlusion is allowed).
    """
    height, width = component.mask.shape
    src_ys, src_xs = np.where(component.mask)
    y0, x0 = src_ys.min(), src_xs.min()
    patch_height = src_ys.max() - y0 + 1
    patch_width = src_xs.max() - x0 + 1

    # Random top-left offset for the patch's bounding box, ranging over
    # [-(patch_size - 1), image_size - 1] so a placement near/past any edge
    # (with the rest clipped off) is just as likely as a fully-inside one.
    new_y0 = rng.randint(-(patch_height - 1), height - 1)
    new_x0 = rng.randint(-(patch_width - 1), width - 1)
    offset_y = new_y0 - y0
    offset_x = new_x0 - x0

    dst_ys = src_ys + offset_y
    dst_xs = src_xs + offset_x
    in_bounds = (dst_ys >= 0) & (dst_ys < height) & (dst_xs >= 0) & (dst_xs < width)
    src_ys, src_xs = src_ys[in_bounds], src_xs[in_bounds]
    dst_ys, dst_xs = dst_ys[in_bounds], dst_xs[in_bounds]
    if len(src_ys) == 0:
        return  # translated entirely out of frame; leave target untouched

    src_ys_t, src_xs_t = torch.from_numpy(src_ys).long(), torch.from_numpy(src_xs).long()
    dst_ys_t, dst_xs_t = torch.from_numpy(dst_ys).long(), torch.from_numpy(dst_xs).long()

    image_tgt[:, dst_ys_t, dst_xs_t] = image_src[:, src_ys_t, src_xs_t]

    if mask_tgt.dim() == 3:  # binary: (1, H, W) float32
        mask_tgt[0, dst_ys_t, dst_xs_t] = float(component.class_value)
    else:  # cvat_5: (H, W) int64 class index
        mask_tgt[dst_ys_t, dst_xs_t] = int(component.class_value)


class _ConcatWithSynthetic(Dataset):
    """Wraps ``base_dataset`` with a fixed list of extra (image, mask) tensor
    pairs appended after it. ``base_dataset``'s own samples are returned
    unchanged; every epoch sees the same synthetic samples (generated once,
    up front, not regenerated per access)."""

    def __init__(self, base_dataset: Dataset, synthetic_samples: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.base_dataset = base_dataset
        self.synthetic_samples = synthetic_samples

    def __len__(self) -> int:
        return len(self.base_dataset) + len(self.synthetic_samples)

    def __getitem__(self, index: int):
        base_len = len(self.base_dataset)
        if index < base_len:
            return self.base_dataset[index]
        return self.synthetic_samples[index - base_len]
