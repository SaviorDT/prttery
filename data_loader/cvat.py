"""``--mask-format cvat_6`` support: 6-class masks from CVAT 1.1 XML exports.

Directory convention for source frames is unchanged from ``data_loader.normal``:

    ./data/{dir_name}/          source frames, named "{second}_{frame}.jpg"

There is no ``{dir_name}_mask`` directory for this format. Instead, one or
more CVAT ("CVAT for images 1.1") XML export files are passed via
``--mask-path``. Each ``<image>`` element in those files carries its own
``subset`` attribute (not the same as its ``<task><name>``, which can differ
from it -- confirmed against a real export), and that ``subset`` string is
matched directly against a ``--dirs`` directory name. An image is only used
as labeled training data if its ``name`` attribute also matches a file
present in ``./data/{dir_name}/``.

Shapes are CVAT ``<mask>`` (RLE-encoded, the common case) or ``<polygon>``
(seen mixed into the same images in real exports); both are rasterized into
the same per-pixel class-index canvas. Where multiple shapes in one image
overlap, they're painted in ascending ``z_order`` so the highest ``z_order``
wins -- CVAT's own rendering convention.
"""

from __future__ import annotations

import glob
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader.normal import DATA_ROOT, _list_images


class CvatLabelMap:
    """Fixed label-name -> class-index mapping for the 6-class pottery task.

    Index 0 is always background. ``ALIASES`` folds known extra CVAT project
    labels (that aren't meant to be their own class) into background; any
    other unrecognized label raises immediately -- a real typo/unexpected
    label in the data should fail loudly, not silently corrupt training data.
    """

    BACKGROUND = "background"
    CLASSES = ["background", "pottery", "teapot", "rotate_plate", "rotate_medal", "clay_and_rotator"]

    # CVAT project labels known to show up in exports that should collapse
    # into background rather than error or become a class of their own.
    ALIASES = {"rotate_plate_shader": "background"}

    # Foreground classes for --mode eval's alpha-matte collapse only (see
    # main.py's run_eval). Independent of BACKGROUND/background_index below,
    # which stays a single class (render_mask's canvas fill value) and also
    # backs train/test's fg metrics via mask_formats.Cvat6Format -- changing
    # this list doesn't touch either of those.
    EVAL_FOREGROUND = ["pottery", "teapot", "rotate_plate", "rotate_medal", "clay_and_rotator"]

    def __init__(self) -> None:
        self._name_to_index = {name: i for i, name in enumerate(self.CLASSES)}

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)

    @property
    def background_index(self) -> int:
        return self._name_to_index[self.BACKGROUND]

    def eval_foreground_indices(self) -> set[int]:
        """Class indices EVAL_FOREGROUND names -- see its docstring above."""
        return {self.index_for(name) for name in self.EVAL_FOREGROUND}

    def index_for(self, label: str) -> int:
        if label in self._name_to_index:
            return self._name_to_index[label]
        alias = self.ALIASES.get(label)
        if alias is not None:
            return self._name_to_index[alias]
        raise ValueError(
            f"Unrecognized CVAT label '{label}'. Known classes: {self.CLASSES}; "
            f"known aliases: {list(self.ALIASES)}"
        )


@dataclass(frozen=True)
class _Shape:
    """One rasterizable CVAT shape, stripped of its XML representation so it's
    plain data (safe to hand to DataLoader worker processes, no ties back to
    an ElementTree document)."""

    z_order: int
    kind: str  # "mask" or "polygon"
    label: str
    rle: tuple[int, ...] | None = None
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    points: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class _CvatImageRef:
    xml_path: str
    shapes: tuple[_Shape, ...]
    width: int
    height: int


def _parse_shapes(image_el: ET.Element) -> tuple[_Shape, ...]:
    shapes: list[_Shape] = []

    for mask_el in image_el.findall("mask"):
        rle = tuple(int(x) for x in mask_el.get("rle", "").split(","))
        shapes.append(
            _Shape(
                z_order=int(mask_el.get("z_order", "0")),
                kind="mask",
                label=mask_el.get("label", ""),
                rle=rle,
                left=int(mask_el.get("left", "0")),
                top=int(mask_el.get("top", "0")),
                width=int(mask_el.get("width", "0")),
                height=int(mask_el.get("height", "0")),
            )
        )

    for poly_el in image_el.findall("polygon"):
        points_str = poly_el.get("points", "")
        points = tuple(tuple(map(float, pair.split(","))) for pair in points_str.split(";") if pair)
        shapes.append(
            _Shape(
                z_order=int(poly_el.get("z_order", "0")),
                kind="polygon",
                label=poly_el.get("label", ""),
                points=points,
            )
        )

    # Painter's algorithm: paint in ascending z_order, so the highest
    # z_order ends up on top for any overlapping pixels.
    shapes.sort(key=lambda s: s.z_order)
    return tuple(shapes)


def _decode_rle_into(shape: _Shape, canvas: np.ndarray, class_index: int) -> None:
    """Decode one CVAT <mask> shape's RLE and paint it onto `canvas` in-place.

    CVAT's mask RLE is a flat list of run lengths alternating background(0)/
    foreground(1), starting with background(0), filling the shape's bounding
    box (left, top, width, height) in row-major order. Verified against a
    real export: sum(rle) == width * height exactly.
    """
    total = shape.width * shape.height
    flat = np.empty(total, dtype=bool)
    idx = 0
    value = False
    for run in shape.rle:
        flat[idx : idx + run] = value
        idx += run
        value = not value
    if idx != total:
        raise ValueError(
            f"CVAT mask RLE for label '{shape.label}' decodes to {idx} pixels, expected {total} "
            f"({shape.width}x{shape.height})"
        )
    patch = flat.reshape(shape.height, shape.width)

    bottom, right = shape.top + shape.height, shape.left + shape.width
    if shape.top < 0 or shape.left < 0 or bottom > canvas.shape[0] or right > canvas.shape[1]:
        raise ValueError(
            f"CVAT mask for label '{shape.label}' (left={shape.left}, top={shape.top}, "
            f"width={shape.width}, height={shape.height}) falls outside the "
            f"{canvas.shape[1]}x{canvas.shape[0]} image"
        )

    region = canvas[shape.top : bottom, shape.left : right]
    region[patch] = class_index


def _decode_polygon_into(shape: _Shape, canvas: np.ndarray, class_index: int) -> None:
    """Rasterize one CVAT <polygon> shape onto `canvas` in-place via cv2.fillPoly."""
    pts = np.array(shape.points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], color=int(class_index))


def render_mask(shapes: tuple[_Shape, ...], height: int, width: int, label_map: CvatLabelMap) -> np.ndarray:
    """Render one image's shapes into a (height, width) uint8 class-index mask.

    Every pixel not covered by any shape stays `label_map.background_index`
    (0) -- matching "background(不標記)": background is never its own shape,
    just whatever isn't covered by the other 4 labels' shapes.
    """
    canvas = np.full((height, width), label_map.background_index, dtype=np.uint8)
    for shape in shapes:  # pre-sorted by z_order in _parse_shapes
        class_index = label_map.index_for(shape.label)
        if shape.kind == "mask":
            _decode_rle_into(shape, canvas, class_index)
        else:
            _decode_polygon_into(shape, canvas, class_index)
    return canvas


def _parse_annotations(mask_paths: list[str]) -> dict[str, dict[str, _CvatImageRef]]:
    """Parse every given CVAT XML file into {subset: {image_name: _CvatImageRef}}.

    Duplicate handling:
    - the same subset appearing in more than one file is fine (their image
      lists are merged), but prints a warning since it's unusual;
    - the same (subset, image name) pair appearing in more than one file is
      a real conflict (ambiguous which annotation is authoritative) and
      raises immediately.
    """
    by_subset: dict[str, dict[str, _CvatImageRef]] = {}
    files_seen_for_subset: dict[str, set[str]] = {}

    for xml_path in mask_paths:
        root = ET.parse(xml_path).getroot()
        subsets_in_this_file: set[str] = set()

        for image_el in root.findall("image"):
            subset = image_el.get("subset")
            name = image_el.get("name")
            if not subset or not name:
                raise ValueError(f"{xml_path}: <image> element missing 'subset' or 'name' attribute")

            subsets_in_this_file.add(subset)
            images = by_subset.setdefault(subset, {})
            if name in images:
                raise ValueError(
                    f"Duplicate CVAT annotation for subset='{subset}' image='{name}': "
                    f"present in both '{images[name].xml_path}' and '{xml_path}'"
                )
            images[name] = _CvatImageRef(
                xml_path=xml_path,
                shapes=_parse_shapes(image_el),
                width=int(image_el.get("width", "0")),
                height=int(image_el.get("height", "0")),
            )

        for subset in subsets_in_this_file:
            prior_files = files_seen_for_subset.setdefault(subset, set())
            if prior_files and xml_path not in prior_files:
                print(
                    f"Warning: subset='{subset}' appears in multiple --mask-path files "
                    f"({sorted(prior_files)} and '{xml_path}'); merging their image lists"
                )
            prior_files.add(xml_path)

    return by_subset


def _resolve_dir_names(dir_names: list[str], data_root: str, subsets: set[str]) -> list[str]:
    """Wildcard analogue of data_loader.normal._resolve_dir_names for cvat_6.

    A '*' entry is expanded to every directory under data_root matching the
    pattern that also appears as a subset in the given --mask-path file(s)
    -- the cvat_6 equivalent of requiring a sibling '{name}_mask' directory.
    Non-wildcard entries are passed through unchanged.
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
            if os.path.basename(rel_dir_name) not in subsets:
                continue
            if rel_dir_name not in seen:
                seen.add(rel_dir_name)
                resolved.append(rel_dir_name)

    return resolved


def scan_dirs(
    dir_names: list[str], mask_paths: list[str], data_root: str = DATA_ROOT
) -> list[tuple[str, tuple[_Shape, ...], int, int]]:
    """CVAT-format analogue of data_loader.normal.scan_dirs.

    Returns a list of (image_path, shapes, xml_width, xml_height) -- enough
    for CvatSegmentationDataset to render that frame's mask on demand. Only
    images with a CVAT annotation are returned (mirrors scan_dirs: only
    labeled pairs are used for train/val/test).
    """
    by_subset = _parse_annotations(mask_paths)
    resolved = _resolve_dir_names(dir_names, data_root, set(by_subset))

    labeled: list[tuple[str, tuple[_Shape, ...], int, int]] = []
    for dir_name in resolved:
        image_dir = os.path.join(data_root, dir_name)
        images_on_disk = _list_images(image_dir)

        # `dir_name` may be a full (possibly shell-expanded) path, e.g.
        # "/videos/0626/com.oculus.vrshell-...-0"; the CVAT `subset`
        # attribute only ever holds the last path component, so look up
        # by basename rather than the full dir_name.
        subset_name = os.path.basename(dir_name)
        for name, ref in by_subset.get(subset_name, {}).items():
            stem, _ext = os.path.splitext(name)
            image_path = images_on_disk.get(stem)
            if image_path is None:
                raise FileNotFoundError(
                    f"CVAT annotation for '{name}' (subset='{subset_name}') has no matching image in '{image_dir}'"
                )
            labeled.append((image_path, ref.shapes, ref.width, ref.height))

    return labeled


class CvatSegmentationDataset(Dataset):
    """Loads (image, 6-class mask) pairs, resized to the model's (H, W).

    Mirrors data_loader.normal.SegmentationDataset, but the mask is rendered
    on demand from CVAT shapes instead of read from a mask image file, and
    the target is a (H, W) int64 class-index tensor (not (1, H, W) float in
    {0, 1}) to match UNetResNet18Mul's multi-class NLL-style loss.
    """

    def __init__(
        self,
        triples: list[tuple[str, tuple[_Shape, ...], int, int]],
        image_size: tuple[int, int] = (180, 320),  # (H, W)
        label_map: CvatLabelMap | None = None,
    ) -> None:
        self.triples = triples
        self.height, self.width = image_size
        self.label_map = label_map or CvatLabelMap()

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, index: int):
        image_path, shapes, xml_width, xml_height = self.triples[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # (C, H, W)

        mask = render_mask(shapes, xml_height, xml_width, self.label_map)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))  # (H, W)

        return image_tensor, mask_tensor


def get_train_val_datasets(
    dir_names: list[str],
    mask_paths: list[str],
    val_ratio: float = 0.2,
    seed: int = 42,
    data_root: str = DATA_ROOT,
    image_size: tuple[int, int] = (180, 320),
):
    """Split labeled (image + CVAT annotation) samples into train/val datasets.

    If ``val_ratio`` is ``0`` (or too small to yield any sample), no data is
    held out for validation: ``val_dataset`` is ``None`` and all labeled
    triples are used for training.
    """
    labeled = scan_dirs(dir_names, mask_paths, data_root=data_root)
    if not labeled:
        raise ValueError(
            "No labeled (image + CVAT annotation) pairs found. Check that --mask-path points "
            "at XML file(s) containing a <image subset=...> matching one of --dirs."
        )

    triples = list(labeled)
    random.Random(seed).shuffle(triples)

    val_count = int(len(triples) * val_ratio)
    val_triples = triples[:val_count]
    train_triples = triples[val_count:]
    if not train_triples:
        train_triples = triples

    train_dataset = CvatSegmentationDataset(train_triples, image_size=image_size)
    val_dataset = CvatSegmentationDataset(val_triples, image_size=image_size) if val_triples else None
    return train_dataset, val_dataset


def get_test_dataset(
    dir_names: list[str],
    mask_paths: list[str],
    data_root: str = DATA_ROOT,
    image_size: tuple[int, int] = (180, 320),
):
    """Return a dataset of every labeled (image + CVAT annotation) pair, train and val combined."""
    labeled = scan_dirs(dir_names, mask_paths, data_root=data_root)
    if not labeled:
        raise ValueError(
            "No labeled (image + CVAT annotation) pairs found. Check that --mask-path points "
            "at XML file(s) containing a <image subset=...> matching one of --dirs."
        )
    return CvatSegmentationDataset(labeled, image_size=image_size)
