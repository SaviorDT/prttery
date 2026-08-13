"""Ad-hoc visualization utilities for eyeballing data, kept separate from the
training/eval pipeline in ``main.py``.

Currently only one visualization exists: ``visualize_gt`` renders every
ground-truth annotation in a CVAT ("CVAT for images 1.1") export as a colored
overlay on top of its source frame, so mislabeled or misaligned annotations
are easy to spot by eye. ``main()`` is kept to a single call so that adding a
second visualization later (e.g. rendering model predictions instead of
ground truth) only means adding a new function and changing that one call --
nothing else in this module needs to change.

Shape parsing/decoding (RLE masks and polygons, painted in ascending
``z_order``) is reused from ``data_loader.cvat`` rather than reimplemented
here, so this stays in lockstep with however the training pipeline
interprets the same XML.
"""

from __future__ import annotations

import glob
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm

# dev/ is a subdirectory, but `data_loader` lives at the repo root -- add it
# to sys.path so this script runs standalone (`python dev/visualize.py`)
# regardless of the caller's cwd, without needing `-m` or a PYTHONPATH set.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.cvat import _Shape, _decode_polygon_into, _decode_rle_into, _parse_shapes  # noqa: E402

ANNOTATIONS_PATH = "/videos/cvat_masks/annotations.xml"
VIDEOS_ROOT = "/videos"
OUTPUT_DIR = "result/gt"
OVERLAY_ALPHA = 0.5

# Fixed, visually distinct BGR colors (cv2 order). Assigned to labels in the
# order they're first encountered, so which color a label gets can change
# between runs, but stays the same for every image within one run -- that's
# all that's needed to compare classes across the output images.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),  # red
    (0, 200, 0),  # green
    (255, 0, 0),  # blue
    (0, 220, 220),  # yellow
    (220, 0, 220),  # magenta
    (220, 220, 0),  # cyan
    (0, 140, 255),  # orange
    (200, 0, 120),  # purple
    (60, 255, 60),  # lime
    (180, 105, 255),  # pink
)


def _color_for(label: str, label_colors: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    """Look up (or lazily assign) `label`'s color, so every image in this run
    that has the same label paints it with the same color."""
    if label not in label_colors:
        label_colors[label] = _PALETTE[len(label_colors) % len(_PALETTE)]
    return label_colors[label]


def _find_source_frame(videos_root: str, subset: str, name: str) -> str:
    """Locate the source frame for one CVAT `<image subset=... name=...>`.

    Source frames live under `{videos_root}/{some_top_level_dir}/{subset}/{name}`
    -- which top-level dir varies per subset, so it's found by glob rather
    than assumed.
    """
    pattern = os.path.join(videos_root, "*", subset, name)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No source frame found for subset='{subset}' name='{name}' (pattern: '{pattern}')")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous source frame for subset='{subset}' name='{name}': {matches}")
    return matches[0]


def _shape_mask(shape: _Shape, height: int, width: int) -> np.ndarray:
    """Decode one CVAT shape into a `(height, width)` boolean mask, reusing
    `data_loader.cvat`'s RLE/polygon decoders (which paint an int class-index
    onto a canvas) by painting `1` onto a zeroed canvas and reading it back
    as a boolean mask."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    if shape.kind == "mask":
        _decode_rle_into(shape, canvas, 1)
    else:
        _decode_polygon_into(shape, canvas, 1)
    return canvas.astype(bool)


def _overlay_shapes(
    image: np.ndarray,
    shapes: tuple[_Shape, ...],
    label_colors: dict[str, tuple[int, int, int]],
    alpha: float,
) -> None:
    """Alpha-blend every shape onto `image` in place, in ascending `z_order`
    (already sorted by `_parse_shapes`) so higher `z_order` shapes end up on
    top for overlapping pixels -- CVAT's own rendering convention."""
    height, width = image.shape[:2]
    for shape in shapes:
        mask = _shape_mask(shape, height, width)
        color = np.array(_color_for(shape.label, label_colors), dtype=np.float32)
        blended = image[mask].astype(np.float32) * (1 - alpha) + color * alpha
        image[mask] = blended.astype(np.uint8)


def _draw_legend(image: np.ndarray, labels: list[str], label_colors: dict[str, tuple[int, int, int]]) -> None:
    """Draw a small color-swatch legend in the top-left corner, listing only
    the labels present in this particular image."""
    if not labels:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    swatch = 22
    pad = 8
    line_height = swatch + 6

    text_widths = [cv2.getTextSize(label, font, font_scale, thickness)[0][0] for label in labels]
    box_w = pad * 3 + swatch + max(text_widths)
    box_h = pad * 2 + line_height * len(labels)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for i, label in enumerate(labels):
        y0 = pad + i * line_height
        color = label_colors[label]
        cv2.rectangle(image, (pad, y0), (pad + swatch, y0 + swatch), color, -1)
        cv2.rectangle(image, (pad, y0), (pad + swatch, y0 + swatch), (255, 255, 255), 1)
        cv2.putText(
            image,
            label,
            (pad * 2 + swatch, y0 + swatch - 6),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def visualize_gt(
    annotations_path: str = ANNOTATIONS_PATH,
    videos_root: str = VIDEOS_ROOT,
    output_dir: str = OUTPUT_DIR,
    alpha: float = OVERLAY_ALPHA,
) -> None:
    """Render every `<image>` in a CVAT XML export as a colored GT overlay.

    One output file per `<image>` element (not per shape): all of that
    image's shapes are painted onto its source frame together, so classes
    can be compared both within an image and across images (same label ->
    same color for the whole run). Written to
    `{output_dir}/{subset}__{stem}.png` -- `subset` is prefixed because the
    same file name (e.g. "4_0.jpg") recurs across different subsets in a
    real export, and would otherwise silently overwrite output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    root = ET.parse(annotations_path).getroot()
    image_elements = root.findall("image")

    label_colors: dict[str, tuple[int, int, int]] = {}

    for image_el in tqdm(image_elements, desc="Rendering GT"):
        name = image_el.get("name")
        subset = image_el.get("subset")
        if not name or not subset:
            raise ValueError(f"{annotations_path}: <image> element missing 'name' or 'subset' attribute")
        xml_width = int(image_el.get("width", "0"))
        xml_height = int(image_el.get("height", "0"))

        source_path = _find_source_frame(videos_root, subset, name)
        image = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read source frame: {source_path}")

        height, width = image.shape[:2]
        if (width, height) != (xml_width, xml_height):
            raise ValueError(
                f"Source frame '{source_path}' is {width}x{height}, but {annotations_path} records "
                f"{xml_width}x{xml_height} for subset='{subset}' name='{name}'"
            )

        shapes = _parse_shapes(image_el)
        _overlay_shapes(image, shapes, label_colors, alpha)

        labels_in_image = sorted({shape.label for shape in shapes})
        _draw_legend(image, labels_in_image, label_colors)

        stem, _ext = os.path.splitext(name)
        out_path = os.path.join(output_dir, f"{subset}__{stem}.png")
        cv2.imwrite(out_path, image)


def main() -> None:
    visualize_gt()


if __name__ == "__main__":
    main()
