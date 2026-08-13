"""Ad-hoc visualizer for the `copy_paste` preprocessor, kept separate from
the training/eval pipeline in `main.py` and from `preprocessors/copy_paste.py`
itself -- neither is modified by this script.

`preprocessors/copy_paste.py` operates on tensors a train `Dataset.__getitem__`
already returns, i.e. after they've been resized down to the model's input
shape. That's fine for training, but it's the wrong thing to look at when
eyeballing whether the paste itself looks right: the composite ends up in
low-res, blurry input-shape pixels instead of the original frame quality.
This script reimplements the same algorithm (random per-class connected
component, pasted at a random clipped offset -- see `_pick_component` /
`_paste_component` below) directly on original-resolution frames read
straight off disk, so the result is full quality. Hardcoded for a quick
demo: fixed source paths, fixed sample count, no CLI args.

Shape parsing/decoding and mask rendering (RLE masks and polygons, painted in
ascending `z_order`) are reused from `data_loader.cvat` rather than
reimplemented, so interpreting the CVAT XML stays in lockstep with the
training pipeline. Everything else -- scanning `--videos-root` for annotated
frames, the overlay rendering style, and the copy-paste algorithm itself --
is self-contained in this one file (the scanning/overlay code mirrors
`dev/visualize.py`; the copy-paste code mirrors `preprocessors/copy_paste.py`).

Output goes to `./result/gt`, one `cp_{n}.png` per sample: the composited
frame (pasted part included) with its full mask -- original annotation plus
the pasted part, both treated identically, no special highlighting for the
pasted region -- alpha-blended on top, same overlay style as
`dev/visualize.py`. These composites are NOT ground truth; they're synthetic
demo images, just written into `result/gt` anyway for convenience. Existing
files there are overwritten without warning. Every run uses fresh randomness
(no fixed seed), so re-running produces different composites.
"""

from __future__ import annotations

import glob
import os
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

# dev/ is a subdirectory, but `data_loader` lives at the repo root -- add it
# to sys.path so this script runs standalone (`python dev/cp_visualizer.py`)
# regardless of the caller's cwd, without needing `-m` or a PYTHONPATH set.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader.cvat import CvatLabelMap, _Shape, _parse_shapes, render_mask  # noqa: E402

ANNOTATIONS_PATH = "/videos/cvat_masks/annotations.xml"
VIDEOS_ROOT = "/videos"
OUTPUT_DIR = "result/gt"
OVERLAY_ALPHA = 0.5
NUM_SAMPLES = 5

# A connected component smaller than this fraction of its *source frame's*
# own pixel count is skipped as a paste candidate -- same threshold and
# rationale as preprocessors/copy_paste.py's MIN_COMPONENT_AREA_FRACTION,
# just evaluated against the native frame size instead of the model's input
# size, since this script never resizes anything.
MIN_COMPONENT_AREA_FRACTION = 0.005

# How many different random source frames to try before giving up on finding
# one with a usable non-background part.
MAX_SOURCE_ATTEMPTS = 50

_LABEL_MAP = CvatLabelMap()

# Fixed, visually distinct BGR colors (cv2 order) -- identical palette/style
# to dev/visualize.py, so a pasted part renders exactly like a real
# annotation of the same class, with nothing marking it as synthetic.
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


@dataclass(frozen=True)
class _Frame:
    """One annotated frame: where to read its pixels from, and enough to
    render its GT mask on demand (shapes are only rasterized when needed,
    and cached -- see `_mask_for`)."""

    image_path: str
    shapes: tuple[_Shape, ...]
    width: int
    height: int


@dataclass(frozen=True)
class _Component:
    class_index: int  # 1..4, never background (0)
    mask: np.ndarray  # (H, W) bool, True where this component's pixels are


def _find_source_frame(videos_root: str, subset: str, name: str) -> str:
    """Locate the source frame for one CVAT `<image subset=... name=...>`.

    Source frames live under `{videos_root}/{some_top_level_dir}/{subset}/{name}`
    -- which top-level dir varies per subset, so it's found by glob rather
    than assumed. Mirrors dev/visualize.py's `_find_source_frame`.
    """
    pattern = os.path.join(videos_root, "*", subset, name)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No source frame found for subset='{subset}' name='{name}' (pattern: '{pattern}')")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous source frame for subset='{subset}' name='{name}': {matches}")
    return matches[0]


def _load_frames(annotations_path: str, videos_root: str) -> list[_Frame]:
    """Parse every `<image>` in the CVAT XML export into a `_Frame`, resolved
    against an actual file on disk. Mirrors dev/visualize.py's scanning loop."""
    root = ET.parse(annotations_path).getroot()
    image_elements = root.findall("image")

    frames: list[_Frame] = []
    for image_el in tqdm(image_elements, desc="Scanning annotated frames"):
        name = image_el.get("name")
        subset = image_el.get("subset")
        if not name or not subset:
            raise ValueError(f"{annotations_path}: <image> element missing 'name' or 'subset' attribute")

        image_path = _find_source_frame(videos_root, subset, name)
        frames.append(
            _Frame(
                image_path=image_path,
                shapes=_parse_shapes(image_el),
                width=int(image_el.get("width", "0")),
                height=int(image_el.get("height", "0")),
            )
        )
    return frames


def _read_frame(frame: _Frame) -> np.ndarray:
    """Read `frame`'s source pixels at full original resolution and check
    them against the XML's declared size (same consistency check as
    dev/visualize.py's `visualize_gt`)."""
    image = cv2.imread(frame.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read source frame: {frame.image_path}")

    height, width = image.shape[:2]
    if (width, height) != (frame.width, frame.height):
        raise ValueError(
            f"Source frame '{frame.image_path}' is {width}x{height}, but its CVAT annotation "
            f"records {frame.width}x{frame.height}"
        )
    return image


def _mask_for(frames: list[_Frame], cache: dict[int, np.ndarray], index: int) -> np.ndarray:
    """Render (and cache) `frames[index]`'s full GT mask at its native
    resolution -- a (H, W) uint8 class-index array, via
    `data_loader.cvat.render_mask` (same renderer the training pipeline
    uses, just not resized afterward)."""
    if index not in cache:
        frame = frames[index]
        cache[index] = render_mask(frame.shapes, frame.height, frame.width, _LABEL_MAP)
    return cache[index]


def _pick_component(mask: np.ndarray, rng: random.Random) -> _Component | None:
    """Connected-component analysis, computed separately per class value
    present. Randomly picks one *class* present in `mask` (uniformly among
    classes that have at least one large-enough component), then randomly
    picks one of that class's components -- so a class with many small
    components isn't over-represented relative to a class with few, large
    ones. Same algorithm as preprocessors/copy_paste.py's `_pick_component`,
    reimplemented here to run on a plain (H, W) uint8 class-index array at
    native resolution instead of a resized torch tensor.
    """
    height, width = mask.shape
    min_area = MIN_COMPONENT_AREA_FRACTION * height * width

    components_by_class: dict[int, list[np.ndarray]] = {}
    for class_index in np.unique(mask):
        if class_index == 0:  # background
            continue
        binary = (mask == class_index).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        valid = [labels == label for label in range(1, num_labels) if np.sum(labels == label) >= min_area]
        if valid:
            components_by_class[int(class_index)] = valid

    if not components_by_class:
        return None

    class_index = rng.choice(list(components_by_class))
    component_mask = rng.choice(components_by_class[class_index])
    return _Component(class_index=class_index, mask=component_mask)


def _pick_source_component(
    frames: list[_Frame], mask_cache: dict[int, np.ndarray], rng: random.Random
) -> tuple[int, _Component]:
    """Retry random source frames until one has a usable non-background part.
    Same retry strategy as preprocessors/copy_paste.py's
    `_pick_source_component`."""
    for _ in range(MAX_SOURCE_ATTEMPTS):
        source_idx = rng.randrange(len(frames))
        mask = _mask_for(frames, mask_cache, source_idx)
        component = _pick_component(mask, rng)
        if component is not None:
            return source_idx, component
    raise RuntimeError(
        f"cp_visualizer: couldn't find an annotated frame with a usable non-background part "
        f"after {MAX_SOURCE_ATTEMPTS} attempts"
    )


def _paste_component(
    image_src: np.ndarray,
    component: _Component,
    image_tgt: np.ndarray,
    mask_tgt: np.ndarray,
    rng: random.Random,
) -> None:
    """Paste `component`'s pixels from `image_src` onto `image_tgt` (and the
    matching class index into `mask_tgt`) at a random offset, in place.

    Only the component's own (irregularly shaped) pixels are copied, not its
    bounding box -- so `image_tgt`'s own background/other objects show
    through everywhere outside the pasted shape. A random translation may
    push part of the shape outside `image_tgt`'s bounds; that part is simply
    clipped, not resampled, and overlapping existing content in the target is
    overwritten (occlusion is allowed). Same algorithm as
    preprocessors/copy_paste.py's `_paste_component`, reimplemented here for
    plain (H, W, 3) / (H, W) numpy arrays at native resolution -- the offset
    range is bounded by `image_tgt`'s own size (not `image_src`'s), since
    source and target frames aren't guaranteed to share a resolution here the
    way same-shape resized tensors always do in the original.
    """
    target_height, target_width = image_tgt.shape[:2]

    src_ys, src_xs = np.where(component.mask)
    y0, x0 = src_ys.min(), src_xs.min()
    patch_height = src_ys.max() - y0 + 1
    patch_width = src_xs.max() - x0 + 1

    # Random top-left offset for the patch's bounding box, ranging over
    # [-(patch_size - 1), target_size - 1] so a placement near/past any edge
    # (with the rest clipped off) is just as likely as a fully-inside one.
    new_y0 = rng.randint(-(patch_height - 1), target_height - 1)
    new_x0 = rng.randint(-(patch_width - 1), target_width - 1)
    offset_y = new_y0 - y0
    offset_x = new_x0 - x0

    dst_ys = src_ys + offset_y
    dst_xs = src_xs + offset_x
    in_bounds = (dst_ys >= 0) & (dst_ys < target_height) & (dst_xs >= 0) & (dst_xs < target_width)
    src_ys, src_xs = src_ys[in_bounds], src_xs[in_bounds]
    dst_ys, dst_xs = dst_ys[in_bounds], dst_xs[in_bounds]
    if len(src_ys) == 0:
        return  # translated entirely out of frame; leave target untouched

    image_tgt[dst_ys, dst_xs] = image_src[src_ys, src_xs]
    mask_tgt[dst_ys, dst_xs] = component.class_index


def _color_for(label: str, label_colors: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    """Look up (or lazily assign) `label`'s color, so every image in this run
    that has the same label paints it with the same color. Identical to
    dev/visualize.py's `_color_for`."""
    if label not in label_colors:
        label_colors[label] = _PALETTE[len(label_colors) % len(_PALETTE)]
    return label_colors[label]


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    label_colors: dict[str, tuple[int, int, int]],
    alpha: float,
) -> None:
    """Alpha-blend every non-background class present in `mask` onto `image`
    in place, using the same fixed palette/style as dev/visualize.py's
    `_overlay_shapes` -- so the pasted part (just another nonzero mask value
    by this point) renders identically to a real annotation of the same
    class, with no special highlighting."""
    for class_index in range(1, _LABEL_MAP.num_classes):
        region = mask == class_index
        if not region.any():
            continue
        label = _LABEL_MAP.CLASSES[class_index]
        color = np.array(_color_for(label, label_colors), dtype=np.float32)
        blended = image[region].astype(np.float32) * (1 - alpha) + color * alpha
        image[region] = blended.astype(np.uint8)


def _draw_legend(image: np.ndarray, labels: list[str], label_colors: dict[str, tuple[int, int, int]]) -> None:
    """Draw a small color-swatch legend in the top-left corner, listing only
    the labels present in this particular image. Identical to
    dev/visualize.py's `_draw_legend`."""
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


def visualize_copy_paste(
    annotations_path: str = ANNOTATIONS_PATH,
    videos_root: str = VIDEOS_ROOT,
    output_dir: str = OUTPUT_DIR,
    num_samples: int = NUM_SAMPLES,
    alpha: float = OVERLAY_ALPHA,
) -> None:
    """Render `num_samples` copy-paste composites for a quick visual sanity
    check of the `copy_paste` preprocessor's algorithm, at full original
    resolution. Each output is a single file, `{output_dir}/cp_{n}.png`: the
    composited frame (pasted part included) with its full mask -- original
    annotation plus the pasted part -- alpha-blended on top. Existing files
    are overwritten without warning; randomness is fresh every call (no fixed
    seed), so re-running yields different composites.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = _load_frames(annotations_path, videos_root)
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 annotated frames to copy-paste between, found {len(frames)}")

    rng = random.Random()
    mask_cache: dict[int, np.ndarray] = {}
    label_colors: dict[str, tuple[int, int, int]] = {}

    for n in tqdm(range(num_samples), desc="Compositing copy-paste samples"):
        source_idx, component = _pick_source_component(frames, mask_cache, rng)

        target_idx = rng.randrange(len(frames))
        while target_idx == source_idx:
            target_idx = rng.randrange(len(frames))

        image_src = _read_frame(frames[source_idx])
        image_tgt = _read_frame(frames[target_idx])
        mask_tgt = _mask_for(frames, mask_cache, target_idx).copy()

        _paste_component(image_src, component, image_tgt, mask_tgt, rng)

        _overlay_mask(image_tgt, mask_tgt, label_colors, alpha)
        labels_in_image = sorted({_LABEL_MAP.CLASSES[i] for i in np.unique(mask_tgt) if i != 0})
        _draw_legend(image_tgt, labels_in_image, label_colors)

        out_path = os.path.join(output_dir, f"cp_{n}.png")
        cv2.imwrite(out_path, image_tgt)


def main() -> None:
    visualize_copy_paste()


if __name__ == "__main__":
    main()
