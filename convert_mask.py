#!/usr/bin/env python3
"""Convert CVAT 6-class annotations into black-and-white JPG masks.

Execution settings live in param.py. Each CVAT image becomes one
three-channel 8-bit JPG at output_dir/subset/name_stem.jpg.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from data_loader.cvat import CvatLabelMap, _parse_shapes, render_mask


def _validate_subset(subset: str, xml_path: Path) -> None:
    """Reject subset values that could escape the configured output folder."""
    if not subset or subset in {".", ".."} or "/" in subset or "\\" in subset or "\x00" in subset:
        raise ValueError(
            f"{xml_path}: unsafe CVAT subset {subset!r}; expected one directory name"
        )


def _parse_positive_int(image_el: ET.Element, attribute: str, xml_path: Path, name: str) -> int:
    value = image_el.get(attribute)
    if value is None:
        raise ValueError(f"{xml_path}: image {name!r} is missing '{attribute}'")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            f"{xml_path}: image {name!r} has invalid {attribute}={value!r}"
        ) from error
    if parsed <= 0:
        raise ValueError(
            f"{xml_path}: image {name!r} requires positive {attribute}, got {parsed}"
        )
    return parsed


def _iter_rendered_annotations(
    xml_path: Path, label_map: CvatLabelMap
) -> list[tuple[str, str, np.ndarray]]:
    """Parse and render every image in one XML file in document order."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"{xml_path}: invalid XML: {error}") from error

    rendered: list[tuple[str, str, np.ndarray]] = []
    for image_el in root.findall("image"):
        subset = image_el.get("subset")
        name = image_el.get("name")
        if not subset or not name:
            raise ValueError(f"{xml_path}: <image> is missing 'subset' or 'name'")
        _validate_subset(subset, xml_path)

        stem = Path(name).stem
        if not stem or stem in {".", ".."}:
            raise ValueError(f"{xml_path}: image name {name!r} has no usable filename stem")

        width = _parse_positive_int(image_el, "width", xml_path, name)
        height = _parse_positive_int(image_el, "height", xml_path, name)
        try:
            # Use the existing CVAT rasterization path, including RLE,
            # polygons, labels, and z-order painter semantics.
            shapes = _parse_shapes(image_el)
            class_mask = render_mask(shapes, height, width, label_map)
        except (ValueError, TypeError, cv2.error) as error:
            raise ValueError(
                f"{xml_path}: failed to render image {name!r}: {error}"
            ) from error
        rendered.append((subset, stem, class_mask))

    return rendered


def _to_black_and_white_bgr(class_mask: np.ndarray, foreground_indices: set[int]) -> np.ndarray:
    """Return a three-channel uint8 mask containing only black or white pixels."""
    foreground = np.isin(class_mask, tuple(foreground_indices))
    grayscale = np.where(foreground, 255, 0).astype(np.uint8)
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def _write_jpeg(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jpeg_parameters = [cv2.IMWRITE_JPEG_QUALITY, 100]
    sampling_parameter = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
    sampling_444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", 0x111111)
    if sampling_parameter is not None:
        jpeg_parameters.extend((sampling_parameter, sampling_444))

    if not cv2.imwrite(str(output_path), image, jpeg_parameters):
        raise OSError(f"could not write JPG: {output_path}")


def convert_mask(
    mask_paths: tuple[Path, ...],
    output_dir: Path,
    foreground_classes: tuple[str, ...],
) -> tuple[int, bool]:
    """Convert all configured XML files and return (count, had_duplicates)."""
    label_map = CvatLabelMap()
    invalid_classes = sorted(set(foreground_classes) - set(label_map.CLASSES))
    if invalid_classes:
        raise ValueError(
            f"unknown foreground class(es) {invalid_classes}; "
            f"known classes: {label_map.CLASSES}"
        )
    foreground_indices = {
        label_map.index_for(class_name) for class_name in foreground_classes
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, str], str] = {}
    converted = 0
    had_duplicates = False

    for xml_path in mask_paths:
        for subset, stem, class_mask in _iter_rendered_annotations(xml_path, label_map):
            key = (subset, stem)
            source = f"{xml_path} ({stem})"
            previous = seen.get(key)
            if previous is not None:
                had_duplicates = True
                print(
                    f"Error: duplicate annotation for subset={subset!r}, stem={stem!r}; "
                    f"overwriting {previous} with {source}",
                    file=sys.stderr,
                )
            seen[key] = source

            output_path = output_dir / subset / f"{stem}.jpg"
            _write_jpeg(
                _to_black_and_white_bgr(class_mask, foreground_indices),
                output_path,
            )
            converted += 1

    return converted, had_duplicates


def main() -> int:
    from param import ACTIVE_CONVERT_MASK

    params = ACTIVE_CONVERT_MASK
    try:
        params.validate()
        converted, had_duplicates = convert_mask(
            params.mask_paths,
            params.output_dir,
            params.foreground_classes,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Done: converted {converted} annotation(s) to {params.output_dir}")
    return 1 if had_duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())

