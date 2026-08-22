"""Renders a per-pixel predicted-class color overlay on top of an original
image, for --mode eval_mul (see main.py).

Deliberately knows nothing about any specific model, ``--mask-format``, or
``data_loader.cvat.CvatLabelMap`` -- it only consumes plain arrays (an image,
an already-collapsed class-index map, a set of background indices to skip,
and an optional list of class names), the same way ``outputer.mp4`` only
consumes plain BGRA frames. That keeps it usable by any current or future
mask format (currently 'binary' and 'cvat_6') without this module changing:
whatever produces the class-index array (mask_formats.MaskFormat) is the only
place format-specific knowledge lives.
"""

from __future__ import annotations

import cv2
import numpy as np


class ClassOverlayRenderer:
    """Alpha-blends a color per predicted class onto an image, plus an
    optional legend. Tunable via the class attributes below.

    ``PALETTE`` cycles (``class_index % len(PALETTE)``) so it supports any
    number of classes, not just the 6 ``cvat_6`` currently has.
    """

    # How opaque the class-color wash is; 0 = invisible, 1 = solid color.
    ALPHA = 0.5

    # Fixed, visually distinct BGR colors (cv2 order) -- same palette style
    # already used by dev/visualize.py and dev/cp_visualizer.py, reimplemented
    # here independently rather than imported so this module stays decoupled
    # from those ad-hoc dev/ scripts.
    PALETTE: tuple[tuple[int, int, int], ...] = (
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

    # Legend box styling (only used when render() is given class_names).
    LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX
    LEGEND_FONT_SCALE = 0.6
    LEGEND_THICKNESS = 1
    LEGEND_SWATCH = 22
    LEGEND_PAD = 8
    LEGEND_BG_ALPHA = 0.55

    def color_for(self, class_index: int) -> tuple[int, int, int]:
        return self.PALETTE[class_index % len(self.PALETTE)]

    def render(
        self,
        image: np.ndarray,
        class_index: np.ndarray,
        background_classes: set[int],
        class_names: list[str] | None = None,
    ) -> np.ndarray:
        """Return a copy of ``image`` (BGR, uint8) with every non-background
        class in ``class_index`` (same (H, W) shape as ``image``) alpha-blended
        with its color. Background pixels (``class_index`` in
        ``background_classes``) are left untouched.

        If ``class_names`` is given, a legend is drawn listing only the
        classes actually present in this particular image (so it stays
        readable frame-to-frame instead of always listing every class the
        format defines). If ``class_names`` is ``None``, no legend is drawn --
        coloring still happens either way.
        """
        out = image.copy()

        present = [int(c) for c in np.unique(class_index) if int(c) not in background_classes]
        for c in present:
            region = class_index == c
            color = np.array(self.color_for(c), dtype=np.float32)
            blended = out[region].astype(np.float32) * (1 - self.ALPHA) + color * self.ALPHA
            out[region] = blended.astype(np.uint8)

        if class_names is not None:
            legend_entries = [(class_names[c] if c < len(class_names) else f"class {c}", self.color_for(c)) for c in present]
            self._draw_legend(out, legend_entries)

        return out

    def _draw_legend(self, image: np.ndarray, entries: list[tuple[str, tuple[int, int, int]]]) -> None:
        """Draw a small color-swatch legend in the top-left corner, in place."""
        if not entries:
            return

        swatch = self.LEGEND_SWATCH
        pad = self.LEGEND_PAD
        line_height = swatch + 6

        text_widths = [
            cv2.getTextSize(label, self.LEGEND_FONT, self.LEGEND_FONT_SCALE, self.LEGEND_THICKNESS)[0][0]
            for label, _color in entries
        ]
        box_w = pad * 3 + swatch + max(text_widths)
        box_h = pad * 2 + line_height * len(entries)

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.LEGEND_BG_ALPHA, image, 1 - self.LEGEND_BG_ALPHA, 0, image)

        for i, (label, color) in enumerate(entries):
            y0 = pad + i * line_height
            cv2.rectangle(image, (pad, y0), (pad + swatch, y0 + swatch), color, -1)
            cv2.rectangle(image, (pad, y0), (pad + swatch, y0 + swatch), (255, 255, 255), 1)
            cv2.putText(
                image,
                label,
                (pad * 2 + swatch, y0 + swatch - 6),
                self.LEGEND_FONT,
                self.LEGEND_FONT_SCALE,
                (255, 255, 255),
                self.LEGEND_THICKNESS,
                cv2.LINE_AA,
            )
