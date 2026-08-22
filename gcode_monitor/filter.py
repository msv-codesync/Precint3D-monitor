"""Drop boxes that sit on the bed / sheet, and demand the same class twice."""

from __future__ import annotations

from collections import deque

from PIL import Image

from gcode_monitor.mask import box_on_mask

# Warping on a textured bed is the usual false alarm. Ask more of those classes.
MIN_CONF = {
    "warping": 0.42,
    "stringing": 0.28,
    "spaghetti": 0.32,
    "spaghetti_failure": 0.32,
    "good_print": 0.45,
    "blob_or_zit": 0.22,
    "zits_blobs": 0.22,
    "under_extrusion": 0.22,
    "layer_shift": 0.30,
    "first_layer_failure": 0.28,
    "nozzle_clog": 0.28,
}


class StickyClass:
    def __init__(self, need: int = 3):
        self.need = max(1, int(need))
        self.recent: deque[str] = deque(maxlen=self.need)

    def push(self, name: str) -> tuple[str, bool]:
        self.recent.append(name)
        if len(self.recent) < self.need:
            return name, False
        return name, all(item == name for item in self.recent)


def keep_box(name: str, conf: float, xyxy, mask: Image.Image | None, *, frame_h: int | None = None) -> bool:
    floor = MIN_CONF.get(name, 0.18)
    if conf < floor:
        return False
    # Bed clips and sheet stripes sit on the frame edge (same failure Obico documents).
    if frame_h and name in {"warping", "stringing"}:
        x1, y1, x2, y2 = xyxy
        cy = (y1 + y2) / 2
        if cy > 0.88 * frame_h or cy < 0.06 * frame_h:
            return False
    if mask is None:
        return True
    if name in {"good_print"}:
        return box_on_mask(mask, xyxy, min_frac=0.15)
    return box_on_mask(mask, xyxy, min_frac=0.30)
