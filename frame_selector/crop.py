"""Detect and crop the ultrasound scan region, removing machine UI borders."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from frame_selector.metrics import to_grayscale


@dataclass
class CropBox:
    """Pixel bounding box (inclusive) of the scan region."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    def as_slice(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1 + 1), slice(self.x0, self.x1 + 1)


def detect_scan_roi(
    frame: np.ndarray,
    search_fraction: float = 0.55,
    tissue_min: int = 20,
    tissue_max: int = 245,
    row_top_fraction: float = 0.25,
    row_bottom_fraction: float = 0.12,
    dark_threshold: int = 25,
    dark_column_fraction: float = 0.55,
) -> CropBox:
    """
    Locate the B-mode scan panel and exclude surrounding machine UI.

    Strategy (designed for fixed-layout ultrasound recorders):
      1. Restrict search to the upper portion of the frame where the fan scan lives.
      2. Find vertical extent from row tissue occupancy (ignores bottom text/toolbar).
      3. Trim left/right by walking inward from the edges until columns are mostly
         tissue rather than black UI bars or the right-hand parameter sidebar.
    """
    gray = to_grayscale(frame)
    height, width = gray.shape
    search_rows = max(1, int(height * search_fraction))
    upper = gray[:search_rows]

    tissue_row_counts = ((upper > tissue_min) & (upper < tissue_max)).sum(axis=1) / width
    top_rows = np.where(tissue_row_counts > row_top_fraction)[0]
    bottom_rows = np.where(tissue_row_counts > row_bottom_fraction)[0]
    if len(top_rows) == 0 or len(bottom_rows) == 0:
        return CropBox(0, 0, width - 1, height - 1)

    y0 = int(top_rows[0])
    y1 = int(bottom_rows[-1])
    band = gray[y0 : y1 + 1]
    band_height = band.shape[0]
    # Ignore tapered fan edges when measuring column occupancy.
    core = band[band_height // 10 : 9 * band_height // 10]

    x1 = width - 1
    for x in range(width - 1, -1, -1):
        if (core[:, x] < dark_threshold).mean() < dark_column_fraction:
            x1 = x
            break

    x0 = 0
    for x in range(width):
        if (core[:, x] < dark_threshold).mean() < dark_column_fraction:
            x0 = x
            break

    return CropBox(x0, y0, x1, y1)


def clamp_crop_box(box: CropBox, width: int, height: int) -> CropBox:
    """Ensure crop coordinates lie within frame bounds."""
    x0 = max(0, min(box.x0, width - 1))
    y0 = max(0, min(box.y0, height - 1))
    x1 = max(x0, min(box.x1, width - 1))
    y1 = max(y0, min(box.y1, height - 1))
    return CropBox(x0, y0, x1, y1)


def standardize_crop_box(
    boxes: list[CropBox],
    width: int,
    height: int,
    method: str = "median",
) -> CropBox:
    """
    Derive a single crop rectangle applied to every frame.

    median:       median of each edge — stable for fixed-layout ultrasound UI.
    intersection: largest box contained in all detections — strictest, same size guaranteed
                  to be valid in every frame.
    """
    if not boxes:
        return CropBox(0, 0, width - 1, height - 1)

    if method == "intersection":
        x0 = max(b.x0 for b in boxes)
        y0 = max(b.y0 for b in boxes)
        x1 = min(b.x1 for b in boxes)
        y1 = min(b.y1 for b in boxes)
        if x1 > x0 and y1 > y0:
            return clamp_crop_box(CropBox(x0, y0, x1, y1), width, height)

    x0 = int(np.median([b.x0 for b in boxes]))
    y0 = int(np.median([b.y0 for b in boxes]))
    x1 = int(np.median([b.x1 for b in boxes]))
    y1 = int(np.median([b.y1 for b in boxes]))
    return clamp_crop_box(CropBox(x0, y0, x1, y1), width, height)


def _detect_kwargs_from_config(config) -> dict:
    return {
        "search_fraction": config.crop_search_fraction,
        "dark_threshold": config.crop_dark_threshold,
        "dark_column_fraction": config.crop_dark_column_fraction,
    }


def crop_scan_region(frame: np.ndarray, box: CropBox | None = None, **detect_kwargs) -> tuple[np.ndarray, CropBox]:
    """Return the cropped scan image and the box used."""
    if box is None:
        box = detect_scan_roi(frame, **detect_kwargs)
    ys, xs = box.as_slice()
    return frame[ys, xs].copy(), box
