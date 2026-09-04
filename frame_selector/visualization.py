"""Contact sheet visualization for frame selection results."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from frame_selector.config import SelectorConfig
from frame_selector.metrics import to_grayscale


# BGR colors for status borders
STATUS_COLORS = {
    "selected": (0, 200, 0),
    "quality_rejected": (0, 0, 220),
    "redundant": (0, 180, 220),
    "overlap_rejected": (180, 80, 0),
    "blank_ignored": (140, 0, 140),
}


def make_thumbnail(frame: np.ndarray, max_size: int) -> np.ndarray:
    """Create a small BGR thumbnail preserving aspect ratio."""
    if frame.ndim == 2:
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        display = frame.copy()

    h, w = display.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        display = cv2.resize(
            display,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return display


def generate_contact_sheet(
    entries: list[dict],
    output_path: Path,
    config: SelectorConfig,
) -> None:
    """
    Build a grid contact sheet showing selected vs discarded frames.

    Each cell is a thumbnail with a colored border indicating status and an index label.
    """
    if not entries:
        return

    thumb_size = config.contact_sheet_thumb_size
    cols = config.contact_sheet_cols
    rows = int(np.ceil(len(entries) / cols))

    sample = make_thumbnail(entries[0]["frame"], thumb_size)
    cell_h, cell_w = sample.shape[:2]
    border = 4
    label_h = 18

    sheet_h = rows * (cell_h + label_h + border * 2) + border
    sheet_w = cols * (cell_w + border * 2) + border
    sheet = np.full((sheet_h, sheet_w, 3), 240, dtype=np.uint8)

    for idx, entry in enumerate(entries):
        row, col = divmod(idx, cols)
        y0 = border + row * (cell_h + label_h + border * 2)
        x0 = border + col * (cell_w + border * 2)

        thumb = make_thumbnail(entry["frame"], thumb_size)
        th, tw = thumb.shape[:2]

        color = STATUS_COLORS.get(entry["status"], (128, 128, 128))
        cv2.rectangle(sheet, (x0 - 2, y0 - 2), (x0 + tw + 2, y0 + th + label_h + 2), color, 2)
        sheet[y0 : y0 + th, x0 : x0 + tw] = thumb

        label = f"#{entry['index']} {entry['status'][:4]}"
        cv2.putText(
            sheet,
            label,
            (x0, y0 + th + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    # Legend strip at bottom
    legend_y = sheet_h - 12
    legends = [
        ("Selected", STATUS_COLORS["selected"]),
        ("Quality rej.", STATUS_COLORS["quality_rejected"]),
        ("Redundant", STATUS_COLORS["redundant"]),
        ("Overlapped", STATUS_COLORS["overlap_rejected"]),
    ]
    lx = 10
    for text, color in legends:
        cv2.rectangle(sheet, (lx, legend_y - 10), (lx + 12, legend_y + 2), color, -1)
        cv2.putText(sheet, text, (lx + 16, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        lx += 120

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def save_frame(frame: np.ndarray, path: Path) -> None:
    """Save frame preserving grayscale when appropriate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.ndim == 2:
        cv2.imwrite(str(path), frame)
    else:
        cv2.imwrite(str(path), frame)
