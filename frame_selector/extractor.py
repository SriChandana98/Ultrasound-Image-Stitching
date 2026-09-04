"""Extract frames from ultrasound video and crop scan region immediately."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from frame_selector.blank_detector import build_blank_detector
from frame_selector.config import SelectorConfig
from frame_selector.crop import (
    CropBox,
    _detect_kwargs_from_config,
    crop_scan_region,
    detect_scan_roi,
    standardize_crop_box,
)
from frame_selector.visualization import save_frame


@dataclass
class ExtractStats:
    """Summary of the video extraction stage."""

    total_video_frames: int = 0
    blank_ignored: int = 0
    total_frames: int = 0
    output_dir: Path | None = None
    crop_box: CropBox | None = None
    ignored_indices: list[int] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Frame Extraction ===",
            f"Total video frames:           {self.total_video_frames}",
            f"Ignored (blank/no-signal):    {self.blank_ignored}",
            f"Saved cropped frames:         {self.total_frames}",
            f"Cropped frames saved to:      {self.output_dir}",
        ]
        if self.crop_box is not None:
            lines.append(
                f"Standardized crop size:       "
                f"{self.crop_box.width}x{self.crop_box.height} "
                f"(x={self.crop_box.x0}-{self.crop_box.x1}, y={self.crop_box.y0}-{self.crop_box.y1})"
            )
        return "\n".join(lines)


class VideoFrameExtractor:
    """
    Stage 1: read video frames one at a time, crop UI borders immediately,
    skip blank/no-signal frames via template matching, and persist the rest.

    When crop_standardize is enabled, a single crop box is computed from the
    whole video and applied to every frame so all outputs share identical size.
    """

    def __init__(self, config: SelectorConfig | None = None):
        self.config = config or SelectorConfig()
        self._standard_crop_box: CropBox | None = None
        self.blank_detector = build_blank_detector(self.config)

    def extract_to_directory(self, video_path: Path, output_dir: Path) -> ExtractStats:
        """
        Convert the video into cropped PNG frames on disk.

        Pass 1 (when standardizing): detect ROI on each frame and compute one fixed box.
        Pass 2: decode video, crop each frame, skip blanks, save the rest.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        if output_dir.exists():
            for old_frame in output_dir.glob("frame_*.png"):
                old_frame.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

        detect_kwargs = _detect_kwargs_from_config(self.config)
        crop_box = self._resolve_crop_box(video_path, detect_kwargs)

        stats = ExtractStats(output_dir=output_dir, crop_box=crop_box)
        reference_shape: tuple[int, int] | None = None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                stats.total_video_frames += 1
                cropped = self._crop_frame(frame, crop_box, detect_kwargs)

                is_blank, _ = self.blank_detector.is_blank(cropped)
                if is_blank:
                    stats.blank_ignored += 1
                    stats.ignored_indices.append(frame_idx)
                    frame_idx += 1
                    continue

                if reference_shape is None:
                    reference_shape = (cropped.shape[1], cropped.shape[0])
                elif (cropped.shape[1], cropped.shape[0]) != reference_shape:
                    raise RuntimeError(
                        f"Frame {frame_idx} crop size {cropped.shape[1]}x{cropped.shape[0]} "
                        f"differs from standard {reference_shape[0]}x{reference_shape[1]}"
                    )

                save_frame(cropped, output_dir / f"frame_{frame_idx:05d}.png")
                stats.total_frames += 1
                frame_idx += 1
        finally:
            cap.release()

        return stats

    def iter_saved_frames(self, cropped_dir: Path) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over previously saved cropped frames in chronological order."""
        cropped_dir = Path(cropped_dir)
        for path in sorted(cropped_dir.glob("frame_*.png")):
            frame_idx = int(path.stem.split("_")[1])
            frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                continue
            yield frame_idx, frame

    def _resolve_crop_box(self, video_path: Path, detect_kwargs: dict) -> CropBox | None:
        """Return the fixed crop box for this video, or None when cropping is disabled."""
        if not self.config.crop_enabled:
            return None

        if not self.config.crop_standardize:
            return None

        return self._compute_standard_crop_box(video_path, detect_kwargs)

    def _compute_standard_crop_box(self, video_path: Path, detect_kwargs: dict) -> CropBox:
        """Scan the video once to derive one crop rectangle for all frames."""
        boxes: list[CropBox] = []
        frame_shape: tuple[int, int] | None = None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_shape is None:
                    frame_shape = (frame.shape[1], frame.shape[0])
                boxes.append(detect_scan_roi(frame, **detect_kwargs))
        finally:
            cap.release()

        if frame_shape is None or not boxes:
            raise ValueError(f"No frames found in video: {video_path}")

        return standardize_crop_box(
            boxes,
            frame_shape[0],
            frame_shape[1],
            method=self.config.crop_standardize_method,
        )

    def _crop_frame(
        self,
        frame: np.ndarray,
        crop_box: CropBox | None,
        detect_kwargs: dict,
    ) -> np.ndarray:
        if not self.config.crop_enabled:
            return frame.copy()
        cropped, _ = crop_scan_region(frame, box=crop_box, **detect_kwargs)
        return cropped
