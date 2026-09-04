"""Detect blank / no-signal ultrasound frames using a reference template."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from frame_selector.config import SelectorConfig
from frame_selector.metrics import (
    mean_absolute_difference,
    ssim_grayscale,
    to_grayscale,
)


def resolve_template_path(path: str) -> Path | None:
    """Resolve a template path from CWD or the project root."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    project_root = Path(__file__).resolve().parent.parent
    bundled = project_root / path
    if bundled.is_file():
        return bundled
    return None


def build_blank_detector(config: SelectorConfig) -> BlankFrameDetector:
    """Create a blank detector from config, with path resolution and warnings."""
    template_path = None
    if config.ignore_template_enabled and config.ignore_template_path:
        template_path = resolve_template_path(config.ignore_template_path)
        if template_path is None:
            print(
                f"Warning: blank template not found at '{config.ignore_template_path}' "
                "— blank-frame filtering disabled."
            )
    return BlankFrameDetector(
        template_path,
        mad_threshold=config.ignore_template_mad,
        ssim_threshold=config.ignore_template_ssim,
        enabled=config.ignore_template_enabled,
    )


class BlankFrameDetector:
    """
    Match frames against a blank ultrasound template and reject no-signal content.

    Blank frames (probe off tissue, between sweeps) are nearly black with only
    faint top-edge artifacts. They match the template closely in both pixel
    difference and structural similarity.
    """

    def __init__(
        self,
        template_path: Path | None,
        mad_threshold: float = 12.0,
        ssim_threshold: float = 0.75,
        enabled: bool = True,
    ):
        self.enabled = enabled and template_path is not None
        self.mad_threshold = mad_threshold
        self.ssim_threshold = ssim_threshold
        self._template_gray: np.ndarray | None = None

        if self.enabled and template_path is not None:
            template_path = Path(template_path)
            if not template_path.is_file():
                raise FileNotFoundError(f"Blank frame template not found: {template_path}")
            template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
            if template is None:
                raise ValueError(f"Could not read blank frame template: {template_path}")
            self._template_gray = to_grayscale(template)

    @property
    def template_shape(self) -> tuple[int, int] | None:
        if self._template_gray is None:
            return None
        h, w = self._template_gray.shape[:2]
        return w, h

    def is_blank(self, frame: np.ndarray) -> tuple[bool, dict[str, float]]:
        """Return True when the frame matches the blank template."""
        if not self.enabled or self._template_gray is None:
            return False, {}

        gray = to_grayscale(frame)
        template = self._template_gray
        if gray.shape != template.shape:
            template = cv2.resize(
                template,
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        scores = {
            "template_mad": mean_absolute_difference(gray, template),
            "template_ssim": ssim_grayscale(gray, template),
        }
        blank = (
            scores["template_mad"] < self.mad_threshold
            and scores["template_ssim"] > self.ssim_threshold
        )
        return blank, scores
