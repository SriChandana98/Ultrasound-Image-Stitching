"""Horizontal panorama stitching for ultrasound frame sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from frame_selector.metrics import to_grayscale
from frame_selector.overlap import _create_detector, _uses_hamming


@dataclass
class StitchConfig:
    """Parameters for pruning, alignment, and blending."""

    # Drop consecutive frames that shift less than this (pixels) before stitching.
    min_advance_pixels: float = 15.0
    # Strip width as a fraction of frame width for template matching.
    strip_fraction: float = 0.30
    # Minimum template-match confidence to trust template shift over flow.
    min_template_peak: float = 0.45
    # Width (px) of linear feathering in overlap regions.
    blend_feather: int = 40
    # ORB/SIFT features for affine fallback.
    max_features: int = 3000
    match_ratio: float = 0.7
    # Try OpenCV Stitcher first when frame count is small enough.
    prefer_opencv_stitcher: bool = False
    opencv_stitcher_max_frames: int = 15


@dataclass
class StitchResult:
    """Output of the stitching pipeline."""

    panorama: np.ndarray
    used_indices: list[int]
    cumulative_shifts: list[float]
    source_paths: list[Path] = field(default_factory=list)
    all_paths: list[Path] = field(default_factory=list)

    @property
    def num_input_frames(self) -> int:
        return len(self.all_paths) if self.all_paths else len(self.source_paths)

    @property
    def num_used_frames(self) -> int:
        return len(self.used_indices)


@dataclass
class PairwiseShift:
    dx: float
    dy: float
    confidence: float
    method: str


def load_frames(input_dir: Path) -> tuple[list[Path], list[np.ndarray]]:
    """Load selected frames in chronological order."""
    paths = sorted(Path(input_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {input_dir}")
    images = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        images.append(img)
    return paths, images


class UltrasoundPanoramaStitcher:
    """
    Build a horizontal landscape panorama from an ultrasound sweep.

    Pipeline:
      1. Prune frames that add too little horizontal displacement vs. the last kept frame.
      2. Estimate pairwise horizontal shift using optical flow + template matching,
         with feature-based affine RANSAC as a fallback.
      3. Accumulate shifts and blend frames onto a wide canvas with feathered overlaps.
    """

    def __init__(self, config: StitchConfig | None = None):
        self.config = config or StitchConfig()

    def stitch_directory(self, input_dir: Path) -> StitchResult:
        paths, images = load_frames(input_dir)
        return self.stitch(images, paths)

    def stitch(
        self,
        images: list[np.ndarray],
        paths: list[Path] | None = None,
    ) -> StitchResult:
        if len(images) == 1:
            return StitchResult(
                panorama=images[0].copy(),
                used_indices=[0],
                cumulative_shifts=[0.0],
                source_paths=paths or [],
                all_paths=paths or [],
            )

        kept = self._prune_indices([to_grayscale(im) for im in images])
        pruned_images = [images[i] for i in kept]
        pruned_paths = [paths[i] for i in kept] if paths else []

        if (
            self.config.prefer_opencv_stitcher
            and len(pruned_images) <= self.config.opencv_stitcher_max_frames
        ):
            pano = self._try_opencv_stitcher(pruned_images)
            if pano is not None:
                return StitchResult(
                    panorama=pano,
                    used_indices=kept,
                    cumulative_shifts=self._positions_from_shifts(pruned_images),
                    source_paths=pruned_paths,
                    all_paths=paths or [],
                )

        positions = self._compute_positions(pruned_images)
        panorama = self._blend_panorama(pruned_images, positions)
        return StitchResult(
            panorama=panorama,
            used_indices=kept,
            cumulative_shifts=positions,
            source_paths=pruned_paths,
            all_paths=paths or [],
        )

    def _prune_indices(self, grays: list[np.ndarray]) -> list[int]:
        """Keep frames separated by enough horizontal motion for a useful panorama."""
        cfg = self.config
        kept = [0]
        for i in range(1, len(grays)):
            shift = self._estimate_shift(grays[kept[-1]], grays[i])
            if shift.dx >= cfg.min_advance_pixels:
                kept.append(i)
        # Always include the final frame for full sweep coverage.
        if kept[-1] != len(grays) - 1:
            kept.append(len(grays) - 1)
        return kept

    def _compute_positions(self, images: list[np.ndarray]) -> list[float]:
        positions = [0.0]
        for i in range(len(images) - 1):
            shift = self._estimate_shift(
                to_grayscale(images[i]),
                to_grayscale(images[i + 1]),
            )
            positions.append(positions[-1] + shift.dx)
        return positions

    def _positions_from_shifts(self, images: list[np.ndarray]) -> list[float]:
        return self._compute_positions(images)

    def _estimate_shift(self, gray_a: np.ndarray, gray_b: np.ndarray) -> PairwiseShift:
        """Estimate horizontal displacement of b relative to a."""
        cfg = self.config
        h, w = gray_a.shape[:2]
        tw = max(30, int(w * cfg.strip_fraction))

        # Optical flow (dense, works on speckle).
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mask = gray_a > 20
        if mask.sum() > 100:
            dx_flow = float(-np.median(flow[..., 0][mask]))
            dy_flow = float(-np.median(flow[..., 1][mask]))
        else:
            dx_flow, dy_flow = 0.0, 0.0

        # Template: right strip of a appears in b.
        strip = gray_a[:, w - tw :]
        response = cv2.matchTemplate(gray_b, strip, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(response)
        dx_tmpl = float((w - tw) - loc[0])

        # Feature affine fallback.
        dx_feat, dy_feat, feat_conf = self._feature_shift(gray_a, gray_b)

        if peak >= cfg.min_template_peak and abs(dx_tmpl) >= 1.0:
            return PairwiseShift(dx=max(dx_tmpl, 0.0), dy=0.0, confidence=peak, method="template")
        if abs(dx_flow) >= 2.0:
            return PairwiseShift(dx=max(dx_flow, 0.0), dy=dy_flow, confidence=0.5, method="flow")
        if feat_conf > 0 and dx_feat is not None:
            return PairwiseShift(dx=max(dx_feat, 0.0), dy=dy_feat or 0.0, confidence=feat_conf, method="features")
        return PairwiseShift(dx=max(dx_tmpl, 0.0), dy=0.0, confidence=peak, method="template_fallback")

    def _feature_shift(
        self,
        gray_a: np.ndarray,
        gray_b: np.ndarray,
    ) -> tuple[float | None, float | None, float]:
        detector = _create_detector(self.config.max_features)
        kp1, des1 = detector.detectAndCompute(gray_a, None)
        kp2, des2 = detector.detectAndCompute(gray_b, None)
        if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
            return None, None, 0.0

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING if _uses_hamming(detector) else cv2.NORM_L2,
        )
        pairs = matcher.knnMatch(des1, des2, k=2)
        pts1, pts2 = [], []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.match_ratio * n.distance:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)

        if len(pts1) < 6:
            return None, None, 0.0

        affine, inliers = cv2.estimateAffinePartial2D(
            np.float32(pts2),
            np.float32(pts1),
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )
        if affine is None or inliers is None:
            return None, None, 0.0

        inlier_count = int(inliers.sum())
        confidence = inlier_count / len(pts1)
        return float(affine[0, 2]), float(affine[1, 2]), confidence

    def _blend_panorama(
        self,
        images: list[np.ndarray],
        positions: list[float],
    ) -> np.ndarray:
        """Place frames on a canvas with feathered linear blending in overlaps."""
        h, w = images[0].shape[:2]
        feather = min(self.config.blend_feather, max(10, w // 8))
        canvas_w = int(max(positions) + w + 10)
        accum = np.zeros((h, canvas_w, 3), dtype=np.float32)
        weight = np.zeros((h, canvas_w), dtype=np.float32)

        ramp = np.ones(w, dtype=np.float32)
        if feather > 0:
            fade = np.linspace(0.0, 1.0, feather, dtype=np.float32)
            ramp[:feather] *= fade
            ramp[-feather:] *= fade[::-1]

        for img, x0f in zip(images, positions):
            x0 = int(round(x0f))
            x1 = x0 + w
            w2d = np.tile(ramp, (h, 1))
            accum[:, x0:x1] += img.astype(np.float32) * w2d[:, :, None]
            weight[:, x0:x1] += w2d

        weight = np.maximum(weight, 1e-6)
        pano = (accum / weight[:, :, None]).clip(0, 255).astype(np.uint8)

        # Trim empty margins.
        col_energy = weight.sum(axis=0)
        valid = np.where(col_energy > 0.01)[0]
        if len(valid) == 0:
            return pano
        return pano[:, valid[0] : valid[-1] + 1]

    def _try_opencv_stitcher(self, images: list[np.ndarray]) -> np.ndarray | None:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        status, pano = stitcher.stitch(images)
        if status != cv2.Stitcher_OK:
            return None
        return pano
