"""Image quality and frame-similarity metrics for ultrasound video frames."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale frame to single-channel uint8."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def laplacian_variance(gray: np.ndarray) -> float:
    """
    Variance of the Laplacian — higher values indicate sharper edges/texture.

    Ultrasound speckle contributes baseline variance; adaptive percentile thresholds
    (configured in SelectorConfig) prevent over-rejection of soft but usable frames.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def noise_estimate(gray: np.ndarray) -> float:
    """
    Estimate noise as the standard deviation of the high-frequency residual.

    A mild Gaussian blur removes structure; the remainder approximates speckle/noise.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(np.std(residual))


def shannon_entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the intensity histogram — detects blank or saturated frames."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-np.sum(prob * np.log2(prob)))


def compute_quality_metrics(frame: np.ndarray) -> dict[str, float]:
    """Compute all quality metrics for a single frame."""
    gray = to_grayscale(frame)
    return {
        "laplacian_var": laplacian_variance(gray),
        "noise": noise_estimate(gray),
        "entropy": shannon_entropy(gray),
    }


def passes_quality(
    metrics: dict[str, float],
    adaptive: dict[str, float],
    blur_threshold: float,
    noise_threshold: float,
    min_entropy: float,
) -> tuple[bool, str]:
    """
    Decide whether a frame meets quality criteria.

    Uses both absolute thresholds and adaptive (percentile-based) cutoffs so that
    naturally low-contrast ultrasound frames are not discarded solely for contrast.
    """
    if metrics["laplacian_var"] < max(blur_threshold, adaptive["blur_cutoff"]):
        return False, "blur"
    if metrics["noise"] > min(noise_threshold, adaptive["noise_cutoff"]):
        return False, "noise"
    if metrics["entropy"] < min_entropy:
        return False, "entropy"
    return True, "ok"


def resize_for_comparison(gray: np.ndarray, scale: float) -> np.ndarray:
    """Downscale grayscale image for faster similarity computation."""
    if scale >= 1.0:
        return gray
    h, w = gray.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def mean_absolute_difference(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean absolute pixel difference on aligned grayscale images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))


def ssim_grayscale(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index for grayscale images (single scale, global).

    Implemented with NumPy to avoid extra dependencies beyond OpenCV.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean())


def histogram_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Pearson correlation between normalized intensity histograms."""
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


def is_redundant(
    frame: np.ndarray,
    reference: np.ndarray,
    mad_threshold: float,
    ssim_threshold: float,
    histogram_threshold: float,
    similarity_scale: float,
    require_all: bool,
) -> tuple[bool, dict[str, float]]:
    """
    Determine if `frame` adds little new information compared to `reference`.

    For ultrasound probe sweeps, consecutive frames are often similar; we only
    mark redundancy when similarity is very high (small probe displacement).
    """
    g1 = resize_for_comparison(to_grayscale(frame), similarity_scale)
    g2 = resize_for_comparison(to_grayscale(reference), similarity_scale)

    scores = {
        "mad": mean_absolute_difference(g1, g2),
        "ssim": ssim_grayscale(g1, g2),
        "histogram_corr": histogram_correlation(g1, g2),
    }

    mad_similar = scores["mad"] < mad_threshold
    ssim_similar = scores["ssim"] > ssim_threshold
    hist_similar = scores["histogram_corr"] > histogram_threshold

    if require_all:
        redundant = mad_similar and ssim_similar and hist_similar
    else:
        redundant = mad_similar or ssim_similar or hist_similar

    return redundant, scores
