"""Feature-based overlap verification for stitching readiness."""

from __future__ import annotations

import cv2
import numpy as np

from frame_selector.metrics import to_grayscale


def count_good_matches(
    frame1: np.ndarray,
    frame2: np.ndarray,
    max_features: int = 2000,
    ratio_threshold: float = 0.75,
) -> tuple[int, int, int]:
    """
    Count ORB feature matches between two frames using Lowe's ratio test.

    Returns (good_matches, keypoints_in_frame1, keypoints_in_frame2).
    SIFT is used automatically when available in the OpenCV build; otherwise ORB.
    """
    gray1 = to_grayscale(frame1)
    gray2 = to_grayscale(frame2)

    detector = _create_detector(max_features)
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    n1 = len(kp1) if kp1 else 0
    n2 = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or n1 == 0 or n2 == 0:
        return 0, n1, n2

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if _uses_hamming(detector) else cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = 0
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good += 1

    return good, n1, n2


def compute_overlap_metrics(
    frame1: np.ndarray,
    frame2: np.ndarray,
    max_features: int = 2000,
    ratio_threshold: float = 0.75,
) -> dict[str, float]:
    """Feature overlap between two frames — high values indicate redundant content."""
    good, n1, n2 = count_good_matches(frame1, frame2, max_features, ratio_threshold)
    denom = min(n1, n2)
    return {
        "matches": float(good),
        "match_ratio": good / denom if denom > 0 else 0.0,
    }


def is_too_overlapped(
    frame1: np.ndarray,
    frame2: np.ndarray,
    max_matches: int,
    max_match_ratio: float,
    max_features: int = 2000,
    ratio_threshold: float = 0.75,
) -> tuple[bool, dict[str, float]]:
    """
    Return True when two frames share too much feature overlap to both be kept.

    Used during selection to skip frames that have not moved enough since the
    last kept frame, even if pixel-level similarity metrics are borderline.
    """
    metrics = compute_overlap_metrics(frame1, frame2, max_features, ratio_threshold)
    overlapped = (
        metrics["matches"] >= max_matches
        or metrics["match_ratio"] >= max_match_ratio
    )
    return overlapped, metrics


def has_sufficient_overlap(
    frame1: np.ndarray,
    frame2: np.ndarray,
    min_matches: int,
    max_features: int = 2000,
    ratio_threshold: float = 0.75,
) -> tuple[bool, int]:
    """Return whether consecutive selected frames likely share enough overlap for stitching."""
    good, _, _ = count_good_matches(frame1, frame2, max_features, ratio_threshold)
    return good >= min_matches, good


def _create_detector(max_features: int):
    """Prefer SIFT when compiled in; fall back to ORB (always available)."""
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=max_features)
    return cv2.ORB_create(nfeatures=max_features)


def _uses_hamming(detector) -> bool:
    return detector.__class__.__name__ == "ORB"
