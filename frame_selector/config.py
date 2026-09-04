"""Configuration dataclass for ultrasound frame selection thresholds."""

from dataclasses import dataclass, field


@dataclass
class SelectorConfig:
    """Tunable parameters for quality filtering, redundancy removal, and overlap checks."""

    # --- Quality thresholds ---
    # Minimum Laplacian variance (absolute floor). Frames below this are considered blurry.
    blur_threshold: float = 30.0
    # Reject frames whose Laplacian variance falls below this percentile of the video (0-100).
    # Adaptive threshold helps avoid rejecting naturally soft ultrasound frames.
    blur_percentile: float = 5.0

    # Maximum allowed noise estimate (std of high-frequency residual). Higher = noisier frame.
    noise_threshold: float = 35.0
    # Reject frames above this noise percentile within the video.
    noise_percentile: float = 95.0

    # Minimum Shannon entropy; only reject extremely flat/uniform frames.
    min_entropy: float = 2.0

    # --- Similarity / redundancy thresholds ---
    # Mean absolute difference (0-255 scale). Below this, frames are nearly identical.
    mad_threshold: float = 3.5
    # SSIM in [0, 1]. Above this, frames are structurally redundant.
    ssim_threshold: float = 0.97
    # Histogram correlation in [-1, 1]. Above this, intensity distributions match closely.
    histogram_threshold: float = 0.98
    # A frame is redundant only when ALL enabled similarity metrics agree (conservative).
    require_all_similarity: bool = True

    # --- Coverage / stitching overlap ---
    # Minimum good ORB matches between consecutive selected frames (stitch readiness).
    min_overlap_matches: int = 25
    # Reject frames whose feature matches with the last kept frame exceed this (too overlapped).
    max_overlap_matches: int = 40
    # Reject when matched-keypoint ratio exceeds this fraction of the smaller keypoint set.
    max_overlap_match_ratio: float = 0.10
    # Use ORB/SIFT overlap to drop redundant frames during selection.
    use_overlap_deduplication: bool = True
    # Lowe ratio test threshold for ORB matching.
    orb_ratio_threshold: float = 0.75
    # Maximum ORB features to detect per frame.
    max_features: int = 2000

    # --- Scan region cropping (remove machine UI borders) ---
    crop_enabled: bool = True
    # Search only the upper fraction of the frame for the fan-shaped scan panel.
    crop_search_fraction: float = 0.55
    # Column is treated as UI if this fraction of pixels are darker than crop_dark_threshold.
    crop_dark_threshold: int = 25
    crop_dark_column_fraction: float = 0.55
    # Apply one fixed crop box to every frame so all outputs share identical dimensions.
    crop_standardize: bool = True
    # How to combine per-frame detections: "median" or "intersection".
    crop_standardize_method: str = "median"

    # --- Blank / no-signal frame filtering ---
    ignore_template_enabled: bool = True
    ignore_template_path: str = "ignore_frame.png"
    # Frame matches template when MAD is below this and SSIM is above ignore_template_ssim.
    ignore_template_mad: float = 12.0
    ignore_template_ssim: float = 0.75

    # --- Processing ---
    # Downscale factor for similarity metrics (speed vs accuracy). 1.0 = full resolution.
    similarity_scale: float = 0.5
    # Thumbnail size for contact sheet (max edge length in pixels).
    contact_sheet_thumb_size: int = 120
    # Contact sheet columns.
    contact_sheet_cols: int = 10

    # Status labels used internally
    STATUS_SELECTED: str = field(default="selected", init=False, repr=False)
    STATUS_QUALITY: str = field(default="quality_rejected", init=False, repr=False)
    STATUS_REDUNDANT: str = field(default="redundant", init=False, repr=False)
    STATUS_OVERLAP: str = field(default="overlap_rejected", init=False, repr=False)
    STATUS_BLANK: str = field(default="blank_ignored", init=False, repr=False)
