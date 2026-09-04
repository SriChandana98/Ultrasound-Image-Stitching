"""Ultrasound video frame selection for image stitching."""

from frame_selector.blank_detector import BlankFrameDetector, build_blank_detector, resolve_template_path
from frame_selector.config import SelectorConfig
from frame_selector.crop import CropBox, crop_scan_region, detect_scan_roi
from frame_selector.extractor import ExtractStats, VideoFrameExtractor
from frame_selector.selector import UltrasoundFrameSelector, SelectionStats, FrameRecord

from frame_selector.pipeline import UltrasoundPipeline, PipelineResult
from frame_selector.stitcher import StitchConfig, UltrasoundPanoramaStitcher, StitchResult

__all__ = [
    "SelectorConfig",
    "UltrasoundFrameSelector",
    "SelectionStats",
    "FrameRecord",
    "VideoFrameExtractor",
    "ExtractStats",
    "BlankFrameDetector",
    "StitchConfig",
    "UltrasoundPanoramaStitcher",
    "StitchResult",
    "UltrasoundPipeline",
    "PipelineResult",
    "CropBox",
    "crop_scan_region",
    "detect_scan_roi",
]
