"""End-to-end ultrasound video pipeline: extract, select, and stitch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from frame_selector.config import SelectorConfig
from frame_selector.selector import FrameRecord, SelectionStats, UltrasoundFrameSelector
from frame_selector.stitcher import StitchConfig, StitchResult, UltrasoundPanoramaStitcher


@dataclass
class PipelineResult:
    """Combined output from frame selection and panorama stitching."""

    records: list[FrameRecord]
    selection_stats: SelectionStats
    stitch_result: StitchResult | None = None


class UltrasoundPipeline:
    """
    Full pipeline for upper-trapezius ultrasound panorama reconstruction:

      1. Extract & crop video frames (remove UI, skip blanks)
      2. Select unique high-quality frames
      3. Stitch selected frames into a horizontal panorama
    """

    PANORAMA_FILENAME = "panorama.png"
    SELECTED_DIR_NAME = "selected_frames"

    def __init__(
        self,
        selector_config: SelectorConfig | None = None,
        stitch_config: StitchConfig | None = None,
        *,
        run_stitch: bool = True,
    ):
        self.selector_config = selector_config or SelectorConfig()
        self.stitch_config = stitch_config or StitchConfig()
        self.run_stitch = run_stitch
        self.selector = UltrasoundFrameSelector(self.selector_config)
        self.stitcher = UltrasoundPanoramaStitcher(self.stitch_config)

    def run(
        self,
        video_path: Path,
        output_dir: Path,
        save_contact_sheet: bool = True,
    ) -> tuple[PipelineResult, Path | None]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        records, stats = self.selector.process(
            video_path,
            output_dir,
            save_contact_sheet=save_contact_sheet,
        )

        panorama_path: Path | None = None
        stitch_result: StitchResult | None = None

        if self.run_stitch:
            selected_dir = output_dir / self.SELECTED_DIR_NAME
            print("\n=== Panorama Stitching ===")
            stitch_result = self.stitcher.stitch_directory(selected_dir)
            panorama_path = output_dir / self.PANORAMA_FILENAME
            cv2.imwrite(str(panorama_path), stitch_result.panorama)
            print(f"Input selected frames:  {stitch_result.num_input_frames}")
            print(f"Frames stitched:        {stitch_result.num_used_frames}")
            print(f"Used frame indices:     {stitch_result.used_indices}")
            print(f"Panorama size:          {stitch_result.panorama.shape[1]}x{stitch_result.panorama.shape[0]}")
            print(f"Panorama saved to:      {panorama_path}")

        return PipelineResult(records, stats, stitch_result), panorama_path
