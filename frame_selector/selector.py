"""Main ultrasound frame selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from frame_selector.config import SelectorConfig
from frame_selector.extractor import VideoFrameExtractor
from frame_selector.metrics import (
    compute_quality_metrics,
    is_redundant,
    passes_quality,
)
from frame_selector.overlap import compute_overlap_metrics, has_sufficient_overlap, is_too_overlapped
from frame_selector.visualization import generate_contact_sheet, make_thumbnail, save_frame


@dataclass
class FrameRecord:
    """Metadata for a single video frame after processing."""

    index: int
    status: str
    quality_reason: str = ""
    similarity_scores: dict = field(default_factory=dict)
    overlap_matches: int = 0


@dataclass
class SelectionStats:
    """Summary statistics printed after processing."""

    total_frames: int = 0
    quality_rejected: int = 0
    redundant_rejected: int = 0
    overlap_rejected: int = 0
    selected: int = 0
    overlap_warnings: int = 0

    def summary(self) -> str:
        lines = [
            "=== Frame Selection Statistics ===",
            f"Total cropped frames:         {self.total_frames}",
            f"Removed (poor quality):       {self.quality_rejected}",
            f"Removed (redundant/similar):  {self.redundant_rejected}",
            f"Removed (overlapped):         {self.overlap_rejected}",
            f"Final selected frames:        {self.selected}",
            f"Overlap warnings:             {self.overlap_warnings}",
        ]
        return "\n".join(lines)


class UltrasoundFrameSelector:
    """
    Two-stage ultrasound frame pipeline:

      Stage 1 — Extract & crop: decode video frames, crop UI borders, skip blank
                frames via template matching, save the rest to ``cropped_frames/``.
      Stage 2 — Select unique: run quality filtering, redundancy removal, and
                overlap checks on the saved cropped frames only.
    """

    CROPPED_DIR_NAME = "cropped_frames"
    SELECTED_DIR_NAME = "selected_frames"

    def __init__(self, config: SelectorConfig | None = None):
        self.config = config or SelectorConfig()
        self.extractor = VideoFrameExtractor(self.config)

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        save_contact_sheet: bool = True,
    ) -> tuple[list[FrameRecord], SelectionStats]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        cropped_dir = output_dir / self.CROPPED_DIR_NAME
        selected_dir = output_dir / self.SELECTED_DIR_NAME
        selected_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: video → cropped frames (crop happens at extraction time).
        extract_stats = self.extractor.extract_to_directory(video_path, cropped_dir)
        print(extract_stats.summary())

        # Stage 2: quality + similarity selection on cropped frames only.
        adaptive = self._compute_adaptive_thresholds(cropped_dir)
        records, stats = self._select_unique_frames(
            cropped_dir, selected_dir, adaptive, save_contact_sheet, output_dir,
        )
        print(stats.summary())
        return records, stats

    def _compute_adaptive_thresholds(self, cropped_dir: Path) -> dict[str, float]:
        """Collect quality metrics from cropped frames to set adaptive cutoffs."""
        lap_vals: list[float] = []
        noise_vals: list[float] = []

        for _, frame in self.extractor.iter_saved_frames(cropped_dir):
            metrics = compute_quality_metrics(frame)
            lap_vals.append(metrics["laplacian_var"])
            noise_vals.append(metrics["noise"])

        if not lap_vals:
            return {
                "blur_cutoff": self.config.blur_threshold,
                "noise_cutoff": self.config.noise_threshold,
            }

        blur_cutoff = float(np.percentile(lap_vals, self.config.blur_percentile))
        noise_cutoff = float(np.percentile(noise_vals, self.config.noise_percentile))
        print(
            f"Adaptive thresholds — blur cutoff (p{self.config.blur_percentile:.0f}): "
            f"{blur_cutoff:.2f}, noise cutoff (p{self.config.noise_percentile:.0f}): {noise_cutoff:.2f}"
        )
        return {"blur_cutoff": blur_cutoff, "noise_cutoff": noise_cutoff}

    def _select_unique_frames(
        self,
        cropped_dir: Path,
        selected_dir: Path,
        adaptive: dict[str, float],
        save_contact_sheet: bool,
        output_dir: Path,
    ) -> tuple[list[FrameRecord], SelectionStats]:
        """Stage 2: filter cropped frames for quality and uniqueness."""
        stats = SelectionStats()
        records: list[FrameRecord] = []
        contact_entries: list[dict] = []

        last_selected_frame: np.ndarray | None = None
        last_selected_record: FrameRecord | None = None
        selection_counter = 0

        for frame_idx, frame in self.extractor.iter_saved_frames(cropped_dir):
            stats.total_frames += 1
            metrics = compute_quality_metrics(frame)
            ok, reason = passes_quality(
                metrics,
                adaptive,
                self.config.blur_threshold,
                self.config.noise_threshold,
                self.config.min_entropy,
            )

            if not ok:
                stats.quality_rejected += 1
                record = FrameRecord(
                    index=frame_idx,
                    status=self.config.STATUS_QUALITY,
                    quality_reason=reason,
                )
                records.append(record)
                contact_entries.append(self._contact_entry(frame_idx, record.status, frame))
                continue

            if last_selected_frame is None:
                selection_counter = self._accept_frame(
                    frame_idx, frame, selected_dir, selection_counter,
                    records, contact_entries,
                )
                stats.selected += 1
                last_selected_frame = frame.copy()
                last_selected_record = records[-1]
                continue

            redundant, sim_scores = is_redundant(
                frame,
                last_selected_frame,
                self.config.mad_threshold,
                self.config.ssim_threshold,
                self.config.histogram_threshold,
                self.config.similarity_scale,
                self.config.require_all_similarity,
            )

            if redundant:
                stats.redundant_rejected += 1
                record = FrameRecord(
                    index=frame_idx,
                    status=self.config.STATUS_REDUNDANT,
                    similarity_scores=sim_scores,
                )
                records.append(record)
                contact_entries.append(self._contact_entry(frame_idx, record.status, frame))
                continue

            overlap_scores: dict[str, float] = {}
            if self.config.use_overlap_deduplication:
                too_overlapped, overlap_scores = is_too_overlapped(
                    last_selected_frame,
                    frame,
                    self.config.max_overlap_matches,
                    self.config.max_overlap_match_ratio,
                    self.config.max_features,
                    self.config.orb_ratio_threshold,
                )
                if too_overlapped:
                    stats.overlap_rejected += 1
                    record = FrameRecord(
                        index=frame_idx,
                        status=self.config.STATUS_OVERLAP,
                        similarity_scores=sim_scores,
                        overlap_matches=int(overlap_scores.get("matches", 0)),
                    )
                    records.append(record)
                    contact_entries.append(self._contact_entry(frame_idx, record.status, frame))
                    continue
            else:
                overlap_scores = compute_overlap_metrics(
                    last_selected_frame,
                    frame,
                    self.config.max_features,
                    self.config.orb_ratio_threshold,
                )

            selection_counter = self._accept_frame(
                frame_idx, frame, selected_dir, selection_counter,
                records, contact_entries,
            )
            stats.selected += 1

            match_count = int(overlap_scores["matches"])
            records[-1].overlap_matches = match_count
            sufficient, _ = has_sufficient_overlap(
                last_selected_frame,
                frame,
                self.config.min_overlap_matches,
                self.config.max_features,
                self.config.orb_ratio_threshold,
            )
            if not sufficient:
                stats.overlap_warnings += 1
                print(
                    f"  [overlap warning] frames {last_selected_record.index} -> {frame_idx}: "
                    f"{match_count} matches (min {self.config.min_overlap_matches})"
                )

            last_selected_frame = frame.copy()
            last_selected_record = records[-1]

        if save_contact_sheet:
            sheet_path = output_dir / "contact_sheet.jpg"
            generate_contact_sheet(contact_entries, sheet_path, self.config)
            print(f"Contact sheet saved to: {sheet_path}")

        return records, stats

    def _accept_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        selected_dir: Path,
        counter: int,
        records: list[FrameRecord],
        contact_entries: list[dict],
    ) -> int:
        """Save a selected cropped frame and update bookkeeping."""
        counter += 1
        out_name = f"frame_{counter:04d}_src{frame_idx:05d}.png"
        save_frame(frame, selected_dir / out_name)

        record = FrameRecord(index=frame_idx, status=self.config.STATUS_SELECTED)
        records.append(record)
        contact_entries.append(self._contact_entry(frame_idx, record.status, frame))
        return counter

    def _contact_entry(self, frame_idx: int, status: str, frame: np.ndarray) -> dict:
        """Store a lightweight thumbnail for the contact sheet."""
        return {
            "index": frame_idx,
            "status": status,
            "frame": make_thumbnail(frame, self.config.contact_sheet_thumb_size),
        }
