#!/usr/bin/env python3
"""
Process an upper-trapezius ultrasound video end-to-end:
extract frames, select unique frames, and stitch a horizontal panorama.

Example:
    python process.py --input video468115.mp4 --output output/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from frame_selector.config import SelectorConfig
from frame_selector.pipeline import UltrasoundPipeline
from frame_selector.stitcher import StitchConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process an ultrasound video: extract cropped frames, select unique "
            "frames, and stitch a horizontal panorama of the upper trapezius."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input ultrasound video (e.g. .mp4).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for all results.",
    )

    quality = parser.add_argument_group("quality thresholds")
    quality.add_argument("--blur-threshold", type=float, default=30.0)
    quality.add_argument("--blur-percentile", type=float, default=5.0)
    quality.add_argument("--noise-threshold", type=float, default=35.0)
    quality.add_argument("--noise-percentile", type=float, default=95.0)
    quality.add_argument("--min-entropy", type=float, default=2.0)

    similarity = parser.add_argument_group("similarity / redundancy")
    similarity.add_argument("--similarity-mad", type=float, default=3.5)
    similarity.add_argument("--similarity-ssim", type=float, default=0.97)
    similarity.add_argument("--similarity-histogram", type=float, default=0.98)
    similarity.add_argument("--any-similarity", action="store_true")

    overlap = parser.add_argument_group("overlap deduplication")
    overlap.add_argument("--min-overlap-matches", type=int, default=25)
    overlap.add_argument("--max-overlap-matches", type=int, default=40)
    overlap.add_argument("--max-overlap-ratio", type=float, default=0.10)
    overlap.add_argument("--no-overlap-dedup", action="store_true")
    overlap.add_argument("--orb-ratio", type=float, default=0.75)

    crop = parser.add_argument_group("cropping")
    crop.add_argument("--no-crop", action="store_true")
    crop.add_argument("--crop-search-fraction", type=float, default=0.55)
    crop.add_argument("--no-crop-standardize", action="store_true")
    crop.add_argument(
        "--crop-standardize-method",
        choices=["median", "intersection"],
        default="median",
    )

    blank = parser.add_argument_group("blank frame filtering")
    blank.add_argument("--ignore-template", type=Path, default=Path("ignore_frame.png"))
    blank.add_argument("--no-ignore-template", action="store_true")
    blank.add_argument("--ignore-template-mad", type=float, default=12.0)
    blank.add_argument("--ignore-template-ssim", type=float, default=0.75)

    stitch = parser.add_argument_group("panorama stitching")
    stitch.add_argument(
        "--no-stitch",
        action="store_true",
        help="Stop after frame selection (do not build panorama).",
    )
    stitch.add_argument(
        "--min-advance",
        type=float,
        default=15.0,
        help="Min horizontal shift (px) to keep a frame during stitching.",
    )
    stitch.add_argument("--strip-fraction", type=float, default=0.30)
    stitch.add_argument("--blend-feather", type=int, default=40)
    stitch.add_argument("--use-opencv-stitcher", action="store_true")

    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> tuple[SelectorConfig, StitchConfig, bool]:
    selector = SelectorConfig(
        blur_threshold=args.blur_threshold,
        blur_percentile=args.blur_percentile,
        noise_threshold=args.noise_threshold,
        noise_percentile=args.noise_percentile,
        min_entropy=args.min_entropy,
        mad_threshold=args.similarity_mad,
        ssim_threshold=args.similarity_ssim,
        histogram_threshold=args.similarity_histogram,
        require_all_similarity=not args.any_similarity,
        min_overlap_matches=args.min_overlap_matches,
        max_overlap_matches=args.max_overlap_matches,
        max_overlap_match_ratio=args.max_overlap_ratio,
        use_overlap_deduplication=not args.no_overlap_dedup,
        orb_ratio_threshold=args.orb_ratio,
        crop_enabled=not args.no_crop,
        crop_search_fraction=args.crop_search_fraction,
        crop_standardize=not args.no_crop_standardize,
        crop_standardize_method=args.crop_standardize_method,
        ignore_template_enabled=not args.no_ignore_template,
        ignore_template_path=str(args.ignore_template),
        ignore_template_mad=args.ignore_template_mad,
        ignore_template_ssim=args.ignore_template_ssim,
    )
    stitch = StitchConfig(
        min_advance_pixels=args.min_advance,
        strip_fraction=args.strip_fraction,
        blend_feather=args.blend_feather,
        prefer_opencv_stitcher=args.use_opencv_stitcher,
    )
    return selector, stitch, not args.no_stitch


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input video not found: {args.input}")

    selector_cfg, stitch_cfg, run_stitch = config_from_args(args)
    pipeline = UltrasoundPipeline(selector_cfg, stitch_cfg, run_stitch=run_stitch)

    print(f"Input video:  {args.input}")
    print(f"Output dir:   {args.output}")
    _, panorama_path = pipeline.run(
        args.input,
        args.output,
        save_contact_sheet=not args.no_contact_sheet,
    )

    print("\n=== Output Summary ===")
    print(f"Cropped frames:   {args.output / 'cropped_frames'}")
    print(f"Selected frames:  {args.output / 'selected_frames'}")
    if not args.no_contact_sheet:
        print(f"Contact sheet:    {args.output / 'contact_sheet.jpg'}")
    if panorama_path is not None:
        print(f"Panorama:         {panorama_path}")


if __name__ == "__main__":
    main()
