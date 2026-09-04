#!/usr/bin/env python3
"""
Stitch already-selected frames into a panorama (stitch-only mode).

For the full video-to-panorama pipeline, use: python process.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from frame_selector.stitcher import StitchConfig, UltrasoundPanoramaStitcher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch pre-selected frames only. For full pipeline use process.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--min-advance", type=float, default=15.0)
    parser.add_argument("--strip-fraction", type=float, default=0.30)
    parser.add_argument("--blend-feather", type=int, default=40)
    parser.add_argument("--use-opencv-stitcher", action="store_true")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")

    config = StitchConfig(
        min_advance_pixels=args.min_advance,
        strip_fraction=args.strip_fraction,
        blend_feather=args.blend_feather,
        prefer_opencv_stitcher=args.use_opencv_stitcher,
    )
    result = UltrasoundPanoramaStitcher(config).stitch_directory(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result.panorama)

    print(f"Input frames:     {result.num_input_frames}")
    print(f"Frames stitched:  {result.num_used_frames}")
    print(f"Panorama size:    {result.panorama.shape[1]}x{result.panorama.shape[0]}")
    print(f"Saved to:         {args.output}")


if __name__ == "__main__":
    main()
