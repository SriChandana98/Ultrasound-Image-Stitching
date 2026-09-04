# Ultrasound Image Stitching

Build a horizontal panorama of the upper trapezius muscle from an ultrasound sweep video. One command extracts frames, removes machine UI and blank frames, selects unique high-quality frames, and stitches them into a single landscape image.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python process.py --input video468115.mp4 --output output/
```

That runs the full pipeline and writes everything under `output/`.

---

## Input

| Item | Description |
|------|-------------|
| **Video file** | Ultrasound recording of an upper trapezius sweep (e.g. `.mp4`, `.avi`). Any resolution and frame rate. |
| **`ignore_frame.png`** (optional) | Reference image of a blank/no-signal frame. Frames matching this template are dropped during extraction. Defaults to `ignore_frame.png` in the project root. |

Example input video (`video468115.mp4`):

<video src="video468115.mp4" controls width="640">
  Your browser does not support embedded video. Open <a href="video468115.mp4">video468115.mp4</a> directly.
</video>

---

## Output

All results are written to the directory passed with `--output`:

| Path | Description |
|------|-------------|
| `output/cropped_frames/` | Every non-blank frame, cropped to the scan region only (standardized size, UI removed) |
| `output/selected_frames/` | Final unique frames chosen for stitching (`frame_0001_src00000.png`, …) |
| `output/contact_sheet.jpg` | Grid showing which frames were selected vs rejected during selection |
| `output/panorama.png` | **Final horizontal landscape panorama** stitched from selected frames |

Example output panorama (`output/panorama.png`):

![Stitched upper trapezius panorama](output/panorama.png)

Use `--no-stitch` to stop after frame selection (no `panorama.png`).  
Use `--no-contact-sheet` to skip the contact sheet.

---

## Pipeline overview

The pipeline has three stages, each designed for ultrasound-specific behaviour:

```
Video
  │
  ▼  Stage 1 — Extract & crop
cropped_frames/
  │
  ▼  Stage 2 — Select unique frames
selected_frames/
  │
  ▼  Stage 3 — Stitch panorama
panorama.png
```

---

## Algorithms and rationale

### Stage 1: Extract, crop, and skip blank frames

**Goal:** Convert the video into clean, same-size scan images with no machine UI.

1. **Standardized crop**  
   Detect the B-mode scan panel on each frame, then compute one fixed crop box (median of all detections) applied to every frame. This removes top metadata, the right parameter sidebar, and bottom toolbar/text, and ensures all images share identical dimensions for downstream metrics and stitching.

   *Why:* Ultrasound recorders use a fixed on-screen layout; the scan region is always in the upper portion of the frame. Percentile-based row/column occupancy separates tissue pixels from black UI bars without needing manual coordinates.

2. **Blank-frame template matching**  
   After cropping, each frame is compared to `ignore_frame.png` using mean absolute difference (MAD) and structural similarity (SSIM). Matching frames are **not saved**.

   *Why:* Probe-off-tissue frames are nearly black with faint top-edge artifacts. Pixel + structural matching is more reliable than brightness alone (blank frames can have non-zero entropy due to noise bands).

---

### Stage 2: Select unique, stitch-ready frames

**Goal:** Keep frames that are sharp, non-redundant, and spaced enough for panorama assembly.

1. **Adaptive quality filtering**  
   - **Laplacian variance** — rejects blurry frames (low edge energy).  
   - **Noise estimate** — high-frequency residual std; rejects excessive speckle/motion noise.  
   - **Shannon entropy** — only rejects completely flat/blank frames (low threshold so normal low-contrast muscle tissue is kept).

   Adaptive **percentile cutoffs** (5th for blur, 95th for noise) are computed across the video so thresholds adapt to each recording’s contrast and speckle level.

   *Why:* Ultrasound has inherently soft contrast; absolute thresholds alone would reject usable frames. Percentiles adapt per video while absolute floors prevent extreme outliers.

2. **Pixel-similarity redundancy removal**  
   Compare each candidate to the **last selected frame** using MAD, SSIM, and histogram correlation. A frame is redundant only when **all three** agree (conservative default).

   *Why:* Consecutive sweep frames are visually similar; requiring all metrics reduces false rejections while still dropping near-duplicates from slow probe motion.

3. **Feature-overlap deduplication**  
   ORB/SIFT feature matches with Lowe’s ratio test. Reject if matches ≥ `max_overlap_matches` or match ratio ≥ `max_overlap_ratio` vs. the last kept frame.

   *Why:* Pixel metrics can miss structural redundancy; feature overlap catches frames that haven’t moved enough in tissue space—important before stitching.

---

### Stage 3: Horizontal panorama stitching

**Goal:** Combine selected frames into one wide landscape image of the full muscle sweep.

1. **Similar-frame pruning**  
   Before stitching, keep only frames separated by at least `--min-advance` pixels of horizontal displacement (always keeps first and last). Of ~61 selected frames, typically ~10–15 remain.

   *Why:* Many selected frames shift only 0–4 px horizontally; stitching all of them would stack duplicate content. Pruning keeps frames that add new spatial coverage.

2. **Pairwise alignment** (best shift between consecutive kept frames)  
   - **Optical flow** (Farneback) — robust on speckle texture; median horizontal displacement.  
   - **Template matching** — match the right-edge strip of frame *n* inside frame *n+1* to estimate overlap.  
   - **ORB/SIFT + RANSAC affine** — fallback when flow/template confidence is low.

   *Why:* Ultrasound speckle weakens sparse feature matchers alone; dense flow and template matching on overlapping strips work better for small translational shifts.

3. **Feathered blending**  
   Frames are placed on a wide canvas at cumulative shift positions. Overlap regions use linear feather weights to avoid hard seams.

   *Why:* Simple averaging without feathering leaves visible boundaries; ultrasound overlap zones benefit from smooth blending.

---

## Common options

```bash
# Full pipeline (default)
python process.py --input video.mp4 --output output/

# Frame selection only (no panorama)
python process.py --input video.mp4 --output output/ --no-stitch

# Tune stitching — keep more frames in panorama
python process.py --input video.mp4 --output output/ --min-advance 8

# Tune selection — fewer redundant frames
python process.py --input video.mp4 --output output/ --max-overlap-matches 30
```

Run `python process.py --help` for all options.

---

## Project layout

```
process.py              # Main entry point (full pipeline)
select_frames.py        # Wrapper → process.py
stitch_frames.py        # Stitch-only from existing selected_frames/
ignore_frame.png        # Blank-frame template (optional, user-provided)

frame_selector/
  pipeline.py           # Orchestrates all three stages
  extractor.py          # Stage 1: video → cropped frames
  crop.py               # Scan ROI detection
  blank_detector.py     # Blank template matching
  selector.py           # Stage 2: frame selection
  metrics.py            # Quality & similarity metrics
  overlap.py            # Feature overlap checks
  stitcher.py           # Stage 3: panorama stitching
  visualization.py      # Contact sheet
  config.py             # Threshold defaults
```
