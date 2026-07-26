# 🖼️ Image & Video Data Extraction

A computer vision pipeline that runs **YOLOv8 object detection**, extracts
the **dominant color palette**, and scores **lighting/sharpness/contrast
quality** for both images and videos — producing structured JSON metadata
and visually annotated output.

## Architecture

```
images/ or a single file        Input images (.jpg/.png/.webp/.avif/...) or
        │                       videos (.mp4/.mov/.avi/.mkv/.webm)
        ▼
  media_tagger.py               For each image (or each extracted video
        │                       frame, at a configurable interval):
        │                         1. Loads it robustly (OpenCV, falling
        │                            back to Pillow for formats like
        │                            AVIF/WebP that OpenCV can't read)
        │                         2. Detects objects with YOLOv8
        │                         3. Extracts the top-k dominant colors
        │                            via K-Means clustering on pixels
        │                         4. Scores sharpness, brightness, and
        │                            contrast → an overall quality score
        ▼
metadata_output.json            One JSON record per image/frame: detected
        │                       objects (label, confidence, bbox), color
        │                       palette (hex + %), and quality scores
        ▼
  visualize_results.py          Re-runs detection to draw:
                                   - bounding boxes + labels on images
                                   - a color palette strip below the image
                                   - a quality-score banner above the image
                                 For videos: draws boxes on every frame and
                                 writes out a new annotated .mp4
        ▼
  annotated_output/             Annotated images (.jpg) and videos (.mp4)
```

**Why this stack:**
- **YOLOv8 (Ultralytics)** — fast, accurate, pre-trained object detector;
  no training required to get started (`bus.jpg` is Ultralytics' own
  standard test image).
- **OpenCV + Pillow fallback** — OpenCV is fast for common formats
  (jpg/png), but can silently fail on AVIF/WebP; falling back to Pillow
  (with `pillow-avif` for AVIF support) makes loading robust to real-world
  mixed-format datasets.
- **K-Means for color palette** — a fast, simple way to summarize an
  image's dominant colors without needing a dedicated model.
- **Laplacian variance + intensity stats for quality scoring** — classic,
  lightweight computer vision metrics for sharpness (blur detection),
  brightness, and contrast that don't require a trained model.
- **Separate tagging vs. visualization scripts** — `media_tagger.py`
  produces the structured metadata once; `visualize_results.py` re-renders
  visuals from the same underlying detection logic, so you can regenerate
  visuals without necessarily re-running the full metadata pipeline.

## Setup

1. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python pillow pillow-avif-plugin numpy
   ```
   *(`pillow-avif-plugin` is optional — only needed if you're working with
   AVIF images; the script degrades gracefully without it.)*

2. **Run the metadata extraction pipeline**:
   ```bash
   python media_tagger.py --input images/ --output metadata_output.json
   ```
   Useful options:
   - `--model yolov8s.pt` — YOLOv8 weights to use (auto-downloads on first
     run if not already present)
   - `--conf 0.4` — confidence threshold for detections
   - `--k 5` — number of dominant colors to extract
   - `--frame-interval 1.0` — seconds between extracted video frames (for
     video inputs)
   - `--input` also accepts a single file path, not just a folder

3. **Generate annotated visuals**:
   ```bash
   python visualize_results.py --input images/ --output annotated_output
   ```
   Same `--model`, `--conf`, `--k` options apply. Images get bounding
   boxes, a color palette strip, and a quality-score banner; videos get a
   fully annotated `_annotated.mp4` copy.

## Using your own data

Drop your own images and/or videos into a folder and point `--input` at
it. Both scripts auto-detect file type by extension
(`IMAGE_EXTENSIONS` / `VIDEO_EXTENSIONS` in each file) and route
accordingly — no code changes needed for standard formats.

## Notes for extending this project

- **Large media files**: annotated videos can easily exceed GitHub's
  100MB file size limit (this happened with `sample5_annotated.mp4` in
  this repo). Use Git LFS if you need to version large sample videos, or
  keep bulky raw/annotated media out of source control.
- **Duplicate detection logic**: `media_tagger.py` and
  `visualize_results.py` both reimplement `extract_color_palette` and
  similar logic independently. Extracting shared functions (color
  palette, quality scoring, safe image loading) into a common module
  would reduce duplication and keep the two scripts in sync.
- **Confidence/threshold tuning**: the default `conf=0.4` is a reasonable
  starting point, but tightening or loosening it per use case (e.g. higher
  precision vs. higher recall) is worth experimenting with per dataset.
- **Performance**: for large batches, consider running YOLO inference on
  GPU (`ultralytics` auto-detects CUDA if available) and/or batching
  frames together rather than one at a time.
