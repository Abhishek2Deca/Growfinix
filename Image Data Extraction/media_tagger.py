import os
import cv2
import json
import argparse
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from ultralytics import YOLO

try:
    import pillow_avif  # noqa: F401  (registers AVIF support with Pillow)
except ImportError:
    pass
from PIL import Image


# ----------------------------
# 0. FORMAT-SAFE IMAGE LOADING
# ----------------------------
def load_image_safe(path):
    """
    Loads an image robustly even if its extension doesn't match its
    real format (e.g. an AVIF or WebP file mislabeled as .jpg).
    Tries OpenCV first (fast path); falls back to Pillow (handles
    AVIF/WebP/HEIC-like formats), converting to a standard array.
    Returns a BGR numpy array (OpenCV format) or None if unreadable.
    """
    image = cv2.imread(path)
    if image is not None:
        return image

    # Fallback: let Pillow figure out the real format regardless of extension
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            rgb_array = np.array(pil_img)
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            return bgr_array
    except Exception:
        return None

# ----------------------------
# 1. OBJECT DETECTION
# ----------------------------
def detect_objects(model, image_input, conf_threshold=0.4):
    """
    Runs YOLOv8 on the image and returns a list of detected objects
    with class name, confidence, and bounding box.
    image_input can be a file path (str) or a BGR numpy array.
    """
    results = model(image_input, conf=conf_threshold, verbose=False)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            detections.append({
                "label": cls_name,
                "confidence": round(conf, 3),
                "bbox": [round(v, 1) for v in xyxy]
            })

    return detections


# ----------------------------
# 2. COLOR PALETTE EXTRACTION
# ----------------------------
def extract_color_palette(image, k=5, resize_dim=(150, 150)):
    """
    Uses K-Means clustering to find the k dominant colors in the image.
    Returns a list of hex colors sorted by prevalence (most dominant first).
    """
    # Resize for speed, convert BGR -> RGB
    small = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)

    # Count how many pixels belong to each cluster -> dominance ranking
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)  # descending order of dominance

    palette = []
    total = counts.sum()
    for idx in order:
        r, g, b = centers[idx]
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(np.clip(r, 0, 255)),
            int(np.clip(g, 0, 255)),
            int(np.clip(b, 0, 255))
        )
        percentage = round(float(counts[idx]) / total * 100, 1)
        palette.append({"hex": hex_color, "percentage": percentage})

    return palette


# ----------------------------
# 3. LIGHTING / CONTRAST SCORING
# ----------------------------
def score_lighting_contrast(image):
    """
    Computes:
      - sharpness: variance of Laplacian (higher = sharper / more in-focus)
      - brightness: mean pixel intensity (0-255)
      - contrast: standard deviation of pixel intensity (higher = more contrast)
    Returns raw values plus a normalized 0-100 "quality score" combining them.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpness via Laplacian variance
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness (mean intensity)
    brightness = float(np.mean(gray))

    # Contrast (std deviation of intensity)
    contrast = float(np.std(gray))

    # Normalize each metric into a rough 0-100 scale for an overall score.
    # These thresholds are reasonable starting points — tune per your dataset.
    sharpness_score = min(sharpness / 1000 * 100, 100)       # 1000+ variance = very sharp
    brightness_score = 100 - abs(brightness - 127.5) / 127.5 * 100  # closer to mid-gray = better exposed
    contrast_score = min(contrast / 80 * 100, 100)            # 80+ std = high contrast

    overall_score = round(
        (sharpness_score * 0.5) + (brightness_score * 0.25) + (contrast_score * 0.25), 1
    )

    return {
        "sharpness_raw": round(sharpness, 2),
        "brightness_raw": round(brightness, 2),
        "contrast_raw": round(contrast, 2),
        "sharpness_score": round(sharpness_score, 1),
        "brightness_score": round(brightness_score, 1),
        "contrast_score": round(contrast_score, 1),
        "overall_quality_score": overall_score
    }


# ----------------------------
# 4. FULL PIPELINE PER IMAGE / FRAME
# ----------------------------
def process_image_array(model, image, label, source_path, palette_k=5, conf_threshold=0.4, extra_fields=None):
    """
    Runs the full metadata pipeline on an already-loaded BGR numpy array.
    Used for both static images and extracted video frames.
    """
    objects = detect_objects(model, image, conf_threshold)
    palette = extract_color_palette(image, k=palette_k)
    lighting = score_lighting_contrast(image)

    result = {
        "file": label,
        "path": source_path,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "objects_detected": objects,
        "object_count": len(objects),
        "unique_labels": sorted(list({o["label"] for o in objects})),
        "color_palette": palette,
        "lighting_contrast": lighting
    }
    if extra_fields:
        result.update(extra_fields)
    return result


def process_image(model, image_path, palette_k=5, conf_threshold=0.4):
    image = load_image_safe(image_path)
    if image is None:
        return {"file": image_path, "error": "Could not read image (unsupported or corrupted format)"}

    return process_image_array(
        model, image,
        label=os.path.basename(image_path),
        source_path=image_path,
        palette_k=palette_k,
        conf_threshold=conf_threshold
    )


def process_video(model, video_path, palette_k=5, conf_threshold=0.4, frame_interval_sec=1.0):
    """
    Extracts frames from a video at a fixed time interval and runs the
    full metadata pipeline (objects, palette, lighting) on each frame.
    Returns a list of per-frame metadata dicts.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [{"file": os.path.basename(video_path), "error": "Could not open video file"}]

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback if fps metadata is missing
    frame_step = max(int(round(fps * frame_interval_sec)), 1)

    results = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp_sec = round(frame_idx / fps, 2)
            label = f"{os.path.basename(video_path)}_frame{saved_idx:04d}"
            result = process_image_array(
                model, frame,
                label=label,
                source_path=video_path,
                palette_k=palette_k,
                conf_threshold=conf_threshold,
                extra_fields={"frame_index": frame_idx, "timestamp_sec": timestamp_sec}
            )
            results.append(result)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return results


# ----------------------------
# 5. BATCH PROCESSING + CLI
# ----------------------------
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".avif")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

def gather_media(input_path):
    """Returns (image_paths, video_paths) found at the given file or folder path."""
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
    videos = [f for f in files if f.lower().endswith(VIDEO_EXTENSIONS)]
    return images, videos


def main():
    parser = argparse.ArgumentParser(description="Automated Image/Video Metadata Extraction Pipeline")
    parser.add_argument("--input", required=True, help="Path to an image/video file or a folder containing them")
    parser.add_argument("--output", default="metadata_output.json", help="Path to save the JSON output")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model weights to use")
    parser.add_argument("--k", type=int, default=5, help="Number of dominant colors to extract")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for object detection")
    parser.add_argument("--frame-interval", type=float, default=1.0,
                         help="Seconds between extracted video frames (default: 1.0)")
    args = parser.parse_args()

    print(f"Loading YOLO model: {args.model} ...")
    model = YOLO(args.model)

    image_paths, video_paths = gather_media(args.input)
    if not image_paths and not video_paths:
        print("No valid images or videos found at the given path.")
        return

    print(f"Found {len(image_paths)} image(s) and {len(video_paths)} video(s). Processing...")
    all_results = []

    for i, path in enumerate(image_paths, 1):
        print(f"  [image {i}/{len(image_paths)}] {os.path.basename(path)}")
        result = process_image(model, path, palette_k=args.k, conf_threshold=args.conf)
        all_results.append(result)

    for i, path in enumerate(video_paths, 1):
        print(f"  [video {i}/{len(video_paths)}] {os.path.basename(path)} (extracting frames every {args.frame_interval}s)")
        video_results = process_video(
            model, path, palette_k=args.k, conf_threshold=args.conf,
            frame_interval_sec=args.frame_interval
        )
        print(f"    -> {len(video_results)} frame(s) processed")
        all_results.extend(video_results)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Metadata saved to: {args.output}")


if __name__ == "__main__":
    main()

