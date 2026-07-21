import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass
from PIL import Image


def load_image_safe(path):
    image = cv2.imread(path)
    if image is not None:
        return image
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def draw_detections(image, results, model):
    """Draws bounding boxes + class labels + confidence on the image."""
    annotated = image.copy()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            color = (0, 220, 0)  # green box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated


def add_palette_strip(image, palette_hexes, strip_height=50):
    """Appends a horizontal strip of color swatches below the image."""
    h, w = image.shape[:2]
    strip = np.zeros((strip_height, w, 3), dtype=np.uint8)
    n = len(palette_hexes)
    seg_width = w // n

    for i, hex_color in enumerate(palette_hexes):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
        bgr = (b, g, r)
        x_start = i * seg_width
        x_end = w if i == n - 1 else (i + 1) * seg_width
        strip[:, x_start:x_end] = bgr

    return np.vstack([image, strip])


def add_score_banner(image, score, banner_height=40):
    """Adds a top banner showing the overall quality score."""
    h, w = image.shape[:2]
    banner = np.full((banner_height, w, 3), (30, 30, 30), dtype=np.uint8)
    text = f"Quality Score: {score}/100"
    cv2.putText(banner, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([banner, image])


def extract_color_palette(image, k=5, resize_dim=(150, 150)):
    from sklearn.cluster import KMeans
    small = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    hexes = []
    for idx in order:
        r, g, b = centers[idx]
        hexes.append("#{:02x}{:02x}{:02x}".format(
            int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))
        ))
    return hexes


def score_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness_score = min(sharpness / 1000 * 100, 100)
    brightness_score = 100 - abs(brightness - 127.5) / 127.5 * 100
    contrast_score = min(contrast / 80 * 100, 100)
    return round((sharpness_score * 0.5) + (brightness_score * 0.25) + (contrast_score * 0.25), 1)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".avif")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def annotate_video(model, video_path, output_folder, conf_threshold=0.4):
    """
    Processes every frame of a video, draws detection boxes on each,
    and writes the result to a new annotated .mp4 file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_name = os.path.splitext(os.path.basename(video_path))[0] + "_annotated.mp4"
    out_path = os.path.join(output_folder, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, verbose=False)
        annotated = draw_detections(frame, results, model)
        writer.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"    ...processed {frame_count} frames")

    cap.release()
    writer.release()
    print(f"    Saved annotated video: {out_path} ({frame_count} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize object detection + palette + quality score")
    parser.add_argument("--input", required=True, help="Path to an image/video file or folder")
    parser.add_argument("--output", default="annotated_output", help="Folder to save annotated output")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--k", type=int, default=5, help="Number of palette colors")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading YOLO model: {args.model} ...")
    model = YOLO(args.model)

    if os.path.isdir(args.input):
        all_files = [os.path.join(args.input, f) for f in os.listdir(args.input)]
    else:
        all_files = [args.input]

    image_paths = [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]
    video_paths = [f for f in all_files if f.lower().endswith(VIDEO_EXTENSIONS)]

    print(f"Found {len(image_paths)} image(s) and {len(video_paths)} video(s).")

    for i, path in enumerate(image_paths, 1):
        print(f"  [image {i}/{len(image_paths)}] {os.path.basename(path)}")
        image = load_image_safe(path)
        if image is None:
            print(f"    Skipped (could not read): {path}")
            continue

        results = model(image, conf=args.conf, verbose=False)
        annotated = draw_detections(image, results, model)

        palette = extract_color_palette(image, k=args.k)
        annotated = add_palette_strip(annotated, palette)

        score = score_quality(image)
        annotated = add_score_banner(annotated, score)

        out_name = os.path.splitext(os.path.basename(path))[0] + "_annotated.jpg"
        out_path = os.path.join(args.output, out_name)
        cv2.imwrite(out_path, annotated)

    for i, path in enumerate(video_paths, 1):
        print(f"  [video {i}/{len(video_paths)}] {os.path.basename(path)}")
        annotate_video(model, path, args.output, conf_threshold=args.conf)

    print(f"\nDone. Annotated output saved to: {args.output}")


if __name__ == "__main__":
    main()

