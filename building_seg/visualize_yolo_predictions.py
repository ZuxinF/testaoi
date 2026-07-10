from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay YOLO segmentation predictions with GT labels without text.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/yolo26_seg_tiles_512_debug")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-pred-mask-dir", default=None)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--gt-outline", default="0,220,80")
    parser.add_argument("--pred-fill", default="255,60,30")
    parser.add_argument("--pred-outline", default="255,0,0")
    return parser.parse_args()


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Color must be R,G,B: {value}")
    return tuple(max(0, min(255, x)) for x in parts)


def read_yolo_segments(label_path: Path, width: int, height: int) -> list[list[tuple[float, float]]]:
    segments = []
    if not label_path.exists():
        return segments
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = np.asarray([float(x) for x in parts[1:]], dtype="float32").reshape(-1, 2)
        points = [(float(x * width), float(y * height)) for x, y in coords]
        if len(points) >= 3:
            segments.append(points)
    return segments


def draw_gt(draw: ImageDraw.ImageDraw, segments, color: tuple[int, int, int]):
    for points in segments:
        draw.line(points + [points[0]], fill=(*color, 255), width=2)


def draw_predictions(overlay: Image.Image, result, fill: tuple[int, int, int], outline: tuple[int, int, int]):
    if result.masks is None:
        return
    masks = result.masks.data.detach().cpu().numpy()
    for mask in masks:
        alpha = (mask > 0.5).astype(np.uint8) * 75
        color = np.zeros((*alpha.shape, 4), dtype=np.uint8)
        color[..., 0] = fill[0]
        color[..., 1] = fill[1]
        color[..., 2] = fill[2]
        color[..., 3] = alpha
        overlay.alpha_composite(Image.fromarray(color, mode="RGBA"))

    # Draw mask contours after the fill so prediction boundaries remain visible.
    draw = ImageDraw.Draw(overlay)
    for mask in masks:
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            continue
        # A light-weight contour approximation from mask bounding pixels.
        # This avoids adding OpenCV as a hard dependency for drawing.
        binary = mask > 0.5
        edge = binary.copy()
        edge[1:, :] &= binary[:-1, :]
        edge[:-1, :] &= binary[1:, :]
        edge[:, 1:] &= binary[:, :-1]
        edge[:, :-1] &= binary[:, 1:]
        ey, ex = np.where(binary & ~edge)
        for x, y in zip(ex[::2], ey[::2]):
            draw.point((int(x), int(y)), fill=(*outline, 255))


def colorize_prediction_mask(result, size: tuple[int, int]) -> Image.Image:
    rgb = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    if result.masks is None:
        return Image.fromarray(rgb)

    masks = result.masks.data.detach().cpu().numpy()
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    else:
        class_ids = [0] * len(masks)

    palette = np.array(
        [
            [31, 119, 180],
            [255, 127, 14],
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [127, 127, 127],
            [188, 189, 34],
            [23, 190, 207],
            [90, 90, 90],
        ],
        dtype=np.uint8,
    )
    for mask, class_id in zip(masks, class_ids):
        color = palette[class_id % len(palette)]
        rgb[mask > 0.5] = color
    return Image.fromarray(rgb)


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split
    out_dir = Path(args.out) if args.out else dataset / "prediction_overlays" / args.split
    pred_mask_dir = Path(args.out_pred_mask_dir) if args.out_pred_mask_dir else dataset / "prediction_masks" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    gt_color = parse_color(args.gt_outline)
    pred_fill = parse_color(args.pred_fill)
    pred_outline = parse_color(args.pred_outline)

    image_paths = sorted(image_dir.glob("*.jpg"))[: args.limit]
    model = YOLO(args.model)
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=False,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw_predictions(overlay, result, pred_fill, pred_outline)
        draw = ImageDraw.Draw(overlay)
        gt_segments = read_yolo_segments(label_dir / f"{image_path.stem}.txt", width, height)
        draw_gt(draw, gt_segments, gt_color)
        out = Image.alpha_composite(image, overlay).convert("RGB")
        out.save(out_dir / image_path.name, quality=95)
        colorize_prediction_mask(result, image.size).save(pred_mask_dir / f"{image_path.stem}.png")

    print(f"Wrote {len(image_paths)} GT/prediction overlays to {out_dir}")
    print(f"Wrote {len(image_paths)} pure prediction masks to {pred_mask_dir}")
    print("GT: green outline; prediction: red translucent mask/outline; no text or confidence is drawn.")


if __name__ == "__main__":
    main()
