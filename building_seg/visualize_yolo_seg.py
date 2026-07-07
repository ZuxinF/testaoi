from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO segmentation labels.")
    parser.add_argument("--dataset", default="data/yolo26_seg_tiles_512_debug")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def load_names(dataset: Path) -> dict[int, str]:
    meta_path = dataset / "metadata" / "dataset.json"
    if not meta_path.exists():
        return {}
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return {idx: name for idx, name in enumerate(metadata.get("class_names", []))}


def draw_one(image_path: Path, label_path: Path, out_path: Path, names: dict[int, str]):
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(float(parts[0]))
            coords = np.asarray([float(x) for x in parts[1:]], dtype="float32").reshape(-1, 2)
            points = [(float(x * width), float(y * height)) for x, y in coords]
            color = COLORS[cls % len(COLORS)]
            draw.polygon(points, fill=(*color, 80), outline=(*color, 230))
            if points:
                draw.text(points[0], names.get(cls, str(cls)), fill=(*color, 255))
    out = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=95)


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    names = load_names(dataset)
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split
    out_dir = dataset / "label_previews" / args.split
    image_paths = sorted(image_dir.glob("*.jpg"))[: args.limit]
    for image_path in image_paths:
        draw_one(image_path, label_dir / f"{image_path.stem}.txt", out_dir / image_path.name, names)
    print(f"Wrote {len(image_paths)} previews to {out_dir}")


if __name__ == "__main__":
    main()
