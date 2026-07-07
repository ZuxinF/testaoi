from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 57, 70],
        [42, 157, 143],
        [244, 162, 97],
        [69, 123, 157],
        [155, 93, 229],
        [255, 183, 3],
        [46, 196, 182],
        [131, 56, 236],
        [38, 70, 83],
        [231, 111, 81],
        [100, 100, 100],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create color previews for class-id segmentation masks.")
    parser.add_argument("--dataset", required=True, help="Prepared dataset directory")
    parser.add_argument("--out", default=None, help="Preview directory, default: <dataset>/mask_previews")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--overlay", action="store_true", help="Also save image+mask overlay previews")
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def colorize(mask: np.ndarray) -> np.ndarray:
    max_id = int(mask.max())
    if max_id >= len(PALETTE):
        extra = np.random.default_rng(20260707).integers(30, 230, size=(max_id + 1 - len(PALETTE), 3), dtype=np.uint8)
        palette = np.vstack([PALETTE, extra])
    else:
        palette = PALETTE
    return palette[mask]


def main():
    args = parse_args()
    root = Path(args.dataset)
    out_dir = Path(args.out) if args.out else root / "mask_previews"
    color_dir = out_dir / "color"
    overlay_dir = out_dir / "overlay"
    color_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = root / "metadata" / "dataset.json"
    class_names = []
    if metadata_path.exists():
        class_names = json.loads(metadata_path.read_text()).get("class_names", [])

    mask_paths = sorted((root / "masks").glob("*.png"))[: args.limit]
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        rgb = colorize(mask)
        Image.fromarray(rgb).save(color_dir / mask_path.name)

        if args.overlay:
            image_path = root / "images" / mask_path.name
            if image_path.exists():
                image = np.array(Image.open(image_path).convert("RGB"))
                mixed = (image * (1 - args.alpha) + rgb * args.alpha).astype(np.uint8)
                Image.fromarray(mixed).save(overlay_dir / mask_path.name)

    legend_path = out_dir / "legend.txt"
    legend_lines = []
    for idx, name in enumerate(class_names):
        if idx < len(PALETTE):
            color = tuple(int(v) for v in PALETTE[idx])
        else:
            color = ("auto", "auto", "auto")
        legend_lines.append(f"{idx}: {name} {color}")
    legend_path.write_text("\n".join(legend_lines) + ("\n" if legend_lines else ""))
    print(f"Wrote previews to {out_dir}")
    print(f"Wrote legend to {legend_path}")


if __name__ == "__main__":
    main()
