from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import box

from building_seg.tiles import parse_xyz_tile_path, tile_bounds_3857, tile_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare segmentation patches directly from XYZ map tiles.")
    parser.add_argument("--tiles", default="data/tianditu/nansha_z18/tiles", help="Root containing z/x/y.jpg tiles")
    parser.add_argument("--labels", default="data/南沙区建筑物.gpkg", help="Building GPKG")
    parser.add_argument("--out", default="data/building_seg_real_tiles", help="Output dataset directory")
    parser.add_argument("--label-field", default="Function")
    parser.add_argument("--extensions", default=".jpg,.png,.jpeg")
    parser.add_argument("--max-positive", type=int, default=5000, help="Max tiles with buildings to export")
    parser.add_argument("--negative", type=int, default=1000, help="Random tiles without buildings to export")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--copy-images", action="store_true", help="Copy original tile bytes instead of re-saving as PNG")
    return parser.parse_args()


def find_tiles(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def load_tile_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def rasterize_tile(tile_path: Path, buildings: gpd.GeoDataFrame, class_to_id: dict[str, int]) -> tuple[np.ndarray, int]:
    image = load_tile_image(tile_path)
    height, width = image.shape[:2]
    z, x, y = parse_xyz_tile_path(tile_path)
    bounds = tile_bounds_3857(x, y, z)
    transform = tile_transform(x, y, z, width, height)
    tile_box = box(*bounds)

    subset = buildings[buildings.intersects(tile_box)]
    if subset.empty:
        return np.zeros((height, width), dtype="uint8"), 0

    shapes = [
        (geom, class_to_id[label])
        for geom, label in zip(subset.geometry, subset["__label__"])
        if geom is not None and not geom.is_empty
    ]
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return mask, len(subset)


def save_sample(tile_path: Path, sample_id: str, out_dir: Path, mask: np.ndarray, copy_images: bool):
    image_out = out_dir / "images" / f"{sample_id}.png"
    mask_out = out_dir / "masks" / f"{sample_id}.png"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)

    if copy_images and tile_path.suffix.lower() == ".png":
        shutil.copy2(tile_path, image_out)
    else:
        Image.open(tile_path).convert("RGB").save(image_out)
    Image.fromarray(mask).save(mask_out)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    for name in ["images", "masks", "splits", "metadata"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.lower().split(",")}
    tile_paths = find_tiles(Path(args.tiles), extensions)
    if not tile_paths:
        raise RuntimeError(f"No tile images found under {args.tiles}")

    buildings = gpd.read_file(args.labels)
    if args.label_field not in buildings.columns:
        raise ValueError(f"Label field '{args.label_field}' not found. Columns: {list(buildings.columns)}")
    if buildings.crs is None:
        raise ValueError("Input GPKG has no CRS. Please define the CRS before preparing masks.")

    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings = buildings.to_crs("EPSG:3857")
    buildings["__label__"] = buildings[args.label_field].fillna("其他").astype(str)

    labels = sorted(buildings["__label__"].unique())
    class_names = ["background"] + labels
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    positive = []
    negative = []
    random.shuffle(tile_paths)

    scanned = 0
    for tile_path in tile_paths:
        scanned += 1
        mask, building_count = rasterize_tile(tile_path, buildings, class_to_id)
        has_positive_pixels = bool(np.any(mask > 0))
        if has_positive_pixels and len(positive) < args.max_positive:
            positive.append((tile_path, mask, building_count))
        elif not has_positive_pixels and len(negative) < args.negative:
            negative.append((tile_path, mask, building_count))

        if len(positive) >= args.max_positive and len(negative) >= args.negative:
            break

        if scanned % 1000 == 0:
            print(f"Scanned {scanned}/{len(tile_paths)} tiles, positive={len(positive)}, negative={len(negative)}")

    selected = positive + negative
    if not selected:
        raise RuntimeError("No usable tiles selected. Check spatial overlap between tiles and GPKG.")

    sample_ids = []
    tile_records = []
    for idx, (tile_path, mask, building_count) in enumerate(selected):
        z, x, y = parse_xyz_tile_path(tile_path)
        sample_id = f"z{z}_x{x}_y{y}"
        save_sample(tile_path, sample_id, out_dir, mask, copy_images=args.copy_images)
        sample_ids.append(sample_id)
        tile_records.append(
            {
                "sample_id": sample_id,
                "tile_path": str(tile_path),
                "z": z,
                "x": x,
                "y": y,
                "building_count": building_count,
                "positive_pixels": int(np.count_nonzero(mask)),
            }
        )

    random.shuffle(sample_ids)
    n_val = max(1, int(len(sample_ids) * args.val_ratio)) if len(sample_ids) > 1 else 0
    val_ids = sample_ids[:n_val]
    train_ids = sample_ids[n_val:]
    (out_dir / "splits" / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""))
    (out_dir / "splits" / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""))

    metadata = {
        "tiles": str(Path(args.tiles).resolve()),
        "labels": str(Path(args.labels).resolve()),
        "label_field": args.label_field,
        "crs": "EPSG:3857",
        "class_names": class_names,
        "class_to_id": class_to_id,
        "total_tiles_found": len(tile_paths),
        "tiles_scanned": scanned,
        "positive_samples": len(positive),
        "negative_samples": len(negative),
        "total_samples": len(sample_ids),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
    }
    (out_dir / "metadata" / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (out_dir / "metadata" / "tiles.json").write_text(json.dumps(tile_records, ensure_ascii=False, indent=2))

    print(f"Prepared {len(sample_ids)} tile samples at {out_dir}")
    print(f"Positive={len(positive)}, negative={len(negative)}, scanned={scanned}/{len(tile_paths)}")
    print(f"Classes: {class_to_id}")


if __name__ == "__main__":
    main()
