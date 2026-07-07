from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

from building_seg.tiles import parse_xyz_tile_path, tile_bounds_3857


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare segmentation patches directly from XYZ map tiles.")
    parser.add_argument("--tiles", default="data/tianditu/nansha_z18/tiles", help="Root containing z/x/y.jpg tiles")
    parser.add_argument("--labels", default="data/南沙区建筑物.gpkg", help="Building GPKG")
    parser.add_argument("--out", default="data/building_seg_real_tiles", help="Output dataset directory")
    parser.add_argument("--label-field", default="Function")
    parser.add_argument("--extensions", default=".jpg,.png,.jpeg")
    parser.add_argument("--patch-tiles", type=int, default=2, help="Number of XYZ tiles per side; 2 means 512x512 from 2x2 tiles")
    parser.add_argument("--max-positive", type=int, default=5000, help="Max patches with labeled buildings to export")
    parser.add_argument("--negative", type=int, default=0, help="Random patches without labels. Keep 0 if GT has many missing buildings.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def find_tiles(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def load_tile_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def build_tile_index(tile_paths: list[Path]) -> dict[tuple[int, int, int], Path]:
    tile_index = {}
    for tile_path in tile_paths:
        z, x, y = parse_xyz_tile_path(tile_path)
        tile_index[(z, x, y)] = tile_path
    return tile_index


def load_mosaic(anchor_path: Path, tile_index: dict[tuple[int, int, int], Path], patch_tiles: int):
    z, x0, y0 = parse_xyz_tile_path(anchor_path)
    first = Image.open(anchor_path).convert("RGB")
    tile_w, tile_h = first.size
    mosaic = Image.new("RGB", (tile_w * patch_tiles, tile_h * patch_tiles))

    for dy in range(patch_tiles):
        for dx in range(patch_tiles):
            path = tile_index.get((z, x0 + dx, y0 + dy))
            if path is None:
                return None, None, None, None
            tile = Image.open(path).convert("RGB")
            if tile.size != (tile_w, tile_h):
                return None, None, None, None
            mosaic.paste(tile, (dx * tile_w, dy * tile_h))

    left = tile_bounds_3857(x0, y0, z)[0]
    top = tile_bounds_3857(x0, y0, z)[3]
    right = tile_bounds_3857(x0 + patch_tiles - 1, y0 + patch_tiles - 1, z)[2]
    bottom = tile_bounds_3857(x0 + patch_tiles - 1, y0 + patch_tiles - 1, z)[1]
    bounds = (left, bottom, right, top)
    transform = from_bounds(*bounds, width=mosaic.size[0], height=mosaic.size[1])
    return mosaic, bounds, transform, (z, x0, y0)


def rasterize_patch(
    anchor_path: Path,
    tile_index: dict[tuple[int, int, int], Path],
    buildings: gpd.GeoDataFrame,
    class_to_id: dict[str, int],
    patch_tiles: int,
):
    mosaic, bounds, transform, xyz = load_mosaic(anchor_path, tile_index, patch_tiles)
    if mosaic is None:
        return None, None, 0, None

    width, height = mosaic.size
    patch_box = box(*bounds)

    subset = buildings[buildings.intersects(patch_box)]
    if subset.empty:
        return mosaic, np.zeros((height, width), dtype="uint8"), 0, xyz

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
    return mosaic, mask, len(subset), xyz


def save_sample(image: Image.Image, sample_id: str, out_dir: Path, mask: np.ndarray):
    image_out = out_dir / "images" / f"{sample_id}.png"
    mask_out = out_dir / "masks" / f"{sample_id}.png"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_out)
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
    tile_index = build_tile_index(tile_paths)

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
        image, mask, building_count, xyz = rasterize_patch(tile_path, tile_index, buildings, class_to_id, args.patch_tiles)
        if image is None:
            continue
        has_positive_pixels = bool(np.any(mask > 0))
        if has_positive_pixels and len(positive) < args.max_positive:
            positive.append((tile_path, image, mask, building_count, xyz))
        elif not has_positive_pixels and len(negative) < args.negative:
            negative.append((tile_path, image, mask, building_count, xyz))

        if len(positive) >= args.max_positive and len(negative) >= args.negative:
            break

        if scanned % 1000 == 0:
            print(f"Scanned {scanned}/{len(tile_paths)} tiles, positive={len(positive)}, negative={len(negative)}")

    selected = positive + negative
    if not selected:
        raise RuntimeError("No usable tiles selected. Check spatial overlap between tiles and GPKG.")

    sample_ids = []
    tile_records = []
    for idx, (tile_path, image, mask, building_count, xyz) in enumerate(selected):
        z, x, y = xyz
        sample_id = f"z{z}_x{x}_y{y}_s{image.size[0]}"
        save_sample(image, sample_id, out_dir, mask)
        sample_ids.append(sample_id)
        tile_records.append(
            {
                "sample_id": sample_id,
                "anchor_tile_path": str(tile_path),
                "z": z,
                "x": x,
                "y": y,
                "patch_tiles": args.patch_tiles,
                "pixel_size": list(image.size),
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
        "patch_tiles": args.patch_tiles,
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
