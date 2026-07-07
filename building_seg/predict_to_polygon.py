from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape

from building_seg.models import load_checkpoint, predict_large_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict building function mask and polygonize it.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-mask", required=True, help="Output single-band class-id GeoTIFF")
    parser.add_argument("--out-gpkg", required=True, help="Output polygon GeoPackage")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window", default=None, help="Optional debug window: col,row,width,height")
    parser.add_argument("--min-area-pixels", type=int, default=12)
    return parser.parse_args()


def read_image_window(src, window: Window | None):
    if window is None:
        data = src.read([1, 2, 3])
        transform = src.transform
        profile = src.profile.copy()
    else:
        data = src.read([1, 2, 3], window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(width=int(window.width), height=int(window.height), transform=transform)
    image = torch.from_numpy(np.transpose(data, (1, 2, 0)).astype("float32") / 255.0).permute(2, 0, 1)
    return image, transform, profile


def write_mask(path: Path, mask: np.ndarray, profile: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    for key in ["photometric", "interleave", "jpeg_quality"]:
        profile.pop(key, None)
    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def polygonize_mask(mask: np.ndarray, transform, crs, class_names: list[str], min_area_pixels: int) -> gpd.GeoDataFrame:
    records = []
    pixel_area = abs(transform.a * transform.e)
    for geom, value in shapes(mask, mask=mask > 0, transform=transform):
        class_id = int(value)
        if class_id <= 0 or class_id >= len(class_names):
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        if poly.area <= 0:
            continue
        pixel_count = poly.area / pixel_area if pixel_area else 0
        if pixel_count < min_area_pixels:
            continue
        records.append({"class_id": class_id, "Function": class_names[class_id], "geometry": poly})
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def main():
    args = parse_args()
    model, class_names, ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    model = model.to(args.device)

    window = None
    if args.window:
        col, row, width, height = [int(v) for v in args.window.split(",")]
        window = Window(col, row, width, height)

    with rasterio.open(args.image) as src:
        if window is None and src.width * src.height > 4096 * 4096:
            raise RuntimeError(
                "Full-image inference is intentionally disabled for large rasters in this debug pipeline. "
                "Pass --window col,row,width,height, or extend predict_to_polygon.py with streaming tiled output."
            )
        image, transform, profile = read_image_window(src, window)
        mask = predict_large_image(model, image, tile_size=args.tile_size, overlap=args.overlap, device=args.device)
        write_mask(Path(args.out_mask), mask, profile)
        gdf = polygonize_mask(mask, transform, src.crs, class_names, args.min_area_pixels)
        Path(args.out_gpkg).parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(args.out_gpkg, driver="GPKG", index=False)

    meta_path = Path(args.out_gpkg).with_suffix(".classes.json")
    meta_path.write_text(json.dumps({"class_names": class_names, "checkpoint": str(args.checkpoint)}, ensure_ascii=False, indent=2))
    print(f"Wrote mask: {args.out_mask}")
    print(f"Wrote polygons: {args.out_gpkg} ({len(gdf)} features)")


if __name__ == "__main__":
    main()
