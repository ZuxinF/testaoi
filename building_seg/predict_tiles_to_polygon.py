from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape

from building_seg.models import load_checkpoint
from building_seg.prepare_seg_dataset_from_tiles import build_tile_index, find_tiles, load_mosaic


def parse_args():
    parser = argparse.ArgumentParser(description="Predict building function polygons directly from XYZ tiles.")
    parser.add_argument("--tiles", default="data/tianditu/nansha_z18/tiles")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-gpkg", required=True)
    parser.add_argument("--out-mask-dir", default=None, help="Optional directory for predicted class-id PNG masks")
    parser.add_argument("--extensions", default=".jpg,.png,.jpeg")
    parser.add_argument("--patch-tiles", type=int, default=2, help="2 means 2x2 tiles -> 512x512 patch")
    parser.add_argument("--stride-tiles", type=int, default=2, help="Anchor stride in tile units; 2 avoids overlap for patch-tiles=2")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of patches to predict")
    parser.add_argument("--min-area-pixels", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def polygonize_mask(mask: np.ndarray, transform, class_names: list[str], patch_id: str, min_area_pixels: int):
    pixel_area = abs(transform.a * transform.e)
    records = []
    for geom, value in shapes(mask, mask=mask > 0, transform=transform):
        class_id = int(value)
        if class_id <= 0 or class_id >= len(class_names):
            continue
        poly = shape(geom)
        if poly.is_empty or poly.area <= 0:
            continue
        pixel_count = poly.area / pixel_area if pixel_area else 0
        if pixel_count < min_area_pixels:
            continue
        records.append(
            {
                "patch_id": patch_id,
                "class_id": class_id,
                "Function": class_names[class_id],
                "geometry": poly,
            }
        )
    return records


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main():
    args = parse_args()
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.lower().split(",")}
    tile_paths = find_tiles(Path(args.tiles), extensions)
    tile_index = build_tile_index(tile_paths)
    if not tile_paths:
        raise RuntimeError(f"No tile images found under {args.tiles}")

    model, class_names, ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    model = model.to(args.device).eval()

    xyz_list = sorted(tile_index)
    min_x = min(x for _, x, _ in xyz_list)
    min_y = min(y for _, _, y in xyz_list)

    if args.out_mask_dir:
        Path(args.out_mask_dir).mkdir(parents=True, exist_ok=True)

    all_records = []
    processed = 0
    for z, x, y in xyz_list:
        if (x - min_x) % args.stride_tiles or (y - min_y) % args.stride_tiles:
            continue
        anchor_path = tile_index[(z, x, y)]
        image, bounds, transform, xyz = load_mosaic(anchor_path, tile_index, args.patch_tiles)
        if image is None:
            continue

        patch_id = f"z{z}_x{x}_y{y}_s{image.size[0]}"
        with torch.no_grad():
            logits = model(image_to_tensor(image).to(args.device)).cpu().squeeze(0)
            mask = torch.argmax(logits, dim=0).numpy().astype("uint8")

        if args.out_mask_dir:
            Image.fromarray(mask).save(Path(args.out_mask_dir) / f"{patch_id}.png")

        all_records.extend(polygonize_mask(mask, transform, class_names, patch_id, args.min_area_pixels))
        processed += 1
        if processed % 100 == 0:
            print(f"Predicted {processed} patches, polygons={len(all_records)}")
        if args.limit is not None and processed >= args.limit:
            break

    gdf = gpd.GeoDataFrame(all_records, geometry="geometry", crs="EPSG:3857")
    out_path = Path(args.out_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", index=False)
    out_path.with_suffix(".classes.json").write_text(
        json.dumps({"class_names": class_names, "checkpoint": str(args.checkpoint)}, ensure_ascii=False, indent=2)
    )
    print(f"Predicted patches: {processed}")
    print(f"Wrote polygons: {out_path} ({len(gdf)} features)")


if __name__ == "__main__":
    main()
