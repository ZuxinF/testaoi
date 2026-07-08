from __future__ import annotations

import argparse
import json
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
from PIL import Image
from shapely.geometry import box
from shapely.ops import unary_union

from building_seg.prepare_seg_dataset_from_tiles import build_tile_index, find_tiles, load_mosaic


_WORKER_TILE_INDEX = None
_WORKER_BUILDINGS = None
_WORKER_CLASS_TO_ID = None
_WORKER_PATCH_TILES = None
_WORKER_MIN_AREA_PIXELS = None
_WORKER_SIMPLIFY_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare YOLO segmentation dataset from XYZ tiles and building GPKG.")
    parser.add_argument("--tiles", default="data/tianditu/nansha_z18/tiles")
    parser.add_argument("--labels", default="data/南沙区建筑物.gpkg")
    parser.add_argument("--out", default="data/yolo26_seg_tiles_512_debug")
    parser.add_argument("--label-field", default="Function")
    parser.add_argument("--extensions", default=".jpg,.png,.jpeg")
    parser.add_argument("--patch-tiles", type=int, default=2, help="2 means 2x2 XYZ tiles -> 512x512 image")
    parser.add_argument("--max-positive", type=int, default=200, help="Max positive patches to export. Use 0 for all positive patches.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-area-pixels", type=float, default=16.0)
    parser.add_argument("--simplify-pixels", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1, help="Process workers for tile mosaic/geometry export. Use 1 to disable parallel export.")
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def safe_class_names(labels: list[str]) -> list[str]:
    return [str(label).replace("\n", " ").strip() or "其他" for label in labels]


def pixel_area(transform) -> float:
    return abs(transform.a * transform.e)


def polygon_to_yolo_segments(geom, transform, width: int, height: int, simplify_pixels: float) -> list[list[float]]:
    if geom.is_empty:
        return []
    if simplify_pixels > 0:
        tolerance = simplify_pixels * abs(transform.a)
        geom = geom.simplify(tolerance, preserve_topology=True)
    geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    segments = []
    for poly in geoms:
        if poly.is_empty or poly.area <= 0:
            continue
        coords = list(poly.exterior.coords)
        if len(coords) < 4:
            continue
        points = []
        for x_geo, y_geo in coords[:-1]:
            col, row = ~transform * (x_geo, y_geo)
            x_norm = min(max(col / width, 0.0), 1.0)
            y_norm = min(max(row / height, 0.0), 1.0)
            points.extend([x_norm, y_norm])
        if len(points) >= 6:
            segments.append(points)
    return segments


def write_sample(
    image: Image.Image,
    yolo_lines: list[str],
    sample_id: str,
    split: str,
    out_dir: Path,
):
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image.save(image_dir / f"{sample_id}.jpg", quality=95)
    (label_dir / f"{sample_id}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))


def build_sample_from_tile(
    tile_path: Path,
    tile_index,
    buildings: gpd.GeoDataFrame,
    class_to_id: dict[str, int],
    patch_tiles: int,
    min_area_pixels: float,
    simplify_pixels: float,
):
    image, bounds, transform, xyz = load_mosaic(tile_path, tile_index, patch_tiles)
    if image is None:
        return None
    width, height = image.size
    patch_box = box(*bounds)
    subset = buildings[buildings.intersects(patch_box)]
    if subset.empty:
        return None

    lines = []
    patch_area = pixel_area(transform)
    for label, geom in zip(subset["__label__"], subset.geometry):
        if geom is None or geom.is_empty:
            continue
        clipped = geom.intersection(patch_box)
        if clipped.is_empty:
            continue
        if clipped.area / patch_area < min_area_pixels:
            continue
        # Dissolve geometry collections from invalid or multipart intersections.
        if clipped.geom_type == "GeometryCollection":
            polys = [g for g in clipped.geoms if g.geom_type in {"Polygon", "MultiPolygon"} and not g.is_empty]
            if not polys:
                continue
            clipped = unary_union(polys)
        segments = polygon_to_yolo_segments(clipped, transform, width, height, simplify_pixels)
        for segment in segments:
            class_id = class_to_id[str(label).strip() or "其他"]
            values = " ".join(f"{v:.6f}" for v in segment)
            lines.append(f"{class_id} {values}")

    if not lines:
        return None

    z, x, y = xyz
    sample_id = f"z{z}_x{x}_y{y}_s{width}"
    return sample_id, image, lines, len(subset)


def init_worker(tile_index, buildings, class_to_id, patch_tiles, min_area_pixels, simplify_pixels):
    global _WORKER_TILE_INDEX
    global _WORKER_BUILDINGS
    global _WORKER_CLASS_TO_ID
    global _WORKER_PATCH_TILES
    global _WORKER_MIN_AREA_PIXELS
    global _WORKER_SIMPLIFY_PIXELS
    _WORKER_TILE_INDEX = tile_index
    _WORKER_BUILDINGS = buildings
    _WORKER_CLASS_TO_ID = class_to_id
    _WORKER_PATCH_TILES = patch_tiles
    _WORKER_MIN_AREA_PIXELS = min_area_pixels
    _WORKER_SIMPLIFY_PIXELS = simplify_pixels


def build_sample_from_tile_worker(tile_path: Path):
    return build_sample_from_tile(
        tile_path,
        _WORKER_TILE_INDEX,
        _WORKER_BUILDINGS,
        _WORKER_CLASS_TO_ID,
        _WORKER_PATCH_TILES,
        _WORKER_MIN_AREA_PIXELS,
        _WORKER_SIMPLIFY_PIXELS,
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)

    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.lower().split(",")}
    tile_paths = find_tiles(Path(args.tiles), extensions)
    if not tile_paths:
        raise RuntimeError(f"No tile images found under {args.tiles}")
    tile_index = build_tile_index(tile_paths)

    buildings = gpd.read_file(args.labels)
    if args.label_field not in buildings.columns:
        raise ValueError(f"Label field '{args.label_field}' not found. Columns: {list(buildings.columns)}")
    if buildings.crs is None:
        raise ValueError("Input GPKG has no CRS.")
    buildings = buildings[buildings.geometry.notnull()].copy().to_crs("EPSG:3857")
    buildings["__label__"] = buildings[args.label_field].fillna("其他").astype(str).map(lambda x: x.strip() or "其他")

    labels = safe_class_names(sorted(buildings["__label__"].unique()))
    class_to_id = {name: idx for idx, name in enumerate(labels)}

    random.shuffle(tile_paths)
    samples = []
    scanned = 0
    if args.workers <= 1:
        for tile_path in tile_paths:
            scanned += 1
            sample = build_sample_from_tile(
                tile_path,
                tile_index,
                buildings,
                class_to_id,
                args.patch_tiles,
                args.min_area_pixels,
                args.simplify_pixels,
            )
            if sample is not None:
                samples.append(sample)
            if args.max_positive > 0 and len(samples) >= args.max_positive:
                break
            if scanned % 1000 == 0:
                print(f"Scanned {scanned}/{len(tile_paths)}, positive={len(samples)}")
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(tile_index, buildings, class_to_id, args.patch_tiles, args.min_area_pixels, args.simplify_pixels),
        ) as executor:
            for sample in executor.map(build_sample_from_tile_worker, tile_paths, chunksize=32):
                scanned += 1
                if sample is not None:
                    samples.append(sample)
                if args.max_positive > 0 and len(samples) >= args.max_positive:
                    break
                if scanned % 1000 == 0:
                    print(f"Scanned {scanned}/{len(tile_paths)}, positive={len(samples)}")

    if not samples:
        raise RuntimeError("No YOLO samples generated. Check tile/GPKG overlap.")

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio)) if len(samples) > 1 else 0
    val_ids = set(sample_id for sample_id, _, _, _ in samples[:n_val])
    records = []
    for sample_id, image, lines, building_count in samples:
        split = "val" if sample_id in val_ids else "train"
        write_sample(image, lines, sample_id, split, out_dir)
        records.append({"sample_id": sample_id, "split": split, "instances": len(lines), "building_count": building_count})

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(labels)}",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(labels)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    metadata = {
        "tiles": str(Path(args.tiles).resolve()),
        "labels": str(Path(args.labels).resolve()),
        "label_field": args.label_field,
        "class_names": labels,
        "class_to_id": class_to_id,
        "patch_tiles": args.patch_tiles,
        "samples": len(samples),
        "train_samples": len(samples) - n_val,
        "val_samples": n_val,
        "tiles_scanned": scanned,
        "workers": args.workers,
    }
    (out_dir / "metadata" / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metadata" / "samples.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared YOLO segmentation dataset: {out_dir}")
    print(f"Samples={len(samples)}, train={len(samples) - n_val}, val={n_val}, classes={class_to_id}")
    print(f"Data YAML: {data_yaml}")


if __name__ == "__main__":
    main()
