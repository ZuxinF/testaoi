from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare image/mask patches from imagery and building GPKG.")
    parser.add_argument("--image", required=True, help="Input georeferenced RGB image, e.g. nansha_img_w_z18.tif")
    parser.add_argument("--labels", required=True, help="Building labels GPKG with a Function field")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--label-field", default="Function")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--max-positive", type=int, default=80, help="Max positive patches centered on buildings")
    parser.add_argument("--negative", type=int, default=20, help="Random background patches")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def save_png(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def window_from_center(cx: float, cy: float, src, patch_size: int) -> Window:
    row, col = src.index(cx, cy)
    col_off = int(np.clip(col - patch_size // 2, 0, max(0, src.width - patch_size)))
    row_off = int(np.clip(row - patch_size // 2, 0, max(0, src.height - patch_size)))
    return Window(col_off, row_off, patch_size, patch_size)


def read_rgb_window(src, window: Window) -> np.ndarray:
    data = src.read([1, 2, 3], window=window, boundless=False)
    return np.transpose(data, (1, 2, 0))


def mask_for_window(src, window: Window, buildings: gpd.GeoDataFrame, class_to_id: dict[str, int]) -> np.ndarray:
    transform = src.window_transform(window)
    bounds = rasterio.windows.bounds(window, src.transform)
    patch_box = box(*bounds)
    subset = buildings[buildings.intersects(patch_box)]
    if subset.empty:
        return np.zeros((int(window.height), int(window.width)), dtype="uint8")

    shapes = [(geom, class_to_id[value]) for geom, value in zip(subset.geometry, subset["__label__"]) if geom and not geom.is_empty]
    mask = rasterize(
        shapes,
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return mask


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    for name in ["images", "masks", "splits", "metadata"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.image) as src:
        buildings = gpd.read_file(args.labels)
        if args.label_field not in buildings.columns:
            raise ValueError(f"Label field '{args.label_field}' not found in {args.labels}")

        buildings = buildings[buildings.geometry.notnull()].copy()
        buildings = buildings.to_crs(src.crs)
        raster_extent = box(*src.bounds)
        buildings = buildings[buildings.intersects(raster_extent)].copy()
        if buildings.empty:
            raise RuntimeError("No building geometries intersect the input image.")

        buildings["__label__"] = buildings[args.label_field].fillna("其他").astype(str)
        labels = sorted(buildings["__label__"].unique())
        class_names = ["background"] + labels
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        windows = []
        sampled = buildings.sample(n=min(args.max_positive, len(buildings)), random_state=args.seed)
        for geom in sampled.geometry:
            pt = geom.representative_point()
            windows.append(window_from_center(pt.x, pt.y, src, args.patch_size))

        for _ in range(args.negative):
            col = random.randint(0, max(0, src.width - args.patch_size))
            row = random.randint(0, max(0, src.height - args.patch_size))
            windows.append(Window(col, row, args.patch_size, args.patch_size))

        sample_ids = []
        for idx, window in enumerate(windows):
            image = read_rgb_window(src, window)
            mask = mask_for_window(src, window, buildings, class_to_id)

            if image.shape[:2] != (args.patch_size, args.patch_size):
                continue

            sid = f"patch_{idx:05d}"
            save_png(out_dir / "images" / f"{sid}.png", image)
            save_png(out_dir / "masks" / f"{sid}.png", mask)
            sample_ids.append(sid)

        random.shuffle(sample_ids)
        n_val = max(1, int(len(sample_ids) * args.val_ratio)) if len(sample_ids) > 1 else 0
        val_ids = sample_ids[:n_val]
        train_ids = sample_ids[n_val:]

        (out_dir / "splits" / "train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""))
        (out_dir / "splits" / "val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""))

        metadata = {
            "image": str(Path(args.image).resolve()),
            "labels": str(Path(args.labels).resolve()),
            "label_field": args.label_field,
            "patch_size": args.patch_size,
            "crs": str(src.crs),
            "class_names": class_names,
            "class_to_id": class_to_id,
            "total_samples": len(sample_ids),
            "train_samples": len(train_ids),
            "val_samples": len(val_ids),
            "intersecting_buildings": int(len(buildings)),
        }
        (out_dir / "metadata" / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(f"Prepared {len(sample_ids)} patches at {out_dir}")
    print(f"Classes: {class_to_id}")


if __name__ == "__main__":
    main()
