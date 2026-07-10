from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rasterio.features import shapes
from shapely.geometry import shape

from building_seg.prepare_seg_dataset_from_tiles import build_tile_index, find_tiles, load_mosaic


LARSE_LABELS = [
    "dense residential",
    "business",
    "commercial",
    "residential",
    "factory",
    "government",
    "hospital",
    "resort",
    "public",
    "school",
    "background",
    "others",
]

DEFAULT_TARGET_CLASS_NAMES = [
    "background",
    "仓储",
    "公共服务",
    "其他",
    "办公",
    "医疗",
    "商业",
    "居住",
    "工业",
    "教育",
]

DEFAULT_LARSE_TO_TARGET = {
    "dense residential": "居住",
    "business": "办公",
    "commercial": "商业",
    "residential": "居住",
    "factory": "工业",
    "government": "公共服务",
    "hospital": "医疗",
    "resort": "其他",
    "public": "公共服务",
    "school": "教育",
    "background": "background",
    "others": "其他",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LaRSE on XYZ tiles and export function polygons.")
    parser.add_argument("--tiles", default="data/tianditu/nansha_z18/tiles")
    parser.add_argument("--larse-dir", default=None, help="LaRSE project root. Auto-detects ../LaRSE and ../../LaRSE if omitted.")
    parser.add_argument("--larse-data-path", default=None, help="LaRSE datasets directory. Defaults to <larse-dir>/datasets.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to <larse-dir>/checkpoints/checkpoint_LARSE.ckpt.")
    parser.add_argument("--backbone", default="clip_vitb32_384")
    parser.add_argument("--dataset", default="buff1w")
    parser.add_argument("--class-json", default=None, help="Optional paqu metadata/dataset.json for target class order.")
    parser.add_argument("--out-gpkg", required=True)
    parser.add_argument("--out-mask-dir", default=None, help="Optional directory for remapped predicted ID masks.")
    parser.add_argument("--out-larse-mask-dir", default=None, help="Optional directory for raw LaRSE 1-12 masks.")
    parser.add_argument("--extensions", default=".jpg,.png,.jpeg")
    parser.add_argument("--patch-tiles", type=int, default=2, help="2 means 2x2 tiles -> 512x512 patch")
    parser.add_argument("--stride-tiles", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-area-pixels", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    resolve_larse_paths(args)
    return args


def resolve_larse_paths(args):
    cwd = Path.cwd()
    candidates = []
    if args.larse_dir:
        candidates.append(Path(args.larse_dir))
    candidates.extend(
        [
            cwd / ".." / "LaRSE",
            cwd / ".." / ".." / "LaRSE",
            Path(__file__).resolve().parents[2] / "LaRSE",
            Path(__file__).resolve().parents[3] / "LaRSE",
        ]
    )

    larse_dir = None
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "modules" / "lseg_module.py").exists():
            larse_dir = candidate
            break
    if larse_dir is None:
        tried = "\n".join(str(c.expanduser()) for c in candidates)
        raise FileNotFoundError(f"Could not locate LaRSE project root. Tried:\n{tried}")

    args.larse_dir = str(larse_dir)
    if args.larse_data_path is None:
        args.larse_data_path = str(larse_dir / "datasets")
    else:
        args.larse_data_path = str(Path(args.larse_data_path).expanduser().resolve())

    if args.checkpoint is None:
        args.checkpoint = str(larse_dir / "checkpoints" / "checkpoint_LARSE.ckpt")
    else:
        args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"LaRSE checkpoint not found: {args.checkpoint}")


def load_target_class_names(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_TARGET_CLASS_NAMES
    metadata = json.loads(Path(path).read_text())
    class_names = metadata.get("class_names")
    if not class_names or class_names[0] != "background":
        raise ValueError(f"{path} must contain class_names with background at index 0")
    return class_names


def make_remap(class_names: list[str]) -> np.ndarray:
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    remap = np.zeros(len(LARSE_LABELS), dtype=np.uint8)
    for idx, larse_name in enumerate(LARSE_LABELS):
        target_name = DEFAULT_LARSE_TO_TARGET[larse_name]
        remap[idx] = class_to_id.get(target_name, class_to_id.get("其他", 0))
    return remap


def load_larse_module(args):
    larse_dir = Path(args.larse_dir).resolve()
    if not larse_dir.exists():
        raise FileNotFoundError(f"LaRSE dir not found: {larse_dir}")
    larse_data_path = Path(args.larse_data_path).resolve()

    old_cwd = Path.cwd()
    sys.path.insert(0, str(larse_dir))
    os.chdir(larse_dir)
    try:
        register_buff1w_dataset(larse_dir)
        ensure_dummy_buff_dataset(larse_data_path)

        from modules.lseg_module import LSegModule

        module = LSegModule.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            map_location=args.device,
            data_path=args.larse_data_path,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=False,
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=0,
            batch_size=1,
            max_epochs=0,
            ignore_index=-1,
            dropout=0.0,
            scale_inv=True,
            augment=False,
            no_batchnorm=False,
            widehead=False,
            widehead_hr=False,
            arch_option=0,
            strict=True,
            block_depth=0,
            activation="lrelu",
        )
    finally:
        os.chdir(old_cwd)

    module = module.to(args.device).eval()
    return module, module.val_transform


def register_buff1w_dataset(larse_dir: Path):
    try:
        import encoding.datasets as enc_datasets
    except Exception as exc:
        raise RuntimeError("Failed to import encoding.datasets. Is torch-encoding installed?") from exc

    registry = getattr(enc_datasets, "datasets", None)
    if isinstance(registry, dict) and "buff1w" in registry:
        return

    buff_path = larse_dir / "buff1w.py"
    if not buff_path.exists():
        raise FileNotFoundError(f"Cannot register buff1w, missing file: {buff_path}")

    spec = importlib.util.spec_from_file_location("encoding.datasets.buff1w", buff_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load buff1w dataset module from {buff_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["encoding.datasets.buff1w"] = module
    spec.loader.exec_module(module)

    dataset_cls = module.BuFF1WChallengeDataset
    setattr(enc_datasets, "BuFF1WChallengeDataset", dataset_cls)
    if isinstance(registry, dict):
        registry["buff1w"] = dataset_cls
    else:
        raise RuntimeError("encoding.datasets has no datasets registry; cannot register buff1w dynamically.")


def ensure_dummy_buff_dataset(data_path: Path):
    """LaRSE builds train/val datasets during module init even for inference."""
    root = data_path / "BFF1WChallenge_for_LaRSE"
    pairs = [
        (root / "images" / "training" / "dummy.jpg", root / "annotations" / "training" / "dummy.png"),
        (root / "images" / "validation" / "dummy.jpg", root / "annotations" / "validation" / "dummy.png"),
    ]
    for image_path, mask_path in pairs:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(image_path)
        if not mask_path.exists():
            Image.fromarray(np.full((16, 16), 11, dtype=np.uint8)).save(mask_path)


def image_to_tensor(image: Image.Image, transform) -> torch.Tensor:
    return transform(image.convert("RGB")).unsqueeze(0)


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


def main():
    args = parse_args()
    target_class_names = load_target_class_names(args.class_json)
    remap = make_remap(target_class_names)

    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.lower().split(",")}
    tile_paths = find_tiles(Path(args.tiles), extensions)
    if not tile_paths:
        raise RuntimeError(f"No tile images found under {args.tiles}")
    tile_index = build_tile_index(tile_paths)

    module, transform = load_larse_module(args)

    for out_dir in [args.out_mask_dir, args.out_larse_mask_dir]:
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

    xyz_list = sorted(tile_index)
    min_x = min(x for _, x, _ in xyz_list)
    min_y = min(y for _, _, y in xyz_list)

    all_records = []
    processed = 0
    with torch.no_grad():
        for z, x, y in xyz_list:
            if (x - min_x) % args.stride_tiles or (y - min_y) % args.stride_tiles:
                continue
            image, bounds, transform_geo, xyz = load_mosaic(tile_index[(z, x, y)], tile_index, args.patch_tiles)
            if image is None:
                continue

            patch_id = f"z{z}_x{x}_y{y}_s{image.size[0]}"
            logits = module(image_to_tensor(image, transform).to(args.device))
            logits = logits.detach()
            if logits.shape[-2:] != (image.size[1], image.size[0]):
                logits = F.interpolate(logits, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False)
            larse_mask_zero_based = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            mask = remap[larse_mask_zero_based]

            if args.out_larse_mask_dir:
                Image.fromarray(larse_mask_zero_based + 1).save(Path(args.out_larse_mask_dir) / f"{patch_id}.png")
            if args.out_mask_dir:
                Image.fromarray(mask).save(Path(args.out_mask_dir) / f"{patch_id}.png")

            all_records.extend(polygonize_mask(mask, transform_geo, target_class_names, patch_id, args.min_area_pixels))
            processed += 1
            if processed % 20 == 0:
                print(f"Predicted {processed} patches, polygons={len(all_records)}")
            if args.limit is not None and processed >= args.limit:
                break

    gdf = gpd.GeoDataFrame(all_records, geometry="geometry", crs="EPSG:3857")
    out_path = Path(args.out_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", index=False)
    out_path.with_suffix(".classes.json").write_text(
        json.dumps(
            {
                "target_class_names": target_class_names,
                "larse_labels": LARSE_LABELS,
                "larse_to_target": DEFAULT_LARSE_TO_TARGET,
                "checkpoint": str(args.checkpoint),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Predicted patches: {processed}")
    print(f"Wrote polygons: {out_path} ({len(gdf)} features)")


if __name__ == "__main__":
    main()
