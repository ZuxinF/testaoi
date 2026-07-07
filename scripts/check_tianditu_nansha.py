#!/usr/bin/env python3
"""Check local Tianditu tile completeness for the configured Nansha bbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from download_tianditu_nansha import (
    DEFAULT_BBOX,
    DEFAULT_LAYER,
    IMAGE_EXTENSIONS,
    load_env,
    parse_bbox,
    tile_range_for_bbox,
)


def tile_exists(base: Path, zoom: int, x: int, y: int) -> bool:
    stem = base / "tiles" / str(zoom) / str(x) / str(y)
    return any(stem.with_suffix(ext).exists() for ext in IMAGE_EXTENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check downloaded Tianditu tiles.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--bbox", default=None)
    parser.add_argument("--zoom", type=int, default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--write-missing", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = load_env(Path(args.env_file))

    bbox = parse_bbox(args.bbox or env.get("NANSHA_BBOX", DEFAULT_BBOX))
    zoom = args.zoom if args.zoom is not None else int(env.get("TIANDITU_ZOOM", "17"))
    layer = args.layer or env.get("TIANDITU_LAYER", DEFAULT_LAYER)
    out_dir = Path(
        args.out_dir or env.get("TILE_OUTPUT_DIR", f"data/tianditu/nansha_z{zoom}")
    )

    tile_range = tile_range_for_bbox(bbox, zoom)
    missing: list[dict[str, int]] = []
    downloaded = 0
    for y in range(tile_range.y_min, tile_range.y_max + 1):
        for x in range(tile_range.x_min, tile_range.x_max + 1):
            if tile_exists(out_dir, zoom, x, y):
                downloaded += 1
            else:
                missing.append({"x": x, "y": y})

    result = {
        "layer": layer,
        "bbox_lonlat": bbox,
        "zoom": zoom,
        "tile_count_total": tile_range.count,
        "downloaded": downloaded,
        "missing": len(missing),
        "complete": not missing,
        "first_missing": missing[0] if missing else None,
        "last_missing": missing[-1] if missing else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.write_missing:
        Path(args.write_missing).write_text(
            json.dumps(missing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

