from __future__ import annotations

import math
from pathlib import Path

import rasterio
from rasterio.transform import from_bounds


WEB_MERCATOR_HALF_WORLD = 20037508.342789244


def tile_bounds_3857(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """Return XYZ Web Mercator tile bounds as left, bottom, right, top."""
    n = 2 ** z
    tile_size = (WEB_MERCATOR_HALF_WORLD * 2) / n
    left = -WEB_MERCATOR_HALF_WORLD + x * tile_size
    right = left + tile_size
    top = WEB_MERCATOR_HALF_WORLD - y * tile_size
    bottom = top - tile_size
    return left, bottom, right, top


def parse_xyz_tile_path(path: str | Path) -> tuple[int, int, int]:
    """Parse .../tiles/z/x/y.jpg as z, x, y."""
    p = Path(path)
    y = int(p.stem)
    x = int(p.parent.name)
    z = int(p.parent.parent.name)
    return z, x, y


def tile_transform(x: int, y: int, z: int, width: int, height: int):
    return from_bounds(*tile_bounds_3857(x, y, z), width=width, height=height)

