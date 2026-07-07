#!/usr/bin/env python3
"""Download Tianditu imagery tiles for Nansha and optionally build a GeoTIFF.

The script reads TIANDITU_TKS_BROWSER / TIANDITU_TKS and other defaults from
.env, but never prints the key. Use --dry-run first to estimate tile count and
output size.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import requests


DEFAULT_BBOX = "113.43,22.56,113.70,22.88"
DEFAULT_LAYER = "img_w"
DEFAULT_SCHEME = "https"
DEFAULT_BROWSER_REFERER = "https://map.tianditu.gov.cn/"
DEFAULT_TILE_SIZE = 256
WEB_MERCATOR_LIMIT = 85.05112878
WEB_MERCATOR_ORIGIN_SHIFT = 20037508.342789244
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclasses.dataclass(frozen=True)
class TileRange:
    zoom: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def x_count(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def y_count(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def count(self) -> int:
        return self.x_count * self.y_count

    @property
    def width_px(self) -> int:
        return self.x_count * DEFAULT_TILE_SIZE

    @property
    def height_px(self) -> int:
        return self.y_count * DEFAULT_TILE_SIZE

    def iter_tiles(self) -> Iterable[tuple[int, int]]:
        for y in range(self.y_min, self.y_max + 1):
            for x in range(self.x_min, self.x_max + 1):
                yield x, y


@dataclasses.dataclass(frozen=True)
class DownloadResult:
    x: int
    y: int
    ok: bool
    skipped: bool
    path: str | None
    error: str | None = None


def is_limit_or_auth_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    markers = (
        "http 401",
        "http 403",
        "http 429",
        "quota",
        "limit",
        "too many",
        "invalid",
        "key",
        "tk",
        "权限",
        "配额",
        "频率",
        "超限",
        "过期",
    )
    return any(marker in lowered for marker in markers)


class RateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        if self.delay_seconds <= 0:
            return

        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_time = now + self.delay_seconds


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be lon_min,lat_min,lon_max,lat_max")

    lon_min, lat_min, lon_max, lat_max = [float(part) for part in parts]
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("bbox min values must be smaller than max values")
    return lon_min, lat_min, lon_max, lat_max


def parse_tks(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_tk_value(args_tk: str | None, env: dict[str, str]) -> tuple[str | None, str]:
    if args_tk:
        return args_tk, "--tk"
    browser_tks = env.get("TIANDITU_TKS_BROWSER")
    if browser_tks:
        return browser_tks, "TIANDITU_TKS_BROWSER"
    return env.get("TIANDITU_TKS"), "TIANDITU_TKS"


def sanitize_error(error: str, tks: list[str]) -> str:
    sanitized = error
    for tk in tks:
        if tk:
            sanitized = sanitized.replace(tk, "<redacted>")
    sanitized = re.sub(r"([?&]tk=)[^&\\s)]+", r"\1<redacted>", sanitized)
    return sanitized


def clamp_lat(lat: float) -> float:
    return max(-WEB_MERCATOR_LIMIT, min(WEB_MERCATOR_LIMIT, lat))


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = clamp_lat(lat)
    n = 1 << zoom
    x = int((lon + 180.0) / 360.0 * n)
    sin_lat = math.sin(math.radians(lat))
    y = int((0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_range_for_bbox(bbox: tuple[float, float, float, float], zoom: int) -> TileRange:
    lon_min, lat_min, lon_max, lat_max = bbox
    x_left, y_top = lonlat_to_tile(lon_min, lat_max, zoom)
    x_right, y_bottom = lonlat_to_tile(lon_max, lat_min, zoom)
    return TileRange(
        zoom=zoom,
        x_min=min(x_left, x_right),
        x_max=max(x_left, x_right),
        y_min=min(y_top, y_bottom),
        y_max=max(y_top, y_bottom),
    )


def build_tianditu_url(layer: str, zoom: int, x: int, y: int, tk: str, scheme: str) -> str:
    if layer.endswith("_w"):
        matrix_set = "w"
        wmts_layer = layer[:-2]
    elif layer.endswith("_c"):
        matrix_set = "c"
        wmts_layer = layer[:-2]
    else:
        matrix_set = "w"
        wmts_layer = layer

    server = f"t{random.randint(0, 7)}"
    query = urlencode(
        {
            "SERVICE": "WMTS",
            "REQUEST": "GetTile",
            "VERSION": "1.0.0",
            "LAYER": wmts_layer,
            "STYLE": "default",
            "TILEMATRIXSET": matrix_set,
            "FORMAT": "tiles",
            "TILEMATRIX": zoom,
            "TILEROW": y,
            "TILECOL": x,
            "tk": tk,
        }
    )
    return f"{scheme}://{server}.tianditu.gov.cn/{layer}/wmts?{query}"


def tile_dir(out_dir: Path, zoom: int, x: int) -> Path:
    return out_dir / "tiles" / str(zoom) / str(x)


def find_tile_path(out_dir: Path, zoom: int, x: int, y: int) -> Path | None:
    base = tile_dir(out_dir, zoom, x) / str(y)
    for ext in IMAGE_EXTENSIONS:
        path = base.with_suffix(ext)
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def detect_image_extension(content: bytes) -> str | None:
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return None


def download_tile(
    *,
    out_dir: Path,
    layer: str,
    scheme: str,
    zoom: int,
    x: int,
    y: int,
    tks: list[str],
    timeout: float,
    retries: int,
    overwrite: bool,
    rate_limiter: RateLimiter,
    use_env_proxy: bool,
    referer: str | None,
) -> DownloadResult:
    existing = find_tile_path(out_dir, zoom, x, y)
    if existing and not overwrite:
        return DownloadResult(x=x, y=y, ok=True, skipped=True, path=str(existing))

    last_error = "unknown error"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/125.0 Safari/537.36 paqu-tianditu-imagery/1.0"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
        headers["Origin"] = referer.rstrip("/")
    session = requests.Session()
    session.trust_env = use_env_proxy

    for attempt in range(retries + 1):
        tk = tks[(x + y + attempt) % len(tks)]
        url = build_tianditu_url(layer, zoom, x, y, tk, scheme)
        try:
            rate_limiter.wait()
            response = session.get(url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                snippet = response.text[:180].replace("\n", " ")
                last_error = sanitize_error(f"HTTP {response.status_code}: {snippet}", tks)
                time.sleep(min(2.0 * (attempt + 1), 10.0))
                continue

            ext = detect_image_extension(response.content)
            if not ext:
                snippet = response.text[:120].replace("\n", " ")
                last_error = f"not an image: {snippet}"
                time.sleep(min(2.0 * (attempt + 1), 10.0))
                continue

            target_dir = tile_dir(out_dir, zoom, x)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{y}{ext}"
            tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
            tmp_path.write_bytes(response.content)
            tmp_path.replace(target_path)
            return DownloadResult(x=x, y=y, ok=True, skipped=False, path=str(target_path))
        except Exception as exc:  # noqa: BLE001 - keep downloader resilient.
            last_error = sanitize_error(str(exc), tks)
            time.sleep(min(2.0 * (attempt + 1), 10.0))

    return DownloadResult(x=x, y=y, ok=False, skipped=False, path=None, error=last_error)


def write_manifest(
    *,
    out_dir: Path,
    bbox: tuple[float, float, float, float],
    layer: str,
    tile_range: TileRange,
    planned_downloads: int,
    completed: int,
    failed: list[DownloadResult],
    stopped_early: bool = False,
    stop_reason: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layer": layer,
        "bbox_lonlat": bbox,
        "zoom": tile_range.zoom,
        "tile_range": dataclasses.asdict(tile_range),
        "tile_count_total": tile_range.count,
        "pixel_size": {
            "width": tile_range.width_px,
            "height": tile_range.height_px,
        },
        "planned_downloads_this_run": planned_downloads,
        "completed_this_run": completed,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "failed_this_run": [
            {"x": item.x, "y": item.y, "error": item.error} for item in failed
        ],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_mosaic(
    *,
    out_dir: Path,
    output_path: Path,
    tile_range: TileRange,
    overwrite: bool,
    compress: str,
    jpeg_quality: int,
) -> None:
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import Affine
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError(
            "mosaic requires rasterio and numpy. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; use --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolution = (WEB_MERCATOR_ORIGIN_SHIFT * 2.0) / (
        DEFAULT_TILE_SIZE * (1 << tile_range.zoom)
    )
    left = -WEB_MERCATOR_ORIGIN_SHIFT + tile_range.x_min * DEFAULT_TILE_SIZE * resolution
    top = WEB_MERCATOR_ORIGIN_SHIFT - tile_range.y_min * DEFAULT_TILE_SIZE * resolution
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)

    profile = {
        "driver": "GTiff",
        "height": tile_range.height_px,
        "width": tile_range.width_px,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": transform,
        "tiled": True,
        "blockxsize": DEFAULT_TILE_SIZE,
        "blockysize": DEFAULT_TILE_SIZE,
        "BIGTIFF": "IF_SAFER",
    }
    compress = compress.upper()
    if compress != "NONE":
        profile["compress"] = compress.lower()
    if compress == "JPEG":
        profile["photometric"] = "YCBCR"
        profile["jpeg_quality"] = jpeg_quality

    missing = 0
    written = 0
    with rasterio.open(output_path, "w", **profile) as dst:
        for y in range(tile_range.y_min, tile_range.y_max + 1):
            for x in range(tile_range.x_min, tile_range.x_max + 1):
                path = find_tile_path(out_dir, tile_range.zoom, x, y)
                if not path:
                    missing += 1
                    continue
                try:
                    with rasterio.open(path) as src:
                        data = src.read()
                except Exception:
                    missing += 1
                    continue

                if data.shape[1] != DEFAULT_TILE_SIZE or data.shape[2] != DEFAULT_TILE_SIZE:
                    missing += 1
                    continue
                if data.shape[0] >= 3:
                    data = data[:3]
                elif data.shape[0] == 1:
                    data = np.repeat(data, 3, axis=0)
                else:
                    missing += 1
                    continue

                window = Window(
                    col_off=(x - tile_range.x_min) * DEFAULT_TILE_SIZE,
                    row_off=(y - tile_range.y_min) * DEFAULT_TILE_SIZE,
                    width=DEFAULT_TILE_SIZE,
                    height=DEFAULT_TILE_SIZE,
                )
                dst.write(data, window=window)
                written += 1

    print(f"mosaic written: {output_path}")
    print(f"tiles written into mosaic: {written}; missing/skipped tiles: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Tianditu WMTS imagery tiles for Nansha."
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--tk", default=None, help="Comma-separated Tianditu tk values")
    parser.add_argument("--bbox", default=None, help="lon_min,lat_min,lon_max,lat_max")
    parser.add_argument("--zoom", type=int, default=None, help="Tile zoom, usually 17 or 18")
    parser.add_argument("--layer", default=None, help="Tianditu layer, default img_w")
    parser.add_argument("--scheme", default=None, choices=("http", "https"), help="Request scheme")
    parser.add_argument("--referer", default=None, help="Referer header for browser-side keys")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use HTTP(S)_PROXY from the shell environment. Off by default for Tianditu.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Downloader worker count")
    parser.add_argument("--delay", type=float, default=None, help="Global delay between requests")
    parser.add_argument("--timeout", type=float, default=20.0, help="Request timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries per tile")
    parser.add_argument("--limit", type=int, default=None, help="Max new tiles to download")
    parser.add_argument(
        "--stop-after-limit-errors",
        type=int,
        default=3,
        help="Stop after this many consecutive auth/quota/rate-limit errors",
    )
    parser.add_argument(
        "--limit-error-cooldown",
        type=float,
        default=0.0,
        help=(
            "Sleep this many seconds after consecutive auth/quota/rate-limit "
            "errors, then retry those tiles. Default 0 stops early."
        ),
    )
    parser.add_argument(
        "--max-limit-cooldowns",
        type=int,
        default=1,
        help=(
            "Max cooldown pauses in one run when --limit-error-cooldown is set; "
            "0 means unlimited."
        ),
    )
    parser.add_argument(
        "--stop-after-consecutive-failures",
        type=int,
        default=12,
        help="Stop after this many consecutive failed tiles",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print tile estimate")
    parser.add_argument("--skip-download", action="store_true", help="Do not download tiles")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing tile/mosaic files")
    parser.add_argument("--mosaic", action="store_true", help="Build GeoTIFF mosaic after download")
    parser.add_argument("--mosaic-output", default=None, help="Output GeoTIFF path")
    parser.add_argument(
        "--mosaic-compress",
        default="JPEG",
        choices=("JPEG", "DEFLATE", "LZW", "NONE"),
        help="GeoTIFF compression",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for mosaic compression",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = load_env(Path(args.env_file))

    bbox_value = args.bbox or env.get("NANSHA_BBOX", DEFAULT_BBOX)
    bbox = parse_bbox(bbox_value)
    zoom = args.zoom if args.zoom is not None else int(env.get("TIANDITU_ZOOM", "18"))
    layer = args.layer or env.get("TIANDITU_LAYER", DEFAULT_LAYER)
    scheme = args.scheme or env.get("TIANDITU_SCHEME", DEFAULT_SCHEME)
    out_dir = Path(
        args.out_dir
        or env.get("TILE_OUTPUT_DIR", f"data/tianditu/nansha_z{zoom}")
    )
    workers = args.workers if args.workers is not None else int(env.get("TILE_MAX_WORKERS", "5"))
    delay = args.delay if args.delay is not None else float(env.get("TILE_REQUEST_DELAY", "0.25"))
    tk_value, tk_source = get_tk_value(args.tk, env)
    tks = parse_tks(tk_value)
    referer = args.referer or env.get("TIANDITU_REFERER")
    if not referer and tk_source == "TIANDITU_TKS_BROWSER":
        referer = DEFAULT_BROWSER_REFERER

    tile_range = tile_range_for_bbox(bbox, zoom)

    print(f"layer: {layer}")
    print(f"scheme: {scheme}")
    print(f"bbox: {bbox}")
    print(f"zoom: {zoom}")
    print(f"tile range: x {tile_range.x_min}-{tile_range.x_max}, y {tile_range.y_min}-{tile_range.y_max}")
    print(f"tile count: {tile_range.count} ({tile_range.x_count} x {tile_range.y_count})")
    print(f"pixel size: {tile_range.width_px} x {tile_range.height_px}")
    print(f"output: {out_dir}")
    print(f"tk source: {tk_source if tks else 'none'}")
    print(f"referer enabled: {bool(referer)}")

    if args.dry_run:
        print("dry run only; no requests were made")
        return 0

    if not args.skip_download and not tks:
        print("ERROR: TIANDITU_TKS is empty. Fill .env or pass --tk.", file=sys.stderr)
        return 2

    planned_tiles = []
    if not args.skip_download:
        for x, y in tile_range.iter_tiles():
            if args.overwrite or not find_tile_path(out_dir, zoom, x, y):
                planned_tiles.append((x, y))

        if args.limit is not None:
            planned_tiles = planned_tiles[: args.limit]

        print(f"new tiles planned this run: {len(planned_tiles)}")
        if planned_tiles:
            out_dir.mkdir(parents=True, exist_ok=True)
            rate_limiter = RateLimiter(delay)
            completed = 0
            skipped = 0
            failed: list[DownloadResult] = []
            processed = 0
            submitted = 0
            consecutive_failures = 0
            consecutive_limit_errors = 0
            limit_cooldowns = 0
            limit_error_retry_tiles: list[tuple[int, int]] = []
            stop_reason = None
            futures: dict[concurrent.futures.Future[DownloadResult], tuple[int, int]] = {}

            def submit_next(
                executor: concurrent.futures.ThreadPoolExecutor,
            ) -> bool:
                nonlocal submitted
                if submitted >= len(planned_tiles):
                    return False
                x, y = planned_tiles[submitted]
                futures[
                    executor.submit(
                        download_tile,
                        out_dir=out_dir,
                        layer=layer,
                        scheme=scheme,
                        zoom=zoom,
                        x=x,
                        y=y,
                        tks=tks,
                        timeout=args.timeout,
                        retries=args.retries,
                        overwrite=args.overwrite,
                        rate_limiter=rate_limiter,
                        use_env_proxy=args.use_env_proxy,
                        referer=referer,
                    )
                ] = (x, y)
                submitted += 1
                return True

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
            try:
                for _ in range(min(workers, len(planned_tiles))):
                    submit_next(executor)

                while futures:
                    done, _pending = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    cooldown_reason = None
                    for future in done:
                        futures.pop(future)
                        processed += 1
                        result = future.result()
                        if result.ok:
                            completed += 1
                            consecutive_failures = 0
                            consecutive_limit_errors = 0
                            limit_error_retry_tiles.clear()
                            if result.skipped:
                                skipped += 1
                        else:
                            failed.append(result)
                            consecutive_failures += 1
                            if is_limit_or_auth_error(result.error):
                                consecutive_limit_errors += 1
                                limit_error_retry_tiles.append((result.x, result.y))
                            else:
                                consecutive_limit_errors = 0
                                limit_error_retry_tiles.clear()

                        if (
                            args.stop_after_limit_errors > 0
                            and consecutive_limit_errors >= args.stop_after_limit_errors
                        ):
                            can_cooldown = (
                                args.limit_error_cooldown > 0
                                and (
                                    args.max_limit_cooldowns == 0
                                    or limit_cooldowns < args.max_limit_cooldowns
                                )
                            )
                            if can_cooldown:
                                cooldown_reason = (
                                    "hit consecutive Tianditu auth/quota/rate-limit "
                                    f"errors: {consecutive_limit_errors}"
                                )
                            else:
                                stop_reason = (
                                    "stopped after consecutive Tianditu auth/quota/rate-limit "
                                    f"errors: {consecutive_limit_errors}"
                                )
                            break

                        if (
                            args.stop_after_consecutive_failures > 0
                            and consecutive_failures >= args.stop_after_consecutive_failures
                        ):
                            stop_reason = (
                                "stopped after consecutive failed tiles: "
                                f"{consecutive_failures}"
                            )
                            break

                        if processed == 1 or processed % 50 == 0 or processed == len(planned_tiles):
                            print(
                                f"progress: {processed}/{len(planned_tiles)} "
                                f"ok={completed} skipped={skipped} failed={len(failed)}"
                            )

                    if cooldown_reason:
                        retry_set = set(limit_error_retry_tiles)
                        failed = [
                            item
                            for item in failed
                            if (item.x, item.y) not in retry_set
                        ]
                        planned_tiles[submitted:submitted] = limit_error_retry_tiles
                        limit_cooldowns += 1
                        print(
                            f"{cooldown_reason}; cooling down for "
                            f"{args.limit_error_cooldown:g}s "
                            f"({limit_cooldowns}/{args.max_limit_cooldowns or 'unlimited'})"
                        )
                        time.sleep(args.limit_error_cooldown)
                        consecutive_failures = 0
                        consecutive_limit_errors = 0
                        limit_error_retry_tiles = []

                    if stop_reason:
                        print(stop_reason)
                        for pending in futures:
                            pending.cancel()
                        break

                    while len(futures) < workers and submit_next(executor):
                        pass
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

            if processed and processed % 50 != 0:
                print(
                    f"progress: {processed}/{len(planned_tiles)} "
                    f"ok={completed} skipped={skipped} failed={len(failed)}"
                )

            if stop_reason:
                print("download stopped early to avoid hammering the service")

            if not stop_reason:
                if processed == len(planned_tiles):
                    print(
                        f"progress: {processed}/{len(planned_tiles)} "
                        f"ok={completed} skipped={skipped} failed={len(failed)}"
                    )

            write_manifest(
                out_dir=out_dir,
                bbox=bbox,
                layer=layer,
                tile_range=tile_range,
                planned_downloads=len(planned_tiles),
                completed=completed,
                failed=failed,
                stopped_early=stop_reason is not None,
                stop_reason=stop_reason,
            )

            if failed:
                failed_path = out_dir / "failed_tiles.json"
                failed_path.write_text(
                    json.dumps(
                        [
                            {"x": item.x, "y": item.y, "error": item.error}
                            for item in failed
                        ],
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                print(f"failed tile list written: {failed_path}")
            else:
                failed_path = out_dir / "failed_tiles.json"
                if failed_path.exists():
                    failed_path.unlink()
        else:
            print("no new tiles to download")

    if args.mosaic:
        mosaic_output = Path(
            args.mosaic_output or out_dir / f"nansha_{layer}_z{zoom}.tif"
        )
        build_mosaic(
            out_dir=out_dir,
            output_path=mosaic_output,
            tile_range=tile_range,
            overwrite=args.overwrite,
            compress=args.mosaic_compress,
            jpeg_quality=args.jpeg_quality,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
