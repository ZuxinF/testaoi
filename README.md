# Nansha Tianditu Imagery

建筑物功能分割训练流程见：

[BUILDING_SEG_PIPELINE.md](BUILDING_SEG_PIPELINE.md)

LaRSE 迁移推理的独立 venv 环境说明见：

[LARSE_VENV_README.md](LARSE_VENV_README.md)

YOLO26 建筑物功能实例分割尝试见：

[YOLO26_README.md](YOLO26_README.md)

Tools for downloading high-resolution Tianditu imagery tiles for Nansha,
then optionally stitching the downloaded tiles into a georeferenced GeoTIFF.

## 1. Configure

Put your Tianditu API key in `.env`:

```env
TIANDITU_TKS_BROWSER=your_browser_side_tianditu_tk
```

Use a Tianditu key that is authorized for map tile services / browser-side map
access. A server-side API key can return `403` with `权限类型错误` for `img_w`
tiles.

`TIANDITU_TKS_BROWSER` is preferred over `TIANDITU_TKS` when both are present.
Browser-side keys are requested with a browser-like `Referer` header. The
default is:

```env
TIANDITU_REFERER=https://map.tianditu.gov.cn/
```

Multiple authorized keys can be separated with commas:

```env
TIANDITU_TKS=tk1,tk2,tk3
```

Useful defaults:

```env
TIANDITU_LAYER=img_w
TIANDITU_ZOOM=17
NANSHA_BBOX=113.43,22.56,113.70,22.88
TILE_MAX_WORKERS=5
TILE_REQUEST_DELAY=0.25
TILE_OUTPUT_DIR=data/tianditu/nansha_z17
```

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Estimate without using quota

```bash
python scripts/download_tianditu_nansha.py --zoom 18 --dry-run
```

For lower quota pressure, use z17:

```bash
python scripts/download_tianditu_nansha.py --zoom 17 --dry-run
```

## 4. Download in resumable batches

For the current Nansha bbox, dry-run estimates are:

- z17: 12,672 tiles, 25,344 x 32,768 pixels
- z18: 50,038 tiles, 50,432 x 65,024 pixels

z17 is the practical full-area baseline. z18 is the highest public-quality
option, but it is much larger and should be downloaded in smaller batches or
used for priority areas first. Use `--limit` to download a safe daily chunk.
Re-run the same command later; existing tiles are skipped.

```bash
python scripts/download_tianditu_nansha.py --zoom 18 --limit 900
```

If Tianditu returns `429` / `该tk已限流`, wait and re-run the same command.
The downloader skips existing tiles and resumes from the next missing tile.
For a slower unattended run, allow one cooldown and retry the rate-limited tiles:

```bash
python scripts/download_tianditu_nansha.py --zoom 18 --limit 900 --workers 2 --delay 1 --limit-error-cooldown 1800
```

For z17:

```bash
python scripts/download_tianditu_nansha.py --zoom 17 --limit 900
```

If HTTPS is unstable on the current network, use Tianditu over HTTP:

```bash
python scripts/download_tianditu_nansha.py --zoom 17 --limit 900 --scheme http
```

The downloader ignores shell proxy variables by default, because local proxies
can break Tianditu's WAF/TLS handshake. Add `--use-env-proxy` only if your
network requires it.

## 5. Build GeoTIFF mosaic

After the tiles are downloaded:

```bash
python scripts/download_tianditu_nansha.py --zoom 18 --skip-download --mosaic
```

The output is written under `data/tianditu/nansha_z18/` by default.

## Notes

- `img_w` uses Web Mercator tiles and the GeoTIFF is written as `EPSG:3857`.
- The bbox covers Nansha broadly; it includes some surrounding water/edge area.
- The script never prints your Tianditu key.
- Use Tianditu data according to your account quota and service terms.
# testaoi
