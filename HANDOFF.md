# Nansha z17 Imagery Handoff

## Final Status

- Status: complete
- Source: Tianditu `img_w`
- Area: Nansha bbox `113.43,22.56,113.70,22.88`
- Zoom: `17` meter-level imagery
- Tile matrix range: `x=106834..106932`, `y=56974..57101`
- Tiles downloaded: `12,672 / 12,672`
- Missing tiles: `0`
- Final GeoTIFF: `data/tianditu/nansha_z17/nansha_img_w_z17.tif`
- Full delivery directory: `data/tianditu/nansha_z17/`

## GeoTIFF Specs

- Size: `25,344 x 32,768`
- Bands: `3`
- Data type: `uint8`
- CRS: `EPSG:3857`
- Compression: JPEG
- Tiled: yes, `256 x 256`
- Internal overviews: `2, 4, 8, 16, 32, 64, 128`

## Validation

Run:

```bash
.venv/bin/python scripts/check_tianditu_nansha.py --zoom 17
```

Expected:

```json
{
  "tile_count_total": 12672,
  "downloaded": 12672,
  "missing": 0,
  "complete": true
}
```

## Notes

- The final GeoTIFF is ready for QGIS/ArcGIS/rasterio.
- The downloader uses `TIANDITU_TKS_BROWSER` first when present.
- Browser-side Tianditu keys are requested with a browser-like `Referer` header.
- Downloading used global request throttling and stopped automatically on earlier
  `429` rate-limit responses.

