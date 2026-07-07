# Building Segmentation Pipeline

This is a minimal, swappable-model pipeline for:

```text
remote sensing image + building GPKG labels
-> image/mask patches
-> segmentation model checkpoint
-> predicted class mask
-> polygon GPKG with Function labels
```

It is intended for debugging the data interface locally with
`data/mock_buildings.gpkg`, then replacing the label path with the private
building GPKG on the target machine.

## Environment

The current tested environment is:

```bash
conda activate cityfm
cd /home/user/code/int/my/paqu
```

The pipeline uses `geopandas`, `rasterio`, `shapely`, `PIL`, and `torch`.

## 1. Prepare Patches

```bash
python -m building_seg.prepare_seg_dataset \
  --image /home/user/code/int/my/paqu/data/nansha_img_w_z18.tif \
  --labels /home/user/code/int/my/paqu/data/mock_buildings.gpkg \
  --out /home/user/code/int/my/paqu/data/building_seg_debug \
  --label-field Function \
  --patch-size 512 \
  --max-positive 50 \
  --negative 10 \
  --val-ratio 0.2
```

Output structure:

```text
data/building_seg_debug/
  images/*.png
  masks/*.png
  splits/train.txt
  splits/val.txt
  metadata/dataset.json
```

The mask uses `0` for background and positive class IDs for `Function` values.
The class map is stored in `metadata/dataset.json`.

## 2. Train Debug Model

```bash
python -m building_seg.train \
  --dataset /home/user/code/int/my/paqu/data/building_seg_debug \
  --out /home/user/code/int/my/paqu/data/building_seg_debug/checkpoints/tiny_unet_debug.pt \
  --model tiny_unet \
  --epochs 2 \
  --batch-size 4 \
  --base-channels 8
```

This default `tiny_unet` is only for pipeline debugging. Accuracy is not
meaningful with the small mock label set.

## 3. Predict and Polygonize

For debug, run inference on a small raster window:

```bash
python -m building_seg.predict_to_polygon \
  --image /home/user/code/int/my/paqu/data/nansha_img_w_z18.tif \
  --checkpoint /home/user/code/int/my/paqu/data/building_seg_debug/checkpoints/tiny_unet_debug.pt \
  --out-mask /home/user/code/int/my/paqu/data/building_seg_debug/predictions/debug_pred_mask.tif \
  --out-gpkg /home/user/code/int/my/paqu/data/building_seg_debug/predictions/debug_pred_polygons.gpkg \
  --window 19558,6981,512,512 \
  --tile-size 512 \
  --overlap 0
```

Outputs:

```text
debug_pred_mask.tif
debug_pred_polygons.gpkg
debug_pred_polygons.classes.json
```

The polygon GPKG has:

```text
class_id
Function
geometry
```

## Model Replacement Interface

Models are registered in `building_seg/models.py`.

To replace `tiny_unet`, add a new class:

```python
@register_model("your_model")
class YourModel(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        ...

    def forward(self, x):
        # x: Bx3xHxW float tensor in [0, 1]
        # return: Bxnum_classesxHxW logits
        return logits
```

Then train with:

```bash
python -m building_seg.train --model your_model ...
```

The checkpoint stores `model_name`, `model_kwargs`, and `class_names`, so
`predict_to_polygon.py` can load the correct model automatically.

## Migration Notes

When moving to the private 200k-building dataset:

- Replace `--labels` with the private GPKG path.
- Keep `--label-field Function` if the field name is unchanged.
- Increase `--max-positive` substantially, or extend sampling to tile the full
  raster.
- Keep the imagery and vector labels in any CRS; the script reprojects vectors
  to the raster CRS before rasterization.
- Current full-image inference is intentionally guarded for large rasters.
  Use `--window` for debugging or extend `predict_to_polygon.py` with streaming
  tiled GeoTIFF writing for production-scale full-scene prediction.
