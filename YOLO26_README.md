# YOLO26 建筑物功能实例分割尝试

这条链路用于先快速评估 YOLO26 在当前天地图瓦片 + 建筑物 GPKG 标注上的效果。

YOLO26 的 `-seg` 模型是实例分割模型，适合输出：

```text
每栋建筑的 polygon/mask + Function 类别
```

这和之前 `tiny_unet` 的语义分割不同：YOLO 会把每个建筑作为一个实例来学，验证指标主要看 box mAP 和 mask mAP。

## 一、安装依赖

在 `footprint/testaoi` 主目录：

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install ultralytics
```

检查：

```bash
python - <<'PY'
import torch
import ultralytics
import geopandas
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("ultralytics:", ultralytics.__version__)
print("ok")
PY
```

## 二、从 tiles + GPKG 生成 YOLO segmentation 数据集

先生成一个小样本调试集：

```bash
python -m building_seg.prepare_yolo_seg_from_tiles \
  --tiles data/tianditu/nansha_z18/tiles \
  --labels data/南沙区建筑物.gpkg \
  --out data/yolo26_seg_tiles_512_debug \
  --label-field Function \
  --patch-tiles 2 \
  --max-positive 100 \
  --val-ratio 0.2
```

输出结构：

```text
data/yolo26_seg_tiles_512_debug/
  data.yaml
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  metadata/dataset.json
  metadata/samples.json
```

YOLO segmentation 的每行 label 格式是：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

其中 `x/y` 都是 `0~1` 的归一化坐标。

## 三、YOLO26 冒烟训练

先用 nano 版跑 1 个 epoch，确认格式能训练：

```bash
yolo segment train \
  model=yolo26n-seg.pt \
  data=data/yolo26_seg_tiles_512_debug/data.yaml \
  imgsz=512 \
  epochs=1 \
  batch=4 \
  device=0 \
  project=data/yolo26_runs \
  name=debug_yolo26n_seg
```

如果没有 GPU，把 `device=0` 改成：

```text
device=cpu
```

## 四、验证精度

训练完成后验证：

```bash
yolo segment val \
  model=data/yolo26_runs/debug_yolo26n_seg/weights/best.pt \
  data=data/yolo26_seg_tiles_512_debug/data.yaml \
  imgsz=512 \
  device=0
```

重点看：

```text
Box(P/R/mAP50/mAP50-95)
Mask(P/R/mAP50/mAP50-95)
```

对我们来说，`Mask mAP50` 和 `Mask mAP50-95` 更重要。

## 五、较大样本训练

调通后再扩大：

```bash
python -m building_seg.prepare_yolo_seg_from_tiles \
  --tiles data/tianditu/nansha_z18/tiles \
  --labels data/南沙区建筑物.gpkg \
  --out data/yolo26_seg_tiles_512_train \
  --label-field Function \
  --patch-tiles 2 \
  --max-positive 5000 \
  --val-ratio 0.2
```

训练：

```bash
yolo segment train \
  model=yolo26n-seg.pt \
  data=data/yolo26_seg_tiles_512_train/data.yaml \
  imgsz=512 \
  epochs=50 \
  batch=8 \
  device=0 \
  project=data/yolo26_runs \
  name=yolo26n_seg_512
```

如果显存足够，可以试：

```text
model=yolo26s-seg.pt
```

## 六、注意事项

- 当前 GT 有漏标，所以不要随便加入大量“无建筑负样本”，否则会把真实未标建筑当背景训练。
- YOLO 是实例分割，适合评估每栋建筑 polygon 的检测和轮廓质量。
- 如果目标是整图语义分割 mask，`tiny_unet/LaRSE/SegFormer` 这类更直接；如果目标是输出建筑 polygon，YOLO 更贴近最终交付。
- YOLO26 权重会在第一次使用 `yolo26n-seg.pt` 时自动从 Ultralytics 下载。

