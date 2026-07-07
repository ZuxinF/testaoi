# 建筑物功能分割训练流程说明

这份文档说明如何把天地图遥感瓦片和建筑物 GPKG 标注接起来，生成训练样本，训练一个可替换的分割模型，并把推理结果转回 polygon GPKG。

整体流程：

```text
天地图瓦片 / 遥感影像
+ 建筑物 GPKG 标注
-> image/mask 训练样本
-> 分割模型 checkpoint
-> 预测类别 mask
-> polygon GPKG，包含 Function 字段
```

当前推荐优先使用 tiles 版本，因为 `data/tianditu/nansha_z18/tiles/18/{x}/{y}.jpg` 本身就是天然裁好的小图。

## 一、运行环境

在项目主目录运行：

```bash
cd /home/user/code/int/my/paqu
source .venv/bin/activate
pip install -r requirements.txt
```

检查依赖：

```bash
python -c "import torch, rasterio, geopandas, shapely, pyogrio; print('ok'); print(torch.__version__); print(torch.cuda.is_available())"
```

## 二、输入数据

推荐目录结构：

```text
data/tianditu/nansha_z18/tiles/18/{x}/{y}.jpg
data/南沙区建筑物.gpkg
```

说明：

- `tiles/18/{x}/{y}.jpg` 是 z18 天地图瓦片。
- 瓦片 JPG 本身没有 CRS 元数据，但路径里的 `z/x/y` 可以计算出 Web Mercator 范围。
- 建筑物 GPKG 可以是 EPSG:4326 或其他 CRS，脚本会自动重投影到 EPSG:3857。
- 默认使用 GPKG 中的 `Function` 字段作为建筑功能类别。

## 三、从 tiles 生成训练样本

先跑一个很小的 smoke test，确认真实数据能接上。

当前脚本会把 `2×2` 个 z18 tile 合成一张 `512×512` 训练图。由于 GT 里存在较多漏标，先不要额外采集纯背景 tile，所以 `--negative` 保持为 `0`。

```bash
python -m building_seg.prepare_seg_dataset_from_tiles \
  --tiles data/tianditu/nansha_z18/tiles \
  --labels data/南沙区建筑物.gpkg \
  --out data/building_seg_tiles_512_debug \
  --label-field Function \
  --patch-tiles 2 \
  --max-positive 20 \
  --negative 0 \
  --val-ratio 0.2
```

成功后会生成：

```text
data/building_seg_tiles_512_debug/
  images/*.png
  masks/*.png
  splits/train.txt
  splits/val.txt
  metadata/dataset.json
  metadata/tiles.json
```

其中：

- `images/*.png` 是从天地图瓦片转来的 RGB 小图。
- `masks/*.png` 是由 GPKG polygon 栅格化得到的类别 mask。
- mask 中 `0` 表示 background，其他数值对应 `Function` 类别。
- 类别映射保存在 `metadata/dataset.json`。

注意：训练用的 `masks/*.png` 是类别 ID 图，像素值通常只有 `0~9` 这种小整数，所以用普通图片查看器打开时会显得几乎全黑。这是正常现象，不代表 mask 没有内容。

可以生成彩色预览图检查标注：

```bash
python -m building_seg.visualize_masks \
  --dataset data/building_seg_tiles_512_debug \
  --overlay \
  --limit 200
```

输出位置：

```text
data/building_seg_tiles_512_debug/mask_previews/color/
data/building_seg_tiles_512_debug/mask_previews/overlay/
data/building_seg_tiles_512_debug/mask_previews/legend.txt
```

如果想确认某个 mask 的实际类别值，也可以运行：

```bash
python -c "from PIL import Image; import numpy as np; import pathlib; root=pathlib.Path('data/building_seg_tiles_512_debug/masks'); p=sorted(root.glob('*.png'))[0]; a=np.array(Image.open(p)); print(p.name, np.unique(a, return_counts=True))"
```

确认 smoke test 成功后，再跑大一点的数据集：

```bash
python -m building_seg.prepare_seg_dataset_from_tiles \
  --tiles data/tianditu/nansha_z18/tiles \
  --labels data/南沙区建筑物.gpkg \
  --out data/building_seg_tiles_512_train \
  --label-field Function \
  --patch-tiles 2 \
  --max-positive 5000 \
  --negative 0 \
  --val-ratio 0.2
```

参数说明：

- `--max-positive`：最多导出多少个含建筑标注的 512×512 patch。
- `--negative`：额外导出多少个全背景 patch。当前真实 GT 有漏标风险，建议保持为 `0`，不要把无标注区域强行当成确定背景。
- `--patch-tiles`：每边拼接几个 tile。`2` 表示 `2×2` 个 256 tile 合成一张 512×512 训练图。
- `--label-field`：GPKG 里表示建筑功能的字段名。如果真实字段不是 `Function`，这里要改。

## 四、训练调试模型

先用默认的 `tiny_unet` 跑通训练链路：

```bash
python -m building_seg.train \
  --dataset data/building_seg_tiles_512_debug \
  --out data/building_seg_tiles_512_debug/checkpoints/tiny_unet_debug.pt \
  --model tiny_unet \
  --epochs 2 \
  --batch-size 4 \
  --base-channels 8
```

大一点的训练：

```bash
python -m building_seg.train \
  --dataset data/building_seg_tiles_512_train \
  --out data/building_seg_tiles_512_train/checkpoints/tiny_unet.pt \
  --model tiny_unet \
  --epochs 20 \
  --batch-size 8 \
  --base-channels 32
```

注意：`tiny_unet` 只是为了调通流程。真实效果要好，需要后续换更强的模型。

## 五、推理并输出 polygon

当前推理脚本使用 GeoTIFF 输入，因为它需要输出带地理坐标的 mask 和 polygon。

调试命令：

```bash
python -m building_seg.predict_to_polygon \
  --image data/nansha_img_w_z18.tif \
  --checkpoint data/building_seg_tiles_512_debug/checkpoints/tiny_unet_debug.pt \
  --out-mask data/building_seg_tiles_512_debug/predictions/debug_pred_mask.tif \
  --out-gpkg data/building_seg_tiles_512_debug/predictions/debug_pred_polygons.gpkg \
  --window 19558,6981,512,512 \
  --tile-size 512 \
  --overlap 0
```

输出：

```text
debug_pred_mask.tif
debug_pred_polygons.gpkg
debug_pred_polygons.classes.json
```

polygon GPKG 字段：

```text
class_id
Function
geometry
```

## 六、模型替换接口

模型注册位置：

```text
building_seg/models.py
```

要替换 `tiny_unet`，新增一个模型类：

```python
@register_model("your_model")
class YourModel(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        ...

    def forward(self, x):
        # x: Bx3xHxW，float tensor，范围 [0, 1]
        # 返回: Bxnum_classesxHxW logits
        return logits
```

训练时指定：

```bash
python -m building_seg.train --model your_model ...
```

checkpoint 会保存：

```text
model_name
model_kwargs
class_names
model_state
```

所以推理脚本可以自动按 checkpoint 加载对应模型。

## 七、迁移到真实环境时重点检查

1. `data/南沙区建筑物.gpkg` 是否能被 `geopandas.read_file()` 读取。
2. GPKG 是否有 `Function` 字段。
3. GPKG 是否有 CRS。如果没有 CRS，需要先补 CRS。
4. tiles 路径是否是 `tiles/18/{x}/{y}.jpg` 这种结构。
5. smoke test 是否能找到正样本：

```text
Positive > 0
```

如果 `Positive=0`，说明建筑物和瓦片空间没有对上，常见原因是：

- GPKG CRS 写错或缺失；
- GPKG 坐标不是南沙区域；
- tiles 不是同一个区域；
- 字段路径或文件名传错。

## 八、当前已验证结果

本机已用：

```text
data/tianditu/nansha_z18/tiles
data/南沙区建筑物.gpkg
```

跑通 smoke test：

```text
total_tiles_found = 92002
positive_samples = 10
negative_samples = 0
patch_tiles = 2
class_names = background, 仓储, 公共服务, 其他, 办公, 医疗, 商业, 居住, 工业, 教育
```

并生成了 512×512 调试样本：

```text
data/building_seg_tiles_512_debug/
```
