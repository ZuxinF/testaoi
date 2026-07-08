# LaRSE 迁移推理 Conda 环境说明

这份说明用于在 `footprint/testaoi` 项目里接入 `../../LaRSE`，用 LaRSE checkpoint 测一测当前南沙 tiles + GPKG 数据上的迁移效果。

你现在已经有一个 **conda 的 Python 3.8.18 环境**，推荐直接用这个环境，不再新建 `.venv`。

本文命令默认从当前项目目录出发：

```bash
cd /home/user/code/int/my/footprint/testaoi
```

LaRSE 项目默认位置：

```text
/home/user/code/int/my/LaRSE
```

脚本会自动查找：

```text
../../LaRSE
../../LaRSE/checkpoints/checkpoint_LARSE.ckpt
../../LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
```

## 一、激活 conda 环境

把下面的 `<env_name>` 换成你的 Python 3.8.18 conda 环境名：

```bash
conda activate <env_name>
python --version
```

确认输出类似：

```text
Python 3.8.18
```

## 二、安装 PyTorch

推荐先用和已验证 LaRSE 环境接近的 CUDA 11.1 wheel：

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

如果你的机器只能 CPU 冒烟测试：

```bash
pip install torch==1.9.1 torchvision==0.10.1
```

说明：即使主机是 CUDA 12.x 驱动，也通常可以运行 `cu111` wheel，因为 PyTorch wheel 自带 CUDA runtime，主要要求是显卡驱动足够新。

## 三、安装 LaRSE 依赖

```bash
pip install \
  numpy==1.21.2 \
  pandas==1.3.4 \
  Pillow==8.4.0 \
  pytorch-lightning==1.4.9 \
  torchmetrics==0.6.0 \
  test-tube==0.7.5 \
  timm==0.4.12 \
  open-clip-torch \
  ftfy \
  regex \
  tqdm \
  scipy \
  matplotlib \
  opencv-python \
  torchinfo \
  thop \
  fvcore
```

安装 OpenAI CLIP：

```bash
pip install git+https://github.com/openai/CLIP.git@04f4dc2ca1ed0acc9893bd1a3b526a7e02c4bb10
```

安装 `torch-encoding`：

```bash
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/@331ecdd5306104614cb414b16fbcd9d1a8d40e1e
```

安装 GPKG / polygon 输出需要的地理包：

```bash
pip install \
  geopandas==0.10.2 \
  shapely==1.8.5.post1 \
  pyproj==3.4.1 \
  fiona==1.8.22 \
  rasterio==1.3.8
```

## 四、注册 LaRSE 的 BUFF 数据集

LaRSE 代码会加载 `encoding.datasets`，需要把 LaRSE 的 `buff1w.py` 放进当前 conda 环境的 `encoding/datasets/` 目录。

先进入 LaRSE 项目：

```bash
cd /home/user/code/int/my/LaRSE
```

查看 `encoding` 安装路径：

```bash
python - <<'PY'
import encoding
from pathlib import Path
print(Path(encoding.__file__).parent)
PY
```

复制数据集文件：

```bash
ENCODING_DIR=$(python - <<'PY'
import encoding
from pathlib import Path
print(Path(encoding.__file__).parent)
PY
)

cp buff1w.py "$ENCODING_DIR/datasets/buff1w.py"
```

编辑：

```bash
nano "$ENCODING_DIR/datasets/__init__.py"
```

加入 import：

```python
from .buff1w import BuFF1WChallengeDataset
```

在数据集字典里加入：

```python
'buff1w': BuFF1WChallengeDataset,
```

## 五、确认 checkpoint

确认下面两个文件存在：

```bash
ls -lh /home/user/code/int/my/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
ls -lh /home/user/code/int/my/LaRSE/checkpoints/checkpoint_LARSE.ckpt
```

如果这两个文件都在，从 `footprint/testaoi` 运行迁移脚本时通常不需要手动传 `--larse-dir` 和 `--checkpoint`。

## 六、检查环境

在 conda 环境已激活时运行：

```bash
python - <<'PY'
mods = [
    "torch", "torchvision", "pytorch_lightning", "timm",
    "open_clip", "clip", "encoding", "geopandas", "rasterio",
    "shapely", "pyproj", "cv2", "torchinfo", "thop", "fvcore"
]
for m in mods:
    try:
        __import__(m)
        print("OK  ", m)
    except Exception as e:
        print("MISS", m, e)

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

## 七、准备当前数据的类别映射

LaRSE 输出的是 BUFF 12 类，脚本会把它映射成当前 GPKG 的 `Function` 类别。推荐先用当前真实数据生成一次 `metadata/dataset.json`，作为类别顺序来源。

如果还没有这个文件，先在 `footprint/testaoi` 目录运行：

```bash
cd /home/user/code/int/my/footprint/testaoi

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

确认文件存在：

```bash
ls data/building_seg_tiles_512_debug/metadata/dataset.json
```

## 八、LaRSE 单 patch 冒烟测试

```bash
cd /home/user/code/int/my/footprint/testaoi

python -m building_seg.predict_tiles_larse_to_polygon \
  --tiles data/tianditu/nansha_z18/tiles \
  --class-json data/building_seg_tiles_512_debug/metadata/dataset.json \
  --out-gpkg data/larse_tiles_debug/pred_larse_polygons.gpkg \
  --out-mask-dir data/larse_tiles_debug/masks \
  --out-larse-mask-dir data/larse_tiles_debug/larse_raw_masks \
  --patch-tiles 2 \
  --stride-tiles 2 \
  --limit 1
```

成功时会看到类似：

```text
Predicted patches: 1
Wrote polygons: data/larse_tiles_debug/pred_larse_polygons.gpkg
```

## 九、用 debug GT 对齐可视化

如果已经有：

```text
data/building_seg_tiles_512_debug/images/*.png
data/building_seg_tiles_512_debug/masks/*.png
```

可以直接让 LaRSE 对这批 debug patch 推理，并和 GT 叠加成 HTML：

```bash
python -m building_seg.predict_larse_debug_dataset \
  --dataset data/building_seg_tiles_512_debug \
  --out data/larse_debug_eval \
  --split val \
  --limit 20
```

输出：

```text
data/larse_debug_eval/index.html
data/larse_debug_eval/gt_overlay/
data/larse_debug_eval/pred_overlay/
data/larse_debug_eval/error/
data/larse_debug_eval/metrics.json
```

打开：

```bash
xdg-open data/larse_debug_eval/index.html
```

图中含义：

```text
GT 叠加：绿色为真实标注区域
LaRSE 预测叠加：红色为预测前景区域
error 图：绿色=前景预测正确，红色=GT 前景错分，橙色=背景误检
```

这个命令最适合先验证 LaRSE 在我们当前数据上的迁移观感，因为它保证 LaRSE 输入和 GT mask 是同一批 debug patch。

如果想看训练集和验证集一起的前 50 张，可以不传 `--split`：

```bash
python -m building_seg.predict_larse_debug_dataset \
  --dataset data/building_seg_tiles_512_debug \
  --out data/larse_debug_eval_all \
  --limit 50
```

## 十、LaRSE 批量测试当前数据

先跑 100 个 512×512 patch 看输出质量：

```bash
python -m building_seg.predict_tiles_larse_to_polygon \
  --tiles data/tianditu/nansha_z18/tiles \
  --class-json data/building_seg_tiles_512_debug/metadata/dataset.json \
  --out-gpkg data/larse_tiles_debug/pred_larse_polygons_100.gpkg \
  --out-mask-dir data/larse_tiles_debug/masks_100 \
  --out-larse-mask-dir data/larse_tiles_debug/larse_raw_masks_100 \
  --patch-tiles 2 \
  --stride-tiles 2 \
  --limit 100 \
  --min-area-pixels 12
```

如果结果看起来有意义，再扩大：

```bash
python -m building_seg.predict_tiles_larse_to_polygon \
  --tiles data/tianditu/nansha_z18/tiles \
  --class-json data/building_seg_tiles_512_debug/metadata/dataset.json \
  --out-gpkg data/larse_tiles_debug/pred_larse_polygons_1000.gpkg \
  --out-mask-dir data/larse_tiles_debug/masks_1000 \
  --out-larse-mask-dir data/larse_tiles_debug/larse_raw_masks_1000 \
  --patch-tiles 2 \
  --stride-tiles 2 \
  --limit 1000 \
  --min-area-pixels 12
```

说明：

- `--out-larse-mask-dir` 保存 LaRSE 原始 1-12 类 mask。
- `--out-mask-dir` 保存映射到当前 `Function` 类之后的 ID mask。
- `--out-gpkg` 保存映射后的 polygon 结果。
- `--limit` 去掉后会尽量遍历全部 tiles，但耗时和输出量会明显增加。

## 十一、用 GIS 叠加检查 LaRSE 输出

LaRSE 当前输出是 polygon GPKG 和 mask。要和真实 GT 叠加看，可以先用 GIS 软件打开：

```text
data/南沙区建筑物.gpkg
data/larse_tiles_debug/pred_larse_polygons_100.gpkg
```

建议检查：

```text
1. 预测 polygon 是否落在影像正确位置
2. 是否大量漏检建筑
3. 是否把背景误检成建筑
4. Function 类别是否有明显偏置
```

注意：LaRSE 这个脚本是跨城市、跨影像源的直接迁移基线，不是重新训练。它能帮你快速判断 LaRSE checkpoint 在当前数据上的迁移可用性，但不能代表最终可训练上限。

## 十二、常见问题

如果 `torch-encoding` 安装失败，优先检查：

```bash
python --version
python -c "import torch; print(torch.__version__)"
```

建议组合是：

```text
Python 3.8.18
torch 1.9.1
torchvision 0.10.1
```

如果 `python -m building_seg.predict_tiles_larse_to_polygon` 找不到 `building_seg`，说明当前目录不在 `footprint/testaoi`。请先：

```bash
cd /home/user/code/int/my/footprint/testaoi
```

如果脚本找不到 LaRSE，可以显式指定：

```bash
python -m building_seg.predict_tiles_larse_to_polygon \
  --larse-dir /home/user/code/int/my/LaRSE \
  --checkpoint /home/user/code/int/my/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --tiles data/tianditu/nansha_z18/tiles \
  --class-json data/building_seg_tiles_512_debug/metadata/dataset.json \
  --out-gpkg data/larse_tiles_debug/pred_larse_polygons.gpkg \
  --patch-tiles 2 \
  --stride-tiles 2 \
  --limit 1
```
