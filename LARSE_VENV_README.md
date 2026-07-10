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

## 另一台机器的路径配置

如果在另一台机器上运行，当前已知路径是：

```text
testaoi 项目：/home/f50059431/code/footprint/testaoi
已准备好的 512 数据：/home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all
LaRSE 项目：/home/f50059431/code/LaRSE
RemoteCLIP：/home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
```

先进入项目目录：

```bash
cd /home/f50059431/code/footprint/testaoi
conda activate <你的_larse_conda环境名>
```

注意：LaRSE 推理需要两个权重文件，不只需要 RemoteCLIP：

```bash
ls -lh /home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
ls -lh /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt
```

其中：

```text
RemoteCLIP-ViT-B-32.pt：视觉-语言 backbone 权重
checkpoint_LARSE.ckpt：LaRSE 分割模型 checkpoint
```

如果 `checkpoint_LARSE.ckpt` 不在这个目录，需要先放进去，或在下面命令里把 `--checkpoint` 改成它的真实路径。

一般不需要改 LaRSE 源码。当前 LaRSE 代码里的 `modules/models/lseg_vit.py` 会按 LaRSE 项目根目录自动找：

```text
<LaRSE>/checkpoints/RemoteCLIP-ViT-B-32.pt
```

所以只要 RemoteCLIP 文件已经在：

```text
/home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
```

就不用改 `lseg_vit.py`。如果你那份 LaRSE 代码仍然写死了旧路径，例如 `/home/user/code/int/my/LaRSE/...`，就在 LaRSE 目录检查：

```bash
grep -R "RemoteCLIP-ViT-B-32.pt\\|/home/user/code/int/my" -n /home/f50059431/code/LaRSE/modules /home/f50059431/code/LaRSE/*.sh
```

如果 grep 到硬编码旧路径，把它改成：

```text
/home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
```

更推荐的写法是在 `/home/f50059431/code/LaRSE/modules/models/lseg_vit.py` 顶部使用项目根目录拼路径：

```python
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REMOTECLIP_VIT_B32 = os.path.join(_PROJECT_ROOT, "checkpoints", "RemoteCLIP-ViT-B-32.pt")
```

这样换机器时不用再改绝对路径。

## 在真实 512 数据上测 LaRSE 效果

如果你的数据目录已经是：

```text
/home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all
```

并且里面有：

```text
images/*.png
masks/*.png
splits/train.txt
splits/val.txt
metadata/dataset.json
```

优先跑这个 GT 对齐可视化命令。它会对验证集 patch 推理，并把原图、GT、LaRSE 预测、误差图写成 HTML：

```bash
cd /home/f50059431/code/footprint/testaoi
conda activate <你的_larse_conda环境名>

python -m building_seg.predict_larse_debug_dataset \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --split val \
  --limit 100 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --device cuda
```

输出结果：

```text
/home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val/index.html
/home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val/metrics.json
```

打开 HTML：

```bash
xdg-open /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val/index.html
```

如果确认能跑，再把 `--limit 100` 改大，例如：

```bash
python -m building_seg.predict_larse_debug_dataset \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_1000 \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --split val \
  --limit 1000 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --device cuda
```

如果要对原始天地图 tiles 直接批量推理并导出 polygon GPKG，用这个命令：

```bash
python -m building_seg.predict_tiles_larse_to_polygon \
  --tiles /home/f50059431/code/footprint/testaoi/data/tianditu/nansha_z18/tiles \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --out-gpkg /home/f50059431/code/footprint/testaoi/data/larse_tiles_512_all/pred_larse_polygons_100.gpkg \
  --out-mask-dir /home/f50059431/code/footprint/testaoi/data/larse_tiles_512_all/masks_100 \
  --out-larse-mask-dir /home/f50059431/code/footprint/testaoi/data/larse_tiles_512_all/larse_raw_masks_100 \
  --patch-tiles 2 \
  --stride-tiles 2 \
  --limit 100 \
  --min-area-pixels 12 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --device cuda
```

这个命令输出的是预测 polygon，不直接和 GT 做 HTML 对比。建议先用上面的 `predict_larse_debug_dataset` 看效果，再跑这个批量导出。

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

### 12.1 `KeyError: 'buff1w'`

如果运行：

```bash
python -m building_seg.predict_larse_debug_dataset ...
```

报错：

```text
KeyError: 'buff1w'
```

含义是：LaRSE 已经开始加载，但当前 conda 环境里的 `torch-encoding` 数据集注册表不认识 `buff1w`。

最新的 `building_seg.predict_tiles_larse_to_polygon` 已经会在运行时自动做两件事：

```text
1. 从 /home/f50059431/code/LaRSE/buff1w.py 动态注册 buff1w 到 encoding.datasets
2. 在 /home/f50059431/code/LaRSE/datasets/BFF1WChallenge_for_LaRSE 下生成最小 dummy train/val 数据
```

这个 dummy 数据只用于通过 LaRSE 初始化，不参与我们的南沙数据测试。

所以在另一台机器上，先更新代码后重跑：

```bash
cd /home/f50059431/code/footprint/testaoi
git pull

conda activate <你的_larse_conda环境名>

python -m building_seg.predict_larse_debug_dataset \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --split val \
  --limit 100 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --device cuda
```

如果暂时不能更新代码，也可以手工注册 `buff1w`：

```bash
conda activate <你的_larse_conda环境名>
cd /home/f50059431/code/LaRSE

ENCODING_DIR=$(python - <<'PY'
import encoding
from pathlib import Path
print(Path(encoding.__file__).parent)
PY
)

cp buff1w.py "$ENCODING_DIR/datasets/buff1w.py"
```

然后编辑：

```bash
nano "$ENCODING_DIR/datasets/__init__.py"
```

加入：

```python
from .buff1w import BuFF1WChallengeDataset
```

并在 `datasets = {...}` 里加入：

```python
'buff1w': BuFF1WChallengeDataset,
```

如果继续报找不到 BUFF 数据目录，可以手工建一个最小目录：

```bash
mkdir -p /home/f50059431/code/LaRSE/datasets/BFF1WChallenge_for_LaRSE/images/training
mkdir -p /home/f50059431/code/LaRSE/datasets/BFF1WChallenge_for_LaRSE/images/validation
mkdir -p /home/f50059431/code/LaRSE/datasets/BFF1WChallenge_for_LaRSE/annotations/training
mkdir -p /home/f50059431/code/LaRSE/datasets/BFF1WChallenge_for_LaRSE/annotations/validation

python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("/home/f50059431/code/LaRSE/datasets/BFF1WChallenge_for_LaRSE")
for split_img, split_ann in [("training", "training"), ("validation", "validation")]:
    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(root / "images" / split_img / "dummy.jpg")
    Image.fromarray(np.full((16, 16), 11, dtype=np.uint8)).save(root / "annotations" / split_ann / "dummy.png")
PY
```

### 12.2 找不到 `/home/heda/.cache/clip/RemoteCLIP-ViT-B-32.pt`

如果报错：

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/home/heda/.cache/clip/RemoteCLIP-ViT-B-32.pt'
```

说明你那份 LaRSE 的 `modules/models/lseg_vit.py` 里还保留了作者机器上的硬编码路径。

最新适配脚本已经会在运行时自动把这个路径重定向到：

```text
/home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
```

同步新版代码后重跑即可：

```bash
cd /home/f50059431/code/footprint/testaoi
git pull

conda activate <你的_larse_conda环境名>

python -m building_seg.predict_larse_debug_dataset \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --split val \
  --limit 100 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --device cuda
```

正常时日志里会出现类似：

```text
Redirect RemoteCLIP checkpoint: /home/heda/.cache/clip/RemoteCLIP-ViT-B-32.pt -> /home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
```

如果想直接改 LaRSE 源码，也可以把 `/home/f50059431/code/LaRSE/modules/models/lseg_vit.py` 里的：

```python
torch.load(f"/home/heda/.cache/clip/RemoteCLIP-ViT-B-32.pt", map_location="cpu")
```

改成：

```python
torch.load("/home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt", map_location="cpu")
```

不过更推荐用适配脚本的运行时重定向，这样 LaRSE 原仓库少改一点。

### 12.3 torch-encoding 编译时报 Unsupported .version

如果安装 `torch-encoding` 时出现类似：

```text
ptxas /tmp/tmpxft_..._encoding_kernel.ptx, line 9 fatal:
Unsupported .version 8.5; current version is '8.2'
```

这通常不是 `torch-encoding` 源码本身的问题，而是 **PyTorch / CUDA / nvcc / ptxas 版本混用**。含义是：当前编译流程生成了 PTX 8.5，但实际调用到的 `ptxas` 只能识别到 PTX 8.2。

先检查当前环境：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

which nvcc || true
nvcc --version || true

which ptxas || true
ptxas --version || true
```

推荐组合是：

```text
Python 3.8.18
torch 1.9.1+cu111
torchvision 0.10.1+cu111
CUDA toolkit / nvcc 11.x
```

如果检查发现当前环境里是 `torch 2.x + cu12x`，不要在这个环境里装 `torch-encoding`。建议新建一个干净的 LaRSE conda 环境，或在当前环境里先卸掉新版 torch：

```bash
pip uninstall -y torch torchvision torchaudio torch-encoding

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

然后确认：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

应该看到类似：

```text
1.9.1+cu111
11.1
```

再安装 `torch-encoding`：

```bash
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/@331ecdd5306104614cb414b16fbcd9d1a8d40e1e
```

如果仍然报 `ptxas` 版本错误，说明系统 PATH 里调用到了不匹配的 CUDA 工具。临时指定 CUDA 11.x：

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

which nvcc
nvcc --version
which ptxas
ptxas --version
```

如果机器上没有 `/usr/local/cuda-11.1`，可以尝试已有的 CUDA 11.3 或 11.7。关键是不要让 `torch 1.9.1+cu111` 的扩展编译过程混到 CUDA 12.x 的工具链。

如果你必须使用 CUDA 12.x 工具链，那更推荐不要强行编译 LaRSE 这套老 `torch-encoding`，而是单独用已跑通的老环境，或者用 conda 重新建一个专门的 LaRSE 环境。

### 12.4 其他安装问题

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
