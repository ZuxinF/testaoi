# LaRSE 迁移推理 venv 环境说明

这份说明用于从 `footprint/testaoi` 项目接入 `../../LaRSE`，运行：

```bash
python -m building_seg.predict_tiles_larse_to_polygon
```

你的主机 Python 是 `3.10.15`。不建议直接用它装 LaRSE，因为 LaRSE 依赖 `torch-encoding`、老版 PyTorch 和老版 Lightning，和新 Python 版本容易冲突。

推荐新建一个单独的 LaRSE venv：

```text
../../LaRSE/.venv-larse-py38
```

推荐 Python 版本：

```text
Python 3.8.18
```

Python 3.8 比 3.7 更容易在新机器上安装，同时仍然能兼容 `torch==1.9.1`、`torchvision==0.10.1` 和 `torch-encoding`。

## 一、安装 Python 3.8.18

如果机器上已经有 `python3.8`，可以跳过本节。

推荐用 `pyenv` 安装旧 Python，不影响系统 Python 3.10.15。

```bash
curl https://pyenv.run | bash
```

按终端提示把下面内容加入 `~/.bashrc`，如果已经有就不要重复加：

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

重新加载 shell：

```bash
exec "$SHELL"
```

安装 Python 3.8.18：

```bash
pyenv install 3.8.18
```

检查：

```bash
pyenv versions
```

## 二、在 LaRSE 目录创建新的 venv

从 `footprint/testaoi` 目录出发：

```bash
cd ../../LaRSE
pyenv shell 3.8.18
python --version
```

确认输出是：

```text
Python 3.8.18
```

创建新的 venv，目录名不要和 `testaoi/.venv` 混用：

```bash
python -m venv .venv-larse-py38
source .venv-larse-py38/bin/activate
python -m pip install -U pip setuptools wheel
```

## 三、安装 PyTorch

GPU 版本，推荐先用和当前已验证 LaRSE 环境接近的 CUDA 11.1 wheel：

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

如果只想 CPU 冒烟测试，可以用：

```bash
pip install torch==1.9.1 torchvision==0.10.1
```

说明：如果主机是 CUDA 12.x 驱动，也通常可以运行 `cu111` wheel，因为 PyTorch wheel 自带 CUDA runtime，主要要求是显卡驱动足够新。

## 四、安装 LaRSE 依赖

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

## 五、注册 LaRSE 的 BUFF 数据集

LaRSE 代码需要把 `buff1w.py` 放进 `encoding.datasets`。

先查看 `encoding` 安装路径：

```bash
python - <<'PY'
import encoding
from pathlib import Path
print(Path(encoding.__file__).parent)
PY
```

假设输出为：

```text
/path/to/LaRSE/.venv-larse-py38/lib/python3.8/site-packages/encoding
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

## 六、准备 LaRSE checkpoint

确认下面两个文件存在：

```text
../../LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
../../LaRSE/checkpoints/checkpoint_LARSE.ckpt
```

从 `footprint/testaoi` 运行迁移脚本时，脚本会自动寻找：

```text
../../LaRSE
../../LaRSE/checkpoints/checkpoint_LARSE.ckpt
```

所以通常不需要手动传 `--larse-dir` 和 `--checkpoint`。

## 七、检查环境

在 `../../LaRSE` 目录、且 venv 已激活时运行：

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

## 八、从 testaoi 跑 LaRSE 迁移推理

激活 LaRSE venv：

```bash
cd ../../LaRSE
source .venv-larse-py38/bin/activate
```

回到 `testaoi` 项目：

```bash
cd ../footprint/testaoi
```

跑 1 个 patch 冒烟测试：

```bash
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

## 九、常见问题

如果 `pyenv install 3.8.18` 编译失败，通常是系统缺少 Python 编译依赖。Ubuntu/Debian 可先安装：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential zlib1g-dev libssl-dev libbz2-dev libreadline-dev \
  libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev
```

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
cd /path/to/footprint/testaoi
```

