# 用 `building_seg_tiles_512_all` 训练 LaRSE 的方案

目标数据：

```text
/home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all
```

这个目录通常包含：

```text
images/*.png 或 images/*.jpg
masks/*.png
splits/train.txt
splits/val.txt
metadata/dataset.json
```

LaRSE 原仓库不能直接吃这个目录训练。本项目已在 `footprint/testaoi` 侧补了适配脚本，核心做 3 件事：

```text
1. 注册一个 LaRSE/encoding 可识别的数据集
2. 让 LaRSE 使用我们的 Function 类别文本，而不是 buff4k_objectInfo.txt
3. 补一个 PyTorch Lightning 训练入口
```

## 一、为什么不能直接训练

LaRSE 当前代码有几个固定假设：

```text
1. dataset 名称要能被 encoding.datasets.get_dataset 找到
2. LSegModule 会在初始化时创建 trainset/valset
3. LSegModule 里写死了 labels = self.get_labels('buff4k')
4. 原始 checkpoint 是 BUFF 12 类，不是当前 Function 类
```

所以如果直接把 `building_seg_tiles_512_all` 塞给 LaRSE，大概率会出现：

```text
KeyError: 数据集未注册
类别数和文本 label 数不一致
checkpoint 严格加载失败
训练时输出通道和 mask 类别不一致
```

## 二、推荐训练方式

推荐先做“适配训练”，不要直接改坏 LaRSE 原项目：

```text
footprint/testaoi 侧新增训练适配脚本
运行时动态注册 building_seg_tiles_512_all
运行时动态替换 LaRSE 的文本类别
从 checkpoint_LARSE.ckpt 非严格加载可兼容权重
保存新的南沙 Function 版 checkpoint
```

这样可以保留原 LaRSE 复现实验能力。

## 三、类别文件检查

先看你的类别顺序：

```bash
cd /home/f50059431/code/footprint/testaoi

python - <<'PY'
import json
from pathlib import Path
p = Path("data/building_seg_tiles_512_all/metadata/dataset.json")
data = json.loads(p.read_text())
for i, name in enumerate(data["class_names"]):
    print(i, name)
PY
```

mask 里的像素值必须和这个顺序一致：

```text
0 = background
1..N = Function 类别
```

如果 mask 不是这个口径，需要先修数据，不能直接训练。

## 四、建议训练配置

A6000 48GB 上，先用保守配置：

```text
backbone: clip_vitb32_384
img size: 512 输入，但 LaRSE 内部 crop_size 默认 1024，需要适配为 512 或接受 resize
batch_size: 2 或 4 起步
lr: 1e-4 到 4e-4
epochs: 30 先冒烟，正式 100+
init checkpoint: /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt
RemoteCLIP: /home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
```

注意：LaRSE 比 YOLO 更吃显存，先别一上来大 batch。

## 五、已补好的代码

训练入口已经新增：

```text
building_seg/train_larse_on_prepared_dataset.py
```

这个脚本会做：

```text
1. 读取 metadata/dataset.json 得到 class_names
2. 定义 PreparedBuildingSegDataset
3. 动态注册 encoding.datasets.datasets["buff1w"] = PreparedBuildingSegDataset
4. monkeypatch LSegModule.get_labels，让它返回 class_names
5. 创建 LSegModule
6. 从 checkpoint_LARSE.ckpt 过滤形状不匹配的 key 后加载
7. pl.Trainer.fit(model)
```

同时，验证脚本也已经支持 fine-tuned checkpoint：

```text
building_seg/predict_larse_debug_dataset.py
```

它新增了 `--label-space auto|larse|target`：

```text
auto：如果 checkpoint 同目录有 class_names.json 且与当前数据一致，就按 target 类别解释输出
larse：按原始 BUFF 12 类输出，再 remap 到当前 Function
target：认为模型已经直接输出当前 Function 类别
```

关键点：不能继续用 `buff4k_objectInfo.txt` 的 12 类文本，否则训练目标和输出文本类别不一致。

## 六、Smoke Test 命令

先只用极少数据跑 1 个 epoch，确认 forward/backward/checkpoint 都通：

```bash
cd /home/f50059431/code/footprint/testaoi
conda activate zx_larse

python -m building_seg.train_larse_on_prepared_dataset \
  --dataset-dir /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --larse-dir /home/f50059431/code/LaRSE \
  --init-checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --out-dir /home/f50059431/code/footprint/testaoi/data/larse_train_building_seg_512_smoke \
  --backbone clip_vitb32_384 \
  --batch-size 1 \
  --epochs 1 \
  --lr 0.0001 \
  --num-workers 2 \
  --max-train-samples 16 \
  --max-val-samples 8 \
  --freeze-clip \
  --freeze-backbone \
  --device cuda
```

如果这个都爆显存，把 `--device cuda` 改成 CPU 只验证链路，或者继续保持 `--freeze-clip --freeze-backbone`，不要先训全量。

## 七、正式训练命令

Smoke test 成功后，再跑较完整训练：

```bash
cd /home/f50059431/code/footprint/testaoi
conda activate zx_larse

python -m building_seg.train_larse_on_prepared_dataset \
  --dataset-dir /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --larse-dir /home/f50059431/code/LaRSE \
  --init-checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --out-dir /home/f50059431/code/footprint/testaoi/data/larse_train_building_seg_512 \
  --backbone clip_vitb32_384 \
  --batch-size 2 \
  --epochs 30 \
  --lr 0.0001 \
  --num-workers 4 \
  --freeze-clip \
  --device cuda
```

建议先冻结 RemoteCLIP 文本/图像模型，只训练 LaRSE 的视觉 backbone/scratch/head。如果还爆显存或不稳定，可以再加：

```bash
--freeze-backbone --batch-size 1 --num-workers 2
```

训练时看：

```text
train_loss 是否下降
val_iou 是否不是 NaN
checkpoint 是否能保存
```

输出目录至少会包含：

```text
best.ckpt
last.ckpt
class_names.json
checkpoint_load_report.json
logs/
```

## 八、5 epoch 快速验证这条线是否值得继续

如果全量训练一个 epoch 需要约 30 分钟，建议先跑一个小规模快速验证，不要一开始就烧完整 30 epoch。

推荐配置：

```text
train samples: 5000
val samples: 1000
epochs: 5
batch_size: 16
num_workers: 16
freeze: 只冻结 RemoteCLIP，不冻结 LaRSE 图像 backbone
```

训练命令：

```bash
cd /home/f50059431/code/footprint/testaoi
conda activate zx_larse

python -m building_seg.train_larse_on_prepared_dataset \
  --dataset-dir /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --larse-dir /home/f50059431/code/LaRSE \
  --init-checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --out-dir /home/f50059431/code/footprint/testaoi/data/larse_train_building_seg_512_fastcheck \
  --backbone clip_vitb32_384 \
  --batch-size 16 \
  --epochs 5 \
  --lr 0.0001 \
  --num-workers 16 \
  --max-train-samples 5000 \
  --max-val-samples 1000 \
  --freeze-clip \
  --device cuda
```

注意这里不要加 `--freeze-backbone`。这一步要让 LaRSE 的图像侧适应当前天地图影像；只冻结 RemoteCLIP，保留文本语义先验。

### 8.1 生成 fine-tuned LaRSE 可视化

训练完后，用 `best.ckpt` 在同一个 val split 上生成 100 张 HTML：

```bash
python -m building_seg.predict_larse_debug_dataset \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out /home/f50059431/code/footprint/testaoi/data/larse_finetuned_fastcheck_eval_val \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --split val \
  --limit 100 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/footprint/testaoi/data/larse_train_building_seg_512_fastcheck/best.ckpt \
  --label-space auto \
  --device cuda
```

打开：

```bash
xdg-open /home/f50059431/code/footprint/testaoi/data/larse_finetuned_fastcheck_eval_val/index.html
```

如果没有桌面，打包下载：

```bash
cd /home/f50059431/code/footprint/testaoi/data
zip -r larse_finetuned_fastcheck_eval_val.zip larse_finetuned_fastcheck_eval_val
```

### 8.2 生成诊断 JSON

```bash
python -m building_seg.analyze_larse_eval \
  --eval-dir /home/f50059431/code/footprint/testaoi/data/larse_finetuned_fastcheck_eval_val \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out-json /home/f50059431/code/footprint/testaoi/data/larse_finetuned_fastcheck_eval_val/diagnosis.json
```

重点看：

```text
fg_acc > 0 samples
pred_fg > 0 samples
avg foreground_accuracy
Remapped prediction classes
GT classes
Diagnosis
```

### 8.3 和未 fine-tune 的 LaRSE 基线对比

如果之前已经有未训练的 LaRSE 结果：

```text
/home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2
```

用下面命令打印两个结果的核心指标：

```bash
python - <<'PY'
import json
from pathlib import Path

items = {
    "baseline": Path("/home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2/metrics.json"),
    "finetuned_fastcheck": Path("/home/f50059431/code/footprint/testaoi/data/larse_finetuned_fastcheck_eval_val/metrics.json"),
}

for name, path in items.items():
    rows = json.loads(path.read_text())
    n = len(rows)
    fg_pos = sum(r["foreground_accuracy"] > 0 for r in rows)
    pred_pos = sum(r["pred_foreground_pixels"] > 0 for r in rows)
    avg_fg = sum(r["foreground_accuracy"] for r in rows) / max(n, 1)
    avg_pred = sum(r["pred_foreground_pixels"] for r in rows) / max(n, 1)
    avg_gt = sum(r["gt_foreground_pixels"] for r in rows) / max(n, 1)
    print(name)
    print("  samples:", n)
    print("  fg_acc > 0:", fg_pos)
    print("  pred_fg > 0:", pred_pos)
    print("  avg foreground_accuracy:", round(avg_fg, 6))
    print("  avg pred_foreground_pixels:", round(avg_pred, 2))
    print("  avg gt_foreground_pixels:", round(avg_gt, 2))
PY
```

判断是否值得继续：

```text
值得继续：
- avg foreground_accuracy 明显高于 baseline
- fg_acc > 0 的样本数明显增加
- HTML 里红色预测区域更贴近 GT
- pred_fg 不再极端偏少或极端泛滥

不太值得继续：
- avg foreground_accuracy 没提升
- HTML 里预测仍然大面积错位
- pred_fg 激增但多数是背景误检
- 类别分布明显塌到一两个类别
```

如果 5 epoch fastcheck 有提升，再跑全量 30 epoch 或 100 epoch；如果没有提升，优先继续优化 YOLO/Mask2Former，不要在 LaRSE 上消耗太多时间。

## 九、训练后怎么测

训练输出 checkpoint 后，用现有 debug 可视化脚本测：

```bash
python -m building_seg.predict_larse_debug_dataset \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out /home/f50059431/code/footprint/testaoi/data/larse_finetuned_eval_val \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --split val \
  --limit 100 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/footprint/testaoi/data/larse_train_building_seg_512/best.ckpt \
  --label-space auto \
  --device cuda
```

然后诊断：

```bash
python -m building_seg.analyze_larse_eval \
  --eval-dir /home/f50059431/code/footprint/testaoi/data/larse_finetuned_eval_val \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out-json /home/f50059431/code/footprint/testaoi/data/larse_finetuned_eval_val/diagnosis.json
```

## 十、常见报错

### 10.1 `module 'distutils' has no attribute 'version'`

如果训练时报：

```text
AttributeError: module 'distutils' has no attribute 'version'
```

这是老版本 PyTorch / PyTorch Lightning 和新版 Python packaging 组合里常见的问题。新版训练脚本已经显式加了：

```python
import distutils.version
```

所以先同步最新代码后重跑：

```bash
cd /home/f50059431/code/footprint/testaoi
git pull
conda activate zx_larse
```

如果仍然报错，在 `zx_larse` 环境里把 `setuptools` 固定到老版本：

```bash
pip install "setuptools==59.5.0"
```

然后确认：

```bash
python - <<'PY'
import distutils.version
print(distutils.version.LooseVersion("1.0"))
PY
```

能输出 `1.0` 就可以重新跑训练命令。

## 十一、现实预期

LaRSE 适合做视觉-语言语义分割基线，但对当前任务不一定比 YOLO-seg 更稳。

当前数据目标是：

```text
遥感影像 -> 建筑 polygon/mask + Function 类别
```

YOLO-seg 更贴近“实例 polygon”输出；LaRSE 更像“语义 mask + 文本类别推理”。所以推荐对比路线是：

```text
YOLO-seg：主线，输出 polygon 更自然
LaRSE fine-tune：语义/类别迁移基线
SegFormer/Mask2Former：后续更强语义分割备选
```

如果 LaRSE fine-tune 后只提升类别但轮廓不如 YOLO，可以考虑把 LaRSE 当类别先验或弱标签参考，而不是最终轮廓模型。

## 十二、给 Minimax2.7 的提示词

如果你在远端机器上使用 Minimax2.7 辅助改代码，可以直接复制下面这段提示词。

```text
你是一个熟悉 PyTorch、PyTorch Lightning、遥感语义分割和 LaRSE/LSeg/CLIP 代码结构的工程师。请你在当前项目中帮我把 LaRSE 接到我们的 prepared dataset 上进行 fine-tune 训练。

当前机器路径如下：

1. footprint/testaoi 项目路径：
   /home/f50059431/code/footprint/testaoi

2. LaRSE 项目路径：
   /home/f50059431/code/LaRSE

3. 我们已经准备好的训练数据：
   /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all

4. 数据目录结构：
   images/*.png 或 images/*.jpg
   masks/*.png
   splits/train.txt
   splits/val.txt
   metadata/dataset.json

5. LaRSE checkpoint：
   /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt

6. RemoteCLIP：
   /home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt

7. 当前 conda 环境：
   zx_larse

我的目标：

用 /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all 训练/微调 LaRSE，让模型学习我们的建筑 Function 类别分割。训练后要能在同一批 prepared dataset 的 val split 上生成可视化 HTML，并能用现有脚本 analyze_larse_eval 做诊断。

请你不要只解释原理，请直接完成代码修改和新增脚本。优先在 /home/f50059431/code/footprint/testaoi/building_seg 里新增适配代码，尽量少改 /home/f50059431/code/LaRSE 原仓库源码。如果必须改 LaRSE 源码，请明确列出改了哪些文件和原因。

必须解决的问题：

1. LaRSE 原代码依赖 encoding.datasets.get_dataset，但我们的数据不是原始 BUFF 数据。
   请实现一个 dataset 适配器，使 LaRSE 能读取：
   /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/images
   /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/masks
   /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/splits/train.txt
   /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/splits/val.txt

2. mask 里的类别 ID 是 0-based：
   0 = background
   1..N = Function 类别
   训练时不要再做 BUFF 那种 label - 1 转换。

3. 类别名必须从：
   /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json
   读取 class_names。
   不要继续使用 LaRSE 的 label_files/buff4k_objectInfo.txt 作为训练文本类别。

4. LaRSE 原始 checkpoint 是 BUFF 类别，不一定和我们的 Function 类别数一致。
   请实现非严格/过滤式加载 checkpoint：
   - 能加载形状匹配的 backbone / encoder / decoder 权重
   - 跳过类别数相关、shape 不一致的参数
   - 打印加载了多少 key、跳过了多少 key

5. 训练入口请新增：
   /home/f50059431/code/footprint/testaoi/building_seg/train_larse_on_prepared_dataset.py

6. 训练命令目标如下：

   cd /home/f50059431/code/footprint/testaoi
   conda activate zx_larse

   python -m building_seg.train_larse_on_prepared_dataset \
     --dataset-dir /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
     --larse-dir /home/f50059431/code/LaRSE \
     --init-checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
     --out-dir /home/f50059431/code/footprint/testaoi/data/larse_train_building_seg_512 \
     --backbone clip_vitb32_384 \
     --batch-size 2 \
     --epochs 30 \
     --lr 0.0001 \
     --device cuda

7. 训练输出目录至少包含：
   best.ckpt
   last.ckpt
   train_log.json 或 metrics.csv
   class_names.json

8. 训练后要能用已有脚本验证：

   python -m building_seg.predict_larse_debug_dataset \
     --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
     --out /home/f50059431/code/footprint/testaoi/data/larse_finetuned_eval_val \
     --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
     --split val \
     --limit 100 \
     --larse-dir /home/f50059431/code/LaRSE \
     --checkpoint /home/f50059431/code/footprint/testaoi/data/larse_train_building_seg_512/best.ckpt \
     --device cuda

   然后：

   python -m building_seg.analyze_larse_eval \
     --eval-dir /home/f50059431/code/footprint/testaoi/data/larse_finetuned_eval_val \
     --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
     --out-json /home/f50059431/code/footprint/testaoi/data/larse_finetuned_eval_val/diagnosis.json

实现要求：

1. 先阅读这些文件再改：
   /home/f50059431/code/LaRSE/modules/lseg_module.py
   /home/f50059431/code/LaRSE/modules/lsegmentation_module.py
   /home/f50059431/code/footprint/testaoi/building_seg/predict_tiles_larse_to_polygon.py
   /home/f50059431/code/footprint/testaoi/building_seg/predict_larse_debug_dataset.py
   /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json

2. 保留现有 predict_larse_debug_dataset.py 和 analyze_larse_eval.py 的使用方式。

3. 如果需要修改 predict_larse_debug_dataset.py 以支持 fine-tuned checkpoint 的新类别文本，也可以改，但要保证旧 checkpoint 仍然能跑。

4. 不要删除已有 YOLO、LaRSE debug、HTML 打包相关脚本。

5. 所有新增命令和注意事项写入：
   /home/f50059431/code/footprint/testaoi/LARSE_TRAIN_ON_BUILDING_SEG.md

6. 先做一个小规模 smoke test：
   - batch_size=1
   - epochs=1
   - limit 或 subset 可以只用少量训练样本
   确认能 forward、backward、保存 checkpoint。

7. 如果遇到显存问题，请提供降显存参数：
   --batch-size 1
   --num-workers 2
   冻结 CLIP/backbone 或只训练 scratch/head 的选项。

8. 完成后请告诉我：
   - 新增/修改了哪些文件
   - 训练命令
   - smoke test 命令
   - 验证 HTML 命令
   - 可能的风险，例如类别文本是否对齐、checkpoint 是否部分加载、是否冻结 backbone

请直接开始实现。
```
