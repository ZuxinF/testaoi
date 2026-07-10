# LaRSE 远端调试命令

本文只放可以直接复制到远端机器运行的命令。

远端路径假设：

```text
项目目录：/home/f50059431/code/footprint/testaoi
LaRSE 目录：/home/f50059431/code/LaRSE
数据目录：/home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all
conda 环境：zx_larse
```

## 1. 进入环境

```bash
cd /home/f50059431/code/footprint/testaoi
conda activate zx_larse
```

确认两个 LaRSE 权重都存在：

```bash
ls -lh /home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
ls -lh /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt
```

## 2. 用旧 raw mask 检查新版类别映射

如果你之前已经跑出了：

```text
/home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val
```

可以先不用重跑 LaRSE，直接用已有的 `larse_raw_masks` 检查新版映射是否把前景正确映射到英文 `Function` 类别：

```bash
python -m building_seg.analyze_larse_eval \
  --eval-dir /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out-json /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val/diagnosis_v2.json
```

重点看输出：

```text
Remapped prediction classes with current code
```

如果这里出现了：

```text
Residential
Public service
Commercial
Industrial
Educational
```

说明新版映射已经生效。之前全 background 是因为旧映射只匹配中文类别名。

## 3. 重新生成 LaRSE + GT 对齐 HTML

用新版映射重新跑 100 张验证 patch：

```bash
python -m building_seg.predict_larse_debug_dataset \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2 \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --split val \
  --limit 100 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --device cuda
```

打开结果：

```bash
xdg-open /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2/index.html
```

如果服务器没有桌面，可以把这个目录打包下载：

```bash
cd /home/f50059431/code/footprint/testaoi/data
zip -r larse_eval_512_all_val_v2.zip larse_eval_512_all_val_v2
```

## 4. 诊断新版结果

```bash
python -m building_seg.analyze_larse_eval \
  --eval-dir /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2 \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out-json /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2/diagnosis.json
```

重点看：

```text
fg_acc > 0 samples
pred_fg > 0 samples
avg foreground_accuracy
Raw LaRSE classes, 1-12
Remapped prediction classes
GT classes
Diagnosis
```

判断：

```text
1. pred_fg > 0 仍然是 0：
   说明新版映射没有生效，或 metadata/dataset.json 类别名和预期仍然不一致。

2. pred_fg > 0，但 fg_acc > 0 仍然是 0：
   说明 LaRSE 有预测前景，但和 GT 没有重合。打开 HTML 看是 GT 漏标、类别错，还是位置不贴。

3. fg_acc > 0：
   说明 LaRSE 在当前数据上有一定迁移命中，可以继续扩大 limit 看整体趋势。
```

## 5. 扩大测试数量

如果 100 张能跑通，可以扩大到 1000 张：

```bash
python -m building_seg.predict_larse_debug_dataset \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2_1000 \
  --class-json /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all/metadata/dataset.json \
  --split val \
  --limit 1000 \
  --larse-dir /home/f50059431/code/LaRSE \
  --checkpoint /home/f50059431/code/LaRSE/checkpoints/checkpoint_LARSE.ckpt \
  --device cuda
```

然后诊断：

```bash
python -m building_seg.analyze_larse_eval \
  --eval-dir /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2_1000 \
  --dataset /home/f50059431/code/footprint/testaoi/data/building_seg_tiles_512_all \
  --out-json /home/f50059431/code/footprint/testaoi/data/larse_eval_512_all_val_v2_1000/diagnosis.json
```

## 6. 导出 polygon GPKG

如果 HTML 看起来还可以，再跑 polygon 导出：

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

输出：

```text
/home/f50059431/code/footprint/testaoi/data/larse_tiles_512_all/pred_larse_polygons_100.gpkg
```

## 7. 常见异常

如果看到：

```text
Redirect RemoteCLIP checkpoint: /home/heda/.cache/clip/RemoteCLIP-ViT-B-32.pt -> /home/f50059431/code/LaRSE/checkpoints/RemoteCLIP-ViT-B-32.pt
```

这是正常的，说明脚本把 LaRSE 源码里的作者机器硬编码路径重定向到了当前机器的 RemoteCLIP。

如果再次出现：

```text
KeyError: 'buff1w'
```

说明你没有同步到新版脚本，或者运行的不是当前项目目录。先确认：

```bash
pwd
python - <<'PY'
import building_seg.predict_tiles_larse_to_polygon as m
print(m.__file__)
PY
```

应该指向：

```text
/home/f50059431/code/footprint/testaoi/building_seg/predict_tiles_larse_to_polygon.py
```
