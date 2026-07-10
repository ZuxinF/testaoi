# YOLO 预测结果 HTML 打包命令

本文用于把已有 YOLO 预测叠加图整理成一个可下载查看的 HTML 文件夹。

远端路径假设：

```text
项目目录：/home/f50059431/code/footprint/testaoi
YOLO 数据集：/home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all
已有预测叠加图：/home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/prediction_overlays/val
```

## 1. 进入环境

```bash
cd /home/f50059431/code/footprint/testaoi
```

如果你在 YOLO 的 venv/conda 环境里，就先激活它；这个 HTML 打包脚本只需要 `Pillow` 和 `numpy`，不需要重新加载 YOLO 模型。

## 2. 生成 HTML 对比报告

```bash
python -m building_seg.package_yolo_prediction_html \
  --dataset /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all \
  --overlay-dir /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/prediction_overlays/val \
  --out /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/yolo_val_html_report \
  --split val \
  --title "YOLO26 Segmentation 验证集 GT 对比"
```

输出目录：

```text
/home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/yolo_val_html_report
```

里面包含：

```text
index.html
summary.json
image_*.jpg
gt_overlay_*.jpg
gt_mask_*.png
prediction_overlay_*.jpg
```

注意：新版脚本会把 `index.html` 和所有图片放在同一个文件夹里，不再使用子目录。这样下载后不容易出现 HTML 图片路径断掉的问题。

HTML 每个样本展示：

```text
原图
GT 叠加
YOLO 预测 + GT
GT mask 预览
```

## 3. 打开 HTML

如果机器有桌面：

```bash
xdg-open /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/yolo_val_html_report/index.html
```

如果没有桌面，直接打包下载：

```bash
cd /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all
zip -r yolo_val_html_report.zip yolo_val_html_report
```

下载这个文件：

```text
/home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/yolo_val_html_report.zip
```

解压后打开：

```text
yolo_val_html_report/index.html
```

如果你之前已经生成过旧版目录，建议先删掉旧目录后重新生成，避免旧的子目录和新版图片混在一起：

```bash
rm -rf /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/yolo_val_html_report
```

## 4. 只打包前 200 张

如果图很多，先看一部分：

```bash
python -m building_seg.package_yolo_prediction_html \
  --dataset /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all \
  --overlay-dir /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/prediction_overlays/val \
  --out /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/yolo_val_html_report_200 \
  --split val \
  --limit 200 \
  --title "YOLO26 Segmentation 验证集 GT 对比 - 200张"
```

然后打包：

```bash
cd /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all
zip -r yolo_val_html_report_200.zip yolo_val_html_report_200
```

## 5. 如果 prediction_overlays 不存在

先生成 YOLO 预测叠加图：

```bash
python -m building_seg.visualize_yolo_predictions \
  --model /path/to/your/yolo/weights/best.pt \
  --dataset /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all \
  --split val \
  --out /home/f50059431/code/footprint/testaoi/data/yolo26_seg_tiles_512_all/prediction_overlays/val \
  --imgsz 512 \
  --conf 0.05 \
  --device 0 \
  --limit 1000
```

把 `/path/to/your/yolo/weights/best.pt` 换成你的真实权重路径。

如果要全部验证集，不传 `--limit` 可能会按脚本默认值只跑一部分；可以显式给一个足够大的数，比如：

```bash
--limit 999999
```
