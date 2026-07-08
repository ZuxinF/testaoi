from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from building_seg.predict_tiles_larse_to_polygon import (
    LARSE_LABELS,
    DEFAULT_LARSE_TO_TARGET,
    image_to_tensor,
    load_larse_module,
    load_target_class_names,
    make_remap,
    resolve_larse_paths,
)


PALETTE = [
    (235, 235, 235),
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (90, 90, 90),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LaRSE on prepared debug image/mask patches and visualize with GT.")
    parser.add_argument("--dataset", default="data/building_seg_tiles_512_debug")
    parser.add_argument("--out", default="data/larse_debug_eval")
    parser.add_argument("--class-json", default=None, help="Defaults to <dataset>/metadata/dataset.json")
    parser.add_argument("--split", default=None, choices=["train", "val"], help="Optional split file to use")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--larse-dir", default=None)
    parser.add_argument("--larse-data-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backbone", default="clip_vitb32_384")
    parser.add_argument("--dataset-name", default="buff1w")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def colorize(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        color = PALETTE[int(class_id) % len(PALETTE)]
        rgb[mask == class_id] = color
    return Image.fromarray(rgb)


def make_error_overlay(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    rgb = np.full((*gt.shape, 3), 238, dtype=np.uint8)
    fg = gt > 0
    rgb[fg & (gt == pred)] = (40, 180, 90)
    rgb[fg & (gt != pred)] = (220, 55, 45)
    rgb[(gt == 0) & (pred > 0)] = (255, 170, 55)
    return Image.fromarray(rgb)


def alpha_overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGBA")
    alpha = ((mask > 0).astype(np.uint8) * 95)
    layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    layer[..., 0] = color[0]
    layer[..., 1] = color[1]
    layer[..., 2] = color[2]
    layer[..., 3] = alpha
    return Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA")).convert("RGB")


def load_sample_ids(dataset: Path, split: str | None) -> list[str]:
    if split:
        split_path = dataset / "splits" / f"{split}.txt"
        return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    return sorted(path.stem for path in (dataset / "images").glob("*.png"))


def foreground_accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    fg = gt > 0
    if not np.any(fg):
        return 0.0
    return float(np.count_nonzero((gt == pred) & fg) / np.count_nonzero(fg))


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    class_json = Path(args.class_json) if args.class_json else dataset / "metadata" / "dataset.json"
    class_names = load_target_class_names(str(class_json))
    remap = make_remap(class_names)

    larse_args = SimpleNamespace(
        larse_dir=args.larse_dir,
        larse_data_path=args.larse_data_path,
        checkpoint=args.checkpoint,
        device=args.device,
        dataset=args.dataset_name,
        backbone=args.backbone,
    )
    resolve_larse_paths(larse_args)
    module, transform = load_larse_module(larse_args)

    out_dir = Path(args.out)
    for name in ["pred_masks", "larse_raw_masks", "gt_color", "pred_color", "error", "gt_overlay", "pred_overlay"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    sample_ids = load_sample_ids(dataset, args.split)[: args.limit]
    records = []
    with torch.no_grad():
        for sample_id in sample_ids:
            image_path = dataset / "images" / f"{sample_id}.png"
            gt_path = dataset / "masks" / f"{sample_id}.png"
            if not image_path.exists() or not gt_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
            logits = module(image_to_tensor(image, transform).to(args.device)).detach()
            if logits.shape[-2:] != (image.size[1], image.size[0]):
                logits = F.interpolate(logits, size=(image.size[1], image.size[0]), mode="bilinear", align_corners=False)
            larse_raw = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred = remap[larse_raw]

            Image.fromarray(pred).save(out_dir / "pred_masks" / f"{sample_id}.png")
            Image.fromarray(larse_raw + 1).save(out_dir / "larse_raw_masks" / f"{sample_id}.png")
            colorize(gt).save(out_dir / "gt_color" / f"{sample_id}.png")
            colorize(pred).save(out_dir / "pred_color" / f"{sample_id}.png")
            make_error_overlay(gt, pred).save(out_dir / "error" / f"{sample_id}.png")
            alpha_overlay(image, gt, (40, 180, 90)).save(out_dir / "gt_overlay" / f"{sample_id}.jpg")
            alpha_overlay(image, pred, (230, 60, 45)).save(out_dir / "pred_overlay" / f"{sample_id}.jpg")

            records.append(
                {
                    "sample_id": sample_id,
                    "foreground_accuracy": foreground_accuracy(gt, pred),
                    "gt_foreground_pixels": int(np.count_nonzero(gt > 0)),
                    "pred_foreground_pixels": int(np.count_nonzero(pred > 0)),
                }
            )
            print(f"{sample_id}: fg_acc={records[-1]['foreground_accuracy']:.4f}")

    (out_dir / "metrics.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out_dir, records, class_names)
    print(f"Wrote LaRSE debug visualization to {out_dir / 'index.html'}")


def write_html(out_dir: Path, records: list[dict], class_names: list[str]):
    cards = []
    for rec in records:
        sid = rec["sample_id"]
        cards.append(
            f"""
      <article>
        <h3>{sid}</h3>
        <p>前景像素准确率：{rec['foreground_accuracy'] * 100:.2f}% ｜ GT前景像素：{rec['gt_foreground_pixels']} ｜ 预测前景像素：{rec['pred_foreground_pixels']}</p>
        <div class="grid">
          <figure><img src="../building_seg_tiles_512_debug/images/{sid}.png"><figcaption>原图</figcaption></figure>
          <figure><img src="gt_overlay/{sid}.jpg"><figcaption>GT 叠加</figcaption></figure>
          <figure><img src="pred_overlay/{sid}.jpg"><figcaption>LaRSE 预测叠加</figcaption></figure>
          <figure><img src="error/{sid}.png"><figcaption>绿色正确 / 红色错分 / 橙色误检</figcaption></figure>
          <figure><img src="gt_color/{sid}.png"><figcaption>GT 类别 mask</figcaption></figure>
          <figure><img src="pred_color/{sid}.png"><figcaption>LaRSE 映射后 mask</figcaption></figure>
        </div>
      </article>
"""
        )
    legend = "".join(f"<span>{idx}: {name}</span>" for idx, name in enumerate(class_names))
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LaRSE Debug GT 对比</title>
  <style>
    body {{ margin: 0; background: #f5f7f8; color: #1f2933; font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", Arial, sans-serif; }}
    main {{ width: min(1320px, calc(100% - 32px)); margin: 24px auto 48px; }}
    header, article {{ background: #fff; border: 1px solid #d8dee5; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
    h1, h3, p {{ margin-top: 0; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; color: #64707d; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    figure {{ margin: 0; border: 1px solid #d8dee5; border-radius: 6px; overflow: hidden; background: #fff; }}
    img {{ display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: contain; image-rendering: pixelated; }}
    figcaption {{ padding: 6px 8px; border-top: 1px solid #d8dee5; text-align: center; color: #64707d; font-size: 12px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>LaRSE Debug GT 对比</h1>
      <p>使用已生成的 debug image/mask 作为 GT，对同一批 patch 运行 LaRSE 迁移推理。</p>
      <div class="legend">{legend}</div>
    </header>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
