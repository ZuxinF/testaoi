from __future__ import annotations

import argparse
import html
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = np.array(
    [
        [238, 238, 238],
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [90, 90, 90],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Package fine-tuned LaRSE non-background visualizations into a flat HTML report.")
    parser.add_argument("--eval-dir", required=True, help="Directory from predict_larse_debug_dataset")
    parser.add_argument("--dataset", required=True, help="Prepared dataset directory with images/masks/metadata")
    parser.add_argument("--out", required=True, help="Output report directory")
    parser.add_argument("--class-json", default=None, help="Defaults to <dataset>/metadata/dataset.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sort", default="fg_acc_asc", choices=["fg_acc_asc", "fg_acc_desc", "sample_id"])
    parser.add_argument("--min-gt-foreground", type=int, default=1)
    parser.add_argument("--min-pred-foreground", type=int, default=1)
    parser.add_argument("--title", default="LaRSE Fine-tuned 非背景结果对比")
    parser.add_argument("--make-zip", action="store_true", help="Also create <out>.zip")
    return parser.parse_args()


def load_class_names(dataset: Path, class_json: str | None) -> list[str]:
    path = Path(class_json) if class_json else dataset / "metadata" / "dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    class_names = data.get("class_names")
    if not class_names or class_names[0] != "background":
        raise ValueError(f"{path} must contain class_names with background at index 0")
    return [str(name) for name in class_names]


def find_image_path(dataset: Path, sample_id: str) -> Path | None:
    for suffix in [".png", ".jpg", ".jpeg"]:
        path = dataset / "images" / f"{sample_id}{suffix}"
        if path.exists():
            return path
    return None


def colorize(mask: np.ndarray, background=(245, 247, 248)) -> Image.Image:
    max_id = int(mask.max()) if mask.size else 0
    if max_id >= len(PALETTE):
        rng = np.random.default_rng(20260714)
        extra = rng.integers(30, 230, size=(max_id + 1 - len(PALETTE), 3), dtype=np.uint8)
        palette = np.vstack([PALETTE, extra])
    else:
        palette = PALETTE
    rgb = palette[mask]
    rgb[mask == 0] = np.array(background, dtype=np.uint8)
    return Image.fromarray(rgb.astype(np.uint8))


def foreground_overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 95) -> Image.Image:
    base = image.convert("RGBA")
    layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    layer[..., 0] = color[0]
    layer[..., 1] = color[1]
    layer[..., 2] = color[2]
    layer[..., 3] = (mask > 0).astype(np.uint8) * alpha
    return Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA")).convert("RGB")


def error_map(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    rgb = np.full((*gt.shape, 3), 238, dtype=np.uint8)
    gt_fg = gt > 0
    pred_fg = pred > 0
    rgb[gt_fg & pred_fg & (gt == pred)] = (40, 180, 90)
    rgb[gt_fg & ~pred_fg] = (220, 55, 45)
    rgb[gt_fg & pred_fg & (gt != pred)] = (190, 75, 170)
    rgb[~gt_fg & pred_fg] = (255, 170, 55)
    return Image.fromarray(rgb)


def class_count(mask: np.ndarray, class_names: list[str]) -> list[dict]:
    counter = Counter(int(v) for v in mask.reshape(-1).tolist() if int(v) > 0)
    rows = []
    for class_id, pixels in counter.most_common():
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        rows.append({"class_id": class_id, "class_name": name, "pixels": pixels})
    return rows


def format_class_count(rows: list[dict]) -> str:
    if not rows:
        return "无非背景"
    return " ｜ ".join(f"{html.escape(row['class_name'])}: {row['pixels']}" for row in rows[:5])


def copy_or_make_image(src: Path | None, dst: Path, fallback_size=(512, 512)) -> Image.Image:
    if src and src.exists():
        image = Image.open(src).convert("RGB")
        image.save(dst, quality=95)
        return image
    image = Image.new("RGB", fallback_size, (245, 247, 248))
    image.save(dst, quality=95)
    return image


def write_html(out_dir: Path, title: str, records: list[dict], class_names: list[str], summary: dict):
    cards = []
    for rec in records:
        cards.append(
            f"""
      <article>
        <h3>{html.escape(rec['sample_id'])}</h3>
        <p>
          fg_acc: {rec['foreground_accuracy'] * 100:.2f}% ｜
          GT前景像素: {rec['gt_foreground_pixels']} ｜
          预测前景像素: {rec['pred_foreground_pixels']}
        </p>
        <p>GT类别：{format_class_count(rec['gt_classes'])}</p>
        <p>预测类别：{format_class_count(rec['pred_classes'])}</p>
        <div class="grid">
          <figure><img src="{html.escape(rec['image_file'])}"><figcaption>原图</figcaption></figure>
          <figure><img src="{html.escape(rec['pred_overlay_file'])}"><figcaption>预测非背景叠加</figcaption></figure>
          <figure><img src="{html.escape(rec['pred_mask_file'])}"><figcaption>预测非背景 mask</figcaption></figure>
          <figure><img src="{html.escape(rec['gt_mask_file'])}"><figcaption>GT 非背景 mask</figcaption></figure>
          <figure><img src="{html.escape(rec['error_file'])}"><figcaption>错误图</figcaption></figure>
        </div>
      </article>
"""
        )

    legend = []
    for idx, name in enumerate(class_names):
        color = PALETTE[idx % len(PALETTE)]
        legend.append(
            f'<span><i style="background: rgb({color[0]}, {color[1]}, {color[2]})"></i>{idx}: {html.escape(name)}</span>'
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; background: #f5f7f8; color: #1f2933; font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", Arial, sans-serif; }}
    main {{ width: min(1380px, calc(100% - 32px)); margin: 24px auto 48px; }}
    header, article {{ background: #fff; border: 1px solid #d8dee5; border-radius: 8px; padding: 18px; margin-bottom: 16px; box-shadow: 0 8px 22px rgba(31, 41, 51, 0.06); }}
    h1, h3, p {{ margin-top: 0; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
    .summary span {{ background: #eef1f3; border-radius: 999px; padding: 4px 10px; font-size: 13px; color: #37424f; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 14px; color: #64707d; font-size: 13px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.15); }}
    .grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }}
    figure {{ margin: 0; border: 1px solid #d8dee5; border-radius: 6px; overflow: hidden; background: #fff; }}
    img {{ display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: contain; image-rendering: pixelated; }}
    figcaption {{ padding: 6px 8px; border-top: 1px solid #d8dee5; text-align: center; color: #64707d; font-size: 12px; }}
    @media (max-width: 1200px) {{ .grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
    @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <p>只突出非 background 区域。错误图：绿色=类别正确，红色=漏检，紫色=类别错，橙色=背景误检。</p>
      <div class="summary">
        <span>样本数：{summary['sample_count']}</span>
        <span>平均 fg_acc：{summary['avg_foreground_accuracy'] * 100:.2f}%</span>
        <span>GT前景均值：{summary['avg_gt_foreground_pixels']:.1f}</span>
        <span>预测前景均值：{summary['avg_pred_foreground_pixels']:.1f}</span>
      </div>
      <div class="legend">{''.join(legend)}</div>
    </header>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(dataset, args.class_json)

    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    filtered = [
        row
        for row in metrics
        if row.get("gt_foreground_pixels", 0) >= args.min_gt_foreground
        or row.get("pred_foreground_pixels", 0) >= args.min_pred_foreground
    ]
    if args.sort == "fg_acc_asc":
        filtered.sort(key=lambda row: (row.get("foreground_accuracy", 0), row.get("sample_id", "")))
    elif args.sort == "fg_acc_desc":
        filtered.sort(key=lambda row: (-row.get("foreground_accuracy", 0), row.get("sample_id", "")))
    else:
        filtered.sort(key=lambda row: row.get("sample_id", ""))
    if args.limit is not None:
        filtered = filtered[: args.limit]

    records = []
    for row in filtered:
        sample_id = row["sample_id"]
        image_path = find_image_path(dataset, sample_id)
        gt_path = dataset / "masks" / f"{sample_id}.png"
        pred_path = eval_dir / "pred_masks" / f"{sample_id}.png"
        if not gt_path.exists() or not pred_path.exists():
            continue

        image_file = f"image_{sample_id}.jpg"
        image = copy_or_make_image(image_path, out_dir / image_file)
        gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
        pred = np.asarray(Image.open(pred_path), dtype=np.uint8)

        pred_overlay_file = f"pred_overlay_{sample_id}.jpg"
        pred_mask_file = f"pred_mask_{sample_id}.png"
        gt_mask_file = f"gt_mask_{sample_id}.png"
        error_file = f"error_{sample_id}.png"

        foreground_overlay(image, pred, (230, 60, 45)).save(out_dir / pred_overlay_file, quality=95)
        colorize(pred).save(out_dir / pred_mask_file)
        colorize(gt).save(out_dir / gt_mask_file)
        error_map(gt, pred).save(out_dir / error_file)

        records.append(
            {
                "sample_id": sample_id,
                "foreground_accuracy": float(row.get("foreground_accuracy", 0)),
                "gt_foreground_pixels": int(row.get("gt_foreground_pixels", int(np.count_nonzero(gt > 0)))),
                "pred_foreground_pixels": int(row.get("pred_foreground_pixels", int(np.count_nonzero(pred > 0)))),
                "image_file": image_file,
                "pred_overlay_file": pred_overlay_file,
                "pred_mask_file": pred_mask_file,
                "gt_mask_file": gt_mask_file,
                "error_file": error_file,
                "gt_classes": class_count(gt, class_names),
                "pred_classes": class_count(pred, class_names),
            }
        )

    summary = {
        "eval_dir": str(eval_dir),
        "dataset": str(dataset),
        "sample_count": len(records),
        "avg_foreground_accuracy": sum(r["foreground_accuracy"] for r in records) / max(len(records), 1),
        "avg_gt_foreground_pixels": sum(r["gt_foreground_pixels"] for r in records) / max(len(records), 1),
        "avg_pred_foreground_pixels": sum(r["pred_foreground_pixels"] for r in records) / max(len(records), 1),
        "class_names": class_names,
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html(out_dir, args.title, records, class_names, summary)
    print(f"Wrote LaRSE non-background HTML to {out_dir / 'index.html'}")
    if args.make_zip:
        zip_base = shutil.make_archive(str(out_dir), "zip", root_dir=str(out_dir))
        print(f"Wrote zip to {zip_base}")


if __name__ == "__main__":
    main()
