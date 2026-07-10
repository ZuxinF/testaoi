from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PALETTE = [
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
    parser = argparse.ArgumentParser(description="Package YOLO prediction overlays, source images, and GT labels into an HTML report.")
    parser.add_argument("--dataset", required=True, help="YOLO segmentation dataset directory, e.g. data/yolo26_seg_tiles_512_all")
    parser.add_argument("--overlay-dir", required=True, help="Directory containing YOLO prediction overlay images")
    parser.add_argument("--out", required=True, help="Output report directory")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--title", default="YOLO Segmentation GT 对比")
    parser.add_argument("--copy-overlay", action="store_true", default=True)
    parser.add_argument("--gt-alpha", type=int, default=88)
    return parser.parse_args()


def load_class_names(dataset: Path) -> list[str]:
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        return []
    try:
        import yaml

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = data.get("names", [])
        if isinstance(names, dict):
            return [str(names[idx]) for idx in sorted(names)]
        if isinstance(names, list):
            return [str(name) for name in names]
    except Exception:
        pass

    names: list[str] = []
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            names.append(stripped[2:].strip().strip("'\""))
    return names


def read_yolo_segments(label_path: Path, width: int, height: int):
    segments = []
    if not label_path.exists():
        return segments
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = np.asarray([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        except ValueError:
            continue
        points = [(float(x * width), float(y * height)) for x, y in coords]
        if len(points) >= 3:
            segments.append((class_id, points))
    return segments


def draw_gt_overlay(image: Image.Image, segments, alpha: int) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(overlay)
    line_draw = ImageDraw.Draw(overlay)
    for class_id, points in segments:
        color = PALETTE[class_id % len(PALETTE)]
        fill_draw.polygon(points, fill=(*color, alpha))
    for class_id, points in segments:
        color = PALETTE[class_id % len(PALETTE)]
        line_draw.line(points + [points[0]], fill=(*color, 255), width=2)
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_gt_mask(size: tuple[int, int], segments) -> Image.Image:
    mask = Image.new("RGB", size, (245, 247, 248))
    draw = ImageDraw.Draw(mask)
    for class_id, points in segments:
        color = PALETTE[class_id % len(PALETTE)]
        draw.polygon(points, fill=color)
    for class_id, points in segments:
        draw.line(points + [points[0]], fill=(255, 255, 255), width=1)
    return mask


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def collect_overlay_paths(overlay_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(
        [
            *overlay_dir.glob("*.jpg"),
            *overlay_dir.glob("*.jpeg"),
            *overlay_dir.glob("*.png"),
        ]
    )
    if limit is not None:
        paths = paths[:limit]
    return paths


def write_html(out_dir: Path, title: str, records: list[dict], class_names: list[str], summary: dict):
    cards = []
    for rec in records:
        class_bits = []
        for item in rec["gt_classes"]:
            name = class_names[item["class_id"]] if item["class_id"] < len(class_names) else str(item["class_id"])
            class_bits.append(f"{html.escape(name)} x {item['count']}")
        class_text = " ｜ ".join(class_bits) if class_bits else "无 GT polygon"
        cards.append(
            f"""
      <article>
        <h3>{html.escape(rec['sample_id'])}</h3>
        <p>GT polygon：{rec['gt_polygon_count']} ｜ GT 类别：{class_text}</p>
        <div class="grid">
          <figure><img src="images/{html.escape(rec['image_file'])}"><figcaption>原图</figcaption></figure>
          <figure><img src="gt_overlay/{html.escape(rec['gt_overlay_file'])}"><figcaption>GT 叠加</figcaption></figure>
          <figure><img src="prediction_overlay/{html.escape(rec['overlay_file'])}"><figcaption>YOLO 预测 + GT</figcaption></figure>
          <figure><img src="gt_mask/{html.escape(rec['gt_mask_file'])}"><figcaption>GT mask 预览</figcaption></figure>
        </div>
      </article>
"""
        )

    legend_items = []
    for idx, name in enumerate(class_names):
        color = PALETTE[idx % len(PALETTE)]
        legend_items.append(
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
    main {{ width: min(1320px, calc(100% - 32px)); margin: 24px auto 48px; }}
    header, article {{ background: #fff; border: 1px solid #d8dee5; border-radius: 8px; padding: 18px; margin-bottom: 16px; box-shadow: 0 8px 22px rgba(31, 41, 51, 0.06); }}
    h1, h3, p {{ margin-top: 0; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
    .summary span {{ background: #eef1f3; border-radius: 999px; padding: 4px 10px; font-size: 13px; color: #37424f; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 14px; color: #64707d; font-size: 13px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.15); }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
    figure {{ margin: 0; border: 1px solid #d8dee5; border-radius: 6px; overflow: hidden; background: #fff; }}
    img {{ display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: contain; image-rendering: pixelated; }}
    figcaption {{ padding: 6px 8px; border-top: 1px solid #d8dee5; text-align: center; color: #64707d; font-size: 12px; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <p>绿色/彩色 GT 轮廓来自 YOLO label；YOLO 预测叠加图来自已有 prediction_overlays 目录，不重新推理。</p>
      <div class="summary">
        <span>样本数：{summary['sample_count']}</span>
        <span>GT polygon：{summary['gt_polygon_count']}</span>
        <span>缺失原图：{summary['missing_images']}</span>
        <span>缺失 label：{summary['missing_labels']}</span>
      </div>
      <div class="legend">{''.join(legend_items)}</div>
    </header>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    overlay_dir = Path(args.overlay_dir)
    out_dir = Path(args.out)
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split

    for name in ["images", "gt_overlay", "gt_mask", "prediction_overlay"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(dataset)
    overlay_paths = collect_overlay_paths(overlay_dir, args.limit)
    if not overlay_paths:
        raise RuntimeError(f"No overlay images found in {overlay_dir}")

    records = []
    missing_images = 0
    missing_labels = 0
    gt_polygon_total = 0

    for overlay_path in overlay_paths:
        sample_id = overlay_path.stem
        image_path = image_dir / f"{sample_id}.jpg"
        if not image_path.exists():
            png_path = image_dir / f"{sample_id}.png"
            image_path = png_path if png_path.exists() else image_path
        label_path = label_dir / f"{sample_id}.txt"

        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            image_file = f"{sample_id}{image_path.suffix.lower()}"
            safe_copy(image_path, out_dir / "images" / image_file)
        else:
            missing_images += 1
            image = Image.new("RGB", (512, 512), (245, 247, 248))
            image_file = f"{sample_id}.jpg"
            image.save(out_dir / "images" / image_file)

        if not label_path.exists():
            missing_labels += 1
        segments = read_yolo_segments(label_path, *image.size)
        gt_polygon_total += len(segments)

        gt_overlay_file = f"{sample_id}.jpg"
        gt_mask_file = f"{sample_id}.png"
        overlay_file = f"{sample_id}{overlay_path.suffix.lower()}"

        draw_gt_overlay(image, segments, args.gt_alpha).save(out_dir / "gt_overlay" / gt_overlay_file, quality=95)
        draw_gt_mask(image.size, segments).save(out_dir / "gt_mask" / gt_mask_file)
        safe_copy(overlay_path, out_dir / "prediction_overlay" / overlay_file)

        class_counter: dict[int, int] = {}
        for class_id, _ in segments:
            class_counter[class_id] = class_counter.get(class_id, 0) + 1
        records.append(
            {
                "sample_id": sample_id,
                "image_file": image_file,
                "gt_overlay_file": gt_overlay_file,
                "gt_mask_file": gt_mask_file,
                "overlay_file": overlay_file,
                "gt_polygon_count": len(segments),
                "gt_classes": [
                    {"class_id": class_id, "count": count}
                    for class_id, count in sorted(class_counter.items(), key=lambda item: item[0])
                ],
            }
        )

    summary = {
        "sample_count": len(records),
        "gt_polygon_count": gt_polygon_total,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "dataset": str(dataset),
        "overlay_dir": str(overlay_dir),
        "split": args.split,
        "class_names": class_names,
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html(out_dir, args.title, records, class_names, summary)
    print(f"Wrote YOLO HTML report to {out_dir / 'index.html'}")
    print(f"Packaged {len(records)} samples, GT polygons: {gt_polygon_total}")


if __name__ == "__main__":
    main()
