from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two LaRSE evaluation diagnosis JSON files.")
    parser.add_argument("--before", required=True, help="Baseline diagnosis.json")
    parser.add_argument("--after", required=True, help="Fine-tuned diagnosis.json")
    return parser.parse_args()


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(value):
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main():
    args = parse_args()
    before = load(args.before)
    after = load(args.after)

    fields = [
        ("samples", "samples"),
        ("fg_acc_positive_samples", "fg_acc > 0"),
        ("pred_fg_positive_samples", "pred_fg > 0"),
        ("avg_foreground_accuracy", "avg foreground accuracy"),
    ]
    pixel_fields = [
        ("overall_accuracy", "pixel OA"),
        ("miou", "pixel mIoU"),
        ("foreground_miou", "foreground mIoU"),
    ]

    print("Overall")
    print("-------")
    print(f"{'metric':<28} {'before':>14} {'after':>14} {'delta':>14}")
    for key, title in fields:
        b = before.get(key)
        a = after.get(key)
        delta = a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        print(f"{title:<28} {fmt(b):>14} {fmt(a):>14} {fmt(delta):>14}")

    print("\nPixel Metrics")
    print("-------------")
    print(f"{'metric':<28} {'before':>14} {'after':>14} {'delta':>14}")
    for key, title in pixel_fields:
        b = (before.get("pixel_metrics") or {}).get(key)
        a = (after.get("pixel_metrics") or {}).get(key)
        delta = a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        print(f"{title:<28} {fmt(b):>14} {fmt(a):>14} {fmt(delta):>14}")

    before_rows = {
        row["class_id"]: row
        for row in (before.get("pixel_metrics") or {}).get("rows", [])
    }
    after_rows = {
        row["class_id"]: row
        for row in (after.get("pixel_metrics") or {}).get("rows", [])
    }
    class_ids = sorted(set(before_rows) | set(after_rows))
    if class_ids:
        print("\nPer-class IoU")
        print("-------------")
        print(f"{'cls':>3} {'class':<24} {'before':>10} {'after':>10} {'delta':>10}")
        for class_id in class_ids:
            b_row = before_rows.get(class_id, {})
            a_row = after_rows.get(class_id, {})
            name = a_row.get("class_name") or b_row.get("class_name") or str(class_id)
            b = b_row.get("iou")
            a = a_row.get("iou")
            delta = a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            print(f"{class_id:>3} {name[:24]:<24} {fmt(b):>10} {fmt(a):>10} {fmt(delta):>10}")


if __name__ == "__main__":
    main()
