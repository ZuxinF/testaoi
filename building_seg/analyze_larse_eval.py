from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from building_seg.predict_tiles_larse_to_polygon import LARSE_LABELS, make_remap


LARSE_LABELS_1_BASED = {
    1: "dense residential",
    2: "business",
    3: "commercial",
    4: "residential",
    5: "factory",
    6: "government",
    7: "hospital",
    8: "resort",
    9: "public",
    10: "school",
    11: "background",
    12: "others",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose LaRSE transfer outputs against prepared GT masks.")
    parser.add_argument("--eval-dir", required=True, help="Output directory from predict_larse_debug_dataset")
    parser.add_argument("--dataset", required=True, help="Prepared dataset directory containing masks/metadata")
    parser.add_argument("--limit", type=int, default=None, help="Analyze at most this many evaluated samples")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--label-space", default="auto", choices=["auto", "larse", "target"])
    parser.add_argument("--out-json", default=None, help="Optional JSON report path")
    return parser.parse_args()


def load_counter(paths: list[Path]) -> Counter:
    counter: Counter[int] = Counter()
    for path in paths:
        arr = np.asarray(Image.open(path))
        values, counts = np.unique(arr, return_counts=True)
        counter.update({int(value): int(count) for value, count in zip(values, counts)})
    return counter


def counter_to_rows(counter: Counter, names: dict[int, str] | list[str] | None = None, top: int = 20):
    total = sum(counter.values())
    rows = []
    for class_id, pixels in counter.most_common(top):
        if isinstance(names, dict):
            name = names.get(class_id, "")
        elif isinstance(names, list) and 0 <= class_id < len(names):
            name = names[class_id]
        else:
            name = ""
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "pixels": pixels,
                "ratio": pixels / total if total else 0.0,
            }
        )
    return rows


def print_rows(title: str, rows: list[dict]):
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        name = f" {row['class_name']}" if row["class_name"] else ""
        print(f"{row['class_id']:>3}{name:<24} {row['pixels']:>12}  {row['ratio'] * 100:7.3f}%")


def resolve_label_space(eval_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    config_path = eval_dir / "eval_config.json"
    if config_path.exists():
        try:
            value = json.loads(config_path.read_text(encoding="utf-8")).get("label_space")
            if value in {"larse", "target"}:
                return value
        except Exception:
            pass
    return "larse"


def make_diagnosis(metrics: list[dict], raw_counter: Counter, pred_counter: Counter, gt_counter: Counter, label_space: str) -> list[str]:
    diagnosis = []
    sample_count = len(metrics)
    fg_acc_positive = sum(float(row.get("foreground_accuracy", 0)) > 0 for row in metrics)
    pred_fg_positive = sum(int(row.get("pred_foreground_pixels", 0)) > 0 for row in metrics)
    gt_fg_positive = sum(int(row.get("gt_foreground_pixels", 0)) > 0 for row in metrics)

    remapped_foreground_pixels = sum(v for k, v in pred_counter.items() if k > 0)
    gt_foreground_pixels = sum(v for k, v in gt_counter.items() if k > 0)

    if sample_count == 0:
        return ["没有读到 metrics.json 记录，请先确认 LaRSE 可视化命令是否跑完。"]

    if gt_fg_positive == 0 or gt_foreground_pixels == 0:
        diagnosis.append("GT 前景为空或没有读到 GT mask，先检查 dataset/masks 和 split 是否对应。")
    if label_space == "larse":
        raw_total = sum(raw_counter.values())
        raw_background_ratio = raw_counter.get(11, 0) / raw_total if raw_total else 0.0
        if raw_background_ratio > 0.995:
            diagnosis.append("LaRSE 原始 1-12 类几乎全是 background：更像模型对当前影像直接迁移失败，或输入预处理/权重加载异常。")
    if remapped_foreground_pixels == 0:
        diagnosis.append("LaRSE 原始输出有非背景，但 remap 后全成背景：优先检查 LaRSE 类别到 Function 类别的映射。")
    elif pred_fg_positive > 0 and fg_acc_positive == 0:
        diagnosis.append("LaRSE 有预测前景，但和 GT 前景类别没有命中：打开 HTML 区分 GT 漏标、类别映射不合适，还是位置完全错。")
    elif fg_acc_positive > 0:
        diagnosis.append("LaRSE 至少有部分样本命中 GT 前景，可继续看 HTML 和逐类分布判断是否有可用迁移能力。")

    if pred_fg_positive == 0:
        diagnosis.append("所有样本 pred_foreground_pixels 都为 0：当前结果等价于全背景预测。")
    return diagnosis


def compute_class_metrics(pred_paths: list[Path], gt_paths: list[Path], class_names: list[str]):
    nclass = len(class_names)
    confusion = np.zeros((nclass, nclass), dtype=np.int64)
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        if not pred_path.exists() or not gt_path.exists():
            continue
        pred = np.asarray(Image.open(pred_path), dtype=np.int64)
        gt = np.asarray(Image.open(gt_path), dtype=np.int64)
        valid = (gt >= 0) & (gt < nclass) & (pred >= 0) & (pred < nclass)
        idx = gt[valid] * nclass + pred[valid]
        confusion += np.bincount(idx.ravel(), minlength=nclass * nclass).reshape(nclass, nclass)

    rows = []
    for class_id, name in enumerate(class_names):
        tp = int(confusion[class_id, class_id])
        gt_pixels = int(confusion[class_id, :].sum())
        pred_pixels = int(confusion[:, class_id].sum())
        union = gt_pixels + pred_pixels - tp
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "intersection": tp,
                "union": int(union),
                "iou": tp / union if union else None,
                "precision": tp / pred_pixels if pred_pixels else None,
                "recall": tp / gt_pixels if gt_pixels else None,
            }
        )
    valid_ious = [row["iou"] for row in rows if row["iou"] is not None]
    foreground_ious = [row["iou"] for row in rows[1:] if row["iou"] is not None]
    overall_accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    return {
        "overall_accuracy": overall_accuracy,
        "miou": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "foreground_miou": float(np.mean(foreground_ious)) if foreground_ious else 0.0,
        "rows": rows,
        "confusion_matrix": confusion.tolist(),
    }


def print_class_metrics(metrics: dict):
    print("\nPer-class Pixel Metrics")
    print("-----------------------")
    print(f"overall_accuracy: {metrics['overall_accuracy']:.6f}")
    print(f"mIoU:             {metrics['miou']:.6f}")
    print(f"foreground_mIoU:  {metrics['foreground_miou']:.6f}")
    print("\n cls class_name                 IoU      Prec     Recall       GT_px     Pred_px")
    for row in metrics["rows"]:
        iou = "nan" if row["iou"] is None else f"{row['iou']:.4f}"
        precision = "nan" if row["precision"] is None else f"{row['precision']:.4f}"
        recall = "nan" if row["recall"] is None else f"{row['recall']:.4f}"
        print(
            f"{row['class_id']:>4} {row['class_name'][:22]:<22} "
            f"{iou:>8} {precision:>8} {recall:>8} "
            f"{row['gt_pixels']:>11} {row['pred_pixels']:>11}"
        )


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    dataset = Path(args.dataset)

    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        metrics = metrics[: args.limit]

    sample_ids = [row["sample_id"] for row in metrics]
    raw_paths = [eval_dir / "larse_raw_masks" / f"{sample_id}.png" for sample_id in sample_ids]
    pred_paths = [eval_dir / "pred_masks" / f"{sample_id}.png" for sample_id in sample_ids]
    gt_paths = [dataset / "masks" / f"{sample_id}.png" for sample_id in sample_ids]

    raw_paths = [path for path in raw_paths if path.exists()]
    pred_paths = [path for path in pred_paths if path.exists()]
    gt_paths = [path for path in gt_paths if path.exists()]

    metadata_path = dataset / "metadata" / "dataset.json"
    target_names: list[str] = []
    if metadata_path.exists():
        target_names = json.loads(metadata_path.read_text(encoding="utf-8")).get("class_names", [])
    label_space = resolve_label_space(eval_dir, args.label_space)

    raw_counter = load_counter(raw_paths)
    pred_counter = load_counter(pred_paths)
    gt_counter = load_counter(gt_paths)
    simulated_pred_counter: Counter[int] = Counter()
    if label_space == "larse" and target_names and raw_paths:
        remap = make_remap(target_names)
        for path in raw_paths:
            raw = np.asarray(Image.open(path), dtype=np.uint8)
            raw_zero_based = np.clip(raw.astype(np.int16) - 1, 0, len(LARSE_LABELS) - 1).astype(np.uint8)
            pred = remap[raw_zero_based]
            values, counts = np.unique(pred, return_counts=True)
            simulated_pred_counter.update({int(value): int(count) for value, count in zip(values, counts)})

    sample_count = len(metrics)
    avg_fg_acc = sum(float(row.get("foreground_accuracy", 0)) for row in metrics) / max(sample_count, 1)
    pred_fg_positive = sum(int(row.get("pred_foreground_pixels", 0)) > 0 for row in metrics)
    gt_fg_positive = sum(int(row.get("gt_foreground_pixels", 0)) > 0 for row in metrics)

    print(f"eval_dir: {eval_dir}")
    print(f"dataset:  {dataset}")
    print(f"label_space: {label_space}")
    print(f"samples:  {sample_count}")
    print(f"fg_acc > 0 samples:  {sum(float(row.get('foreground_accuracy', 0)) > 0 for row in metrics)}")
    print(f"pred_fg > 0 samples: {pred_fg_positive}")
    print(f"gt_fg > 0 samples:   {gt_fg_positive}")
    print(f"avg foreground_accuracy: {avg_fg_acc:.6f}")
    print(f"avg pred_foreground_pixels: {sum(int(row.get('pred_foreground_pixels', 0)) for row in metrics) / max(sample_count, 1):.2f}")
    print(f"avg gt_foreground_pixels:   {sum(int(row.get('gt_foreground_pixels', 0)) for row in metrics) / max(sample_count, 1):.2f}")

    if label_space == "target":
        raw_names = {idx + 1: name for idx, name in enumerate(target_names)}
        raw_rows = counter_to_rows(raw_counter, raw_names, args.top)
    else:
        raw_rows = counter_to_rows(raw_counter, LARSE_LABELS_1_BASED, args.top)
    pred_rows = counter_to_rows(pred_counter, target_names, args.top)
    simulated_pred_rows = counter_to_rows(simulated_pred_counter, target_names, args.top)
    gt_rows = counter_to_rows(gt_counter, target_names, args.top)

    raw_title = "Raw target argmax classes saved as +1" if label_space == "target" else "Raw LaRSE classes, 1-12"
    print_rows(raw_title, raw_rows)
    print_rows("Remapped prediction classes", pred_rows)
    if simulated_pred_counter and simulated_pred_counter != pred_counter:
        print_rows("Remapped prediction classes with current code", simulated_pred_rows)
    print_rows("GT classes", gt_rows)
    class_metrics = compute_class_metrics(pred_paths, gt_paths, target_names) if target_names else None
    if class_metrics:
        print_class_metrics(class_metrics)

    diagnosis = make_diagnosis(metrics, raw_counter, pred_counter, gt_counter, label_space)
    print("\nDiagnosis")
    print("---------")
    for item in diagnosis:
        print(f"- {item}")

    report = {
        "eval_dir": str(eval_dir),
        "dataset": str(dataset),
        "samples": sample_count,
        "label_space": label_space,
        "fg_acc_positive_samples": sum(float(row.get("foreground_accuracy", 0)) > 0 for row in metrics),
        "pred_fg_positive_samples": pred_fg_positive,
        "gt_fg_positive_samples": gt_fg_positive,
        "avg_foreground_accuracy": avg_fg_acc,
        "raw_larse_classes": raw_rows,
        "remapped_prediction_classes": pred_rows,
        "remapped_prediction_classes_with_current_code": simulated_pred_rows,
        "gt_classes": gt_rows,
        "pixel_metrics": class_metrics,
        "diagnosis": diagnosis,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {out_path}")


if __name__ == "__main__":
    main()
