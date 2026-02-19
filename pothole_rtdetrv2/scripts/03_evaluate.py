#!/usr/bin/env python3
"""RT-DETRv2 modeli için recall-odaklı detaylı evaluation yapar.

Dependencies:
    - pycocotools
    - pandas
    - matplotlib

Usage:
    python scripts/03_evaluate.py \
      --gt-json /content/dataset/combined_dataset/annotations/val_annotations.json \
      --pred-json /content/outputs/predictions_val.json \
      --output-dir /content/outputs/eval
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI argümanlarını parse eder."""
    parser = argparse.ArgumentParser(description="Recall focused evaluator")
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--pred-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recall-target", type=float, default=0.85)
    parser.add_argument("--f1-floor", type=float, default=0.70)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    """JSON dosyasını okur."""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def box_iou_xywh(a: List[float], b: List[float]) -> float:
    """[x,y,w,h] kutular için IoU hesaplar."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_tp_fp_fn(
    gt_by_img: Dict[int, List[Dict]], pred_by_img: Dict[int, List[Dict]], threshold: float
) -> Tuple[int, int, int, List[Dict], List[Dict]]:
    """Belirli threshold için TP/FP/FN döndürür."""
    tp = fp = fn = 0
    matched_gt_records: List[Dict] = []
    missed_gt_records: List[Dict] = []

    all_image_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())
    for img_id in all_image_ids:
        gts = gt_by_img.get(img_id, [])
        preds = [p for p in pred_by_img.get(img_id, []) if p.get("score", 0.0) >= threshold]
        preds.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        used_gt = set()
        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gts):
                if idx in used_gt:
                    continue
                iou = box_iou_xywh(gt["bbox"], pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= 0.5 and best_gt_idx >= 0:
                tp += 1
                used_gt.add(best_gt_idx)
                matched_gt_records.append(gts[best_gt_idx])
            else:
                fp += 1

        for idx, gt in enumerate(gts):
            if idx not in used_gt:
                fn += 1
                missed_gt_records.append(gt)

    return tp, fp, fn, matched_gt_records, missed_gt_records


def size_bucket(area: float) -> str:
    """COCO size bucket etiketini döndürür."""
    if area < 32 * 32:
        return "small"
    if area <= 96 * 96:
        return "medium"
    return "large"


def compute_size_recalls(matched: List[Dict], missed: List[Dict]) -> Dict[str, float]:
    """Boyut bazında recall hesaplar."""
    total = defaultdict(int)
    hit = defaultdict(int)

    for ann in matched:
        b = size_bucket(float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])))
        total[b] += 1
        hit[b] += 1

    for ann in missed:
        b = size_bucket(float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])))
        total[b] += 1

    out = {}
    for b in ["small", "medium", "large"]:
        out[f"recall_{b}"] = (hit[b] / total[b]) if total[b] > 0 else 0.0
    return out


def pick_optimal_threshold(
    df: pd.DataFrame, recall_target: float, f1_floor: float
) -> Tuple[float, str]:
    """Safety-first kurallarına göre optimal threshold seçer."""
    c1 = df[df["recall"] >= recall_target]
    if not c1.empty:
        best = c1.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        return float(best["threshold"]), f"Recall>={recall_target} koşulunda en yüksek F1"

    c2 = df[df["f1"] >= f1_floor]
    if not c2.empty:
        best = c2.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
        return float(best["threshold"]), f"F1>={f1_floor} koşulunda en yüksek recall"

    best = df.sort_values(["recall", "f1"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), "Fallback: en yüksek recall"


def run_coco_eval(gt_json: Path, pred_json: Path) -> Dict[str, float]:
    """pycocotools ile standart COCO metriklerini hesaplar."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        LOGGER.warning("pycocotools yok, COCO metrics atlandı: %s", exc)
        return {}

    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR": float(evaluator.stats[8]),
    }


def main() -> None:
    """Program giriş noktası."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_json(args.gt_json)
    preds = load_json(args.pred_json)

    gt_by_img = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_img[ann["image_id"]].append(ann)

    pred_by_img = defaultdict(list)
    for pred in preds:
        pred_by_img[pred["image_id"]].append(pred)

    rows = []
    thresholds = np.round(np.arange(0.1, 0.901, 0.05), 2)
    for thr in thresholds:
        tp, fp, fn, _, _ = match_tp_fp_fn(gt_by_img, pred_by_img, float(thr))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": precision,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "threshold_sweep.csv", index=False)
    LOGGER.info("Threshold tablosu kaydedildi: %s", args.output_dir / "threshold_sweep.csv")

    best_thr, reason = pick_optimal_threshold(df, args.recall_target, args.f1_floor)
    tp, fp, fn, matched, missed = match_tp_fp_fn(gt_by_img, pred_by_img, best_thr)
    size_recalls = compute_size_recalls(matched, missed)

    coco_metrics = run_coco_eval(args.gt_json, args.pred_json)

    plt.figure(figsize=(8, 6))
    plt.plot(df["recall"], df["precision"], marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(args.output_dir / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    result = {
        "optimal_threshold": best_thr,
        "selection_reason": reason,
        "at_optimal_threshold": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        },
        "size_based_recall": size_recalls,
        "coco_metrics": coco_metrics,
    }

    out_json = args.output_dir / "evaluation_results.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Evaluation sonucu kaydedildi: %s", out_json)
    LOGGER.info("Seçilen threshold=%.2f | Gerekçe=%s", best_thr, reason)


if __name__ == "__main__":
    main()
