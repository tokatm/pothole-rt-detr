#!/usr/bin/env python3
"""FN/FP hata analizi üretir ve recall iyileştirme önerileri çıkarır.

Dependencies:
    - numpy
    - matplotlib

Usage:
    python scripts/07_analyze_errors.py \
      --gt-json /content/dataset/combined_dataset/annotations/val_annotations.json \
      --pred-json /content/outputs/predictions_val.json \
      --output-dir /content/outputs/error_analysis
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


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI argümanlarını parse eder."""
    parser = argparse.ArgumentParser(description="FN/FP analyzer")
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--pred-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf-thres", type=float, default=0.35)
    return parser.parse_args()


def load_json(path: Path):
    """JSON okur."""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def iou(a: List[float], b: List[float]) -> float:
    """[x,y,w,h] bbox IoU."""
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


def match(gt_by_img: Dict[int, List[Dict]], pred_by_img: Dict[int, List[Dict]], conf_thres: float):
    """GT ve pred eşleştirip FN/FP listesi döndürür."""
    fns, fps = [], []

    for img_id in sorted(set(gt_by_img.keys()) | set(pred_by_img.keys())):
        gts = gt_by_img.get(img_id, [])
        preds = [p for p in pred_by_img.get(img_id, []) if p.get("score", 0.0) >= conf_thres]
        preds.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        used = set()
        for pred in preds:
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gts):
                if i in used:
                    continue
                cur = iou(gt["bbox"], pred["bbox"])
                if cur > best_iou:
                    best_iou, best_idx = cur, i
            if best_iou >= 0.5 and best_idx >= 0:
                used.add(best_idx)
            else:
                fps.append({"image_id": img_id, **pred})

        for i, gt in enumerate(gts):
            if i not in used:
                fns.append({"image_id": img_id, **gt})

    return fns, fps


def center_of_bbox(box: List[float]) -> Tuple[float, float]:
    """BBox merkezini döndürür."""
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def plot_hist(values: List[float], title: str, out_path: Path) -> None:
    """Histogram çizip kaydeder."""
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_heatmap(xs: List[float], ys: List[float], title: str, out_path: Path) -> None:
    """2D spatial yoğunluk haritası üretir."""
    plt.figure(figsize=(7, 6))
    plt.hist2d(xs, ys, bins=40)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.colorbar()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


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
    for p in preds:
        pred_by_img[p["image_id"]].append(p)

    fns, fps = match(gt_by_img, pred_by_img, args.conf_thres)

    fn_areas = [x["bbox"][2] * x["bbox"][3] for x in fns]
    fn_ratios = [x["bbox"][2] / max(1e-6, x["bbox"][3]) for x in fns]
    fn_centers = [center_of_bbox(x["bbox"]) for x in fns]

    fp_scores = [float(x.get("score", 0.0)) for x in fps]
    fp_centers = [center_of_bbox(x["bbox"]) for x in fps]

    plot_hist(fn_areas, "FN Area Distribution", args.output_dir / "fn_area_hist.png")
    plot_hist(fn_ratios, "FN Aspect Ratio Distribution", args.output_dir / "fn_aspect_hist.png")
    plot_hist(fp_scores, "FP Confidence Distribution", args.output_dir / "fp_conf_hist.png")

    if fn_centers:
        xs, ys = zip(*fn_centers)
        plot_heatmap(list(xs), list(ys), "FN Spatial Heatmap", args.output_dir / "fn_heatmap.png")
    if fp_centers:
        xs, ys = zip(*fp_centers)
        plot_heatmap(list(xs), list(ys), "FP Spatial Heatmap", args.output_dir / "fp_heatmap.png")

    total_fn = len(fns)
    total_gt = len(gt["annotations"])
    small_fn = sum(1 for a in fn_areas if a < 32 * 32)
    edge_fn = 0
    for c in fn_centers:
        x, y = c
        if x < 0.15 * 640 or x > 0.85 * 640 or y < 0.15 * 640 or y > 0.85 * 640:
            edge_fn += 1
    upper_fp = sum(1 for c in fp_centers if c[1] < 0.3 * 640)

    report = {
        "kucuk_pothole_kacirma_orani": (small_fn / total_fn * 100.0) if total_fn else 0.0,
        "kenar_bolgelerindeki_fn_orani": (edge_fn / total_fn * 100.0) if total_fn else 0.0,
        "ust_bolgedeki_fp_orani": (upper_fp / len(fps) * 100.0) if fps else 0.0,
        "totals": {"gt": total_gt, "fn": total_fn, "fp": len(fps)},
        "oneriler": [
            "Küçük nesneler için multi-scale eğitim ve çözünürlük artışı",
            "Copy-paste augmentation oranını 0.2-0.3 bandında tut",
            "Üst %30 ROI için confidence azaltma/spatial filter uygula",
            "Recall hedefi için düşük threshold + dual-threshold postprocess kullan",
        ],
    }

    out = args.output_dir / "error_analysis_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Hata analizi raporu kaydedildi: %s", out)


if __name__ == "__main__":
    main()
