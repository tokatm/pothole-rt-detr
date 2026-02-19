#!/usr/bin/env python3
"""FN/FP hata analizi üretir, görselleri kaydeder ve recall iyileştirme önerileri çıkarır.

Dependencies:
    - numpy
    - matplotlib
    - opencv-python
    - Pillow

Usage:
    python scripts/07_analyze_errors.py \
      --gt-json /content/dataset/combined_dataset/annotations/val_annotations.json \
      --pred-json /content/outputs/predictions_val.json \
      --image-dir /content/dataset/combined_dataset/images/val \
      --output-dir /content/outputs/error_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
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
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Val görsellerin bulunduğu klasör. Verilirse FN/FP görselleri kaydedilir.",
    )
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument(
        "--max-save-images",
        type=int,
        default=50,
        help="Kaydedilecek maksimum FN/FP görsel sayısı.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    """JSON dosyasını okur."""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def iou(a: List[float], b: List[float]) -> float:
    """[x,y,w,h] bbox IoU hesaplar."""
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


def match(
    gt_by_img: Dict[int, List[Dict]], pred_by_img: Dict[int, List[Dict]], conf_thres: float
) -> Tuple[List[Dict], List[Dict]]:
    """GT ve pred eşleştirip FN/FP listesi döndürür."""
    fns: List[Dict] = []
    fps: List[Dict] = []

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
    """BBox merkezini döndürür (x,y,w,h formatı)."""
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def plot_hist(values: List[float], title: str, xlabel: str, out_path: Path) -> None:
    """Histogram çizip kaydeder."""
    if not values:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frekans")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_heatmap(xs: List[float], ys: List[float], title: str, out_path: Path) -> None:
    """2D spatial yoğunluk haritası üretir."""
    if not xs:
        return
    plt.figure(figsize=(7, 6))
    plt.hist2d(xs, ys, bins=40, cmap="YlOrRd")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("X (piksel)")
    plt.ylabel("Y (piksel)")
    plt.colorbar(label="Yoğunluk")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def draw_bbox_on_image(
    image: np.ndarray,
    bbox_xywh: List[float],
    color: Tuple[int, int, int],
    label: str,
) -> np.ndarray:
    """Görsel üzerine bbox ve label çizer."""
    out = image.copy()
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def save_error_images(
    errors: List[Dict],
    error_type: str,
    image_dir: Path,
    output_dir: Path,
    image_id_to_filename: Dict[int, str],
    max_save: int,
) -> int:
    """FN veya FP görsellerini bbox overlay ile kaydeder.

    Returns:
        Kaydedilen görsel sayısı.
    """
    save_dir = output_dir / f"{error_type}_images"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Görselleri image_id'ye göre grupla
    by_img: Dict[int, List[Dict]] = defaultdict(list)
    for err in errors:
        by_img[err["image_id"]].append(err)

    saved = 0
    color = (0, 0, 255) if error_type == "fn" else (255, 0, 0)  # FN=kırmızı, FP=mavi
    label_prefix = "MISSED" if error_type == "fn" else "FALSE"

    for img_id, errs in sorted(by_img.items()):
        if saved >= max_save:
            break
        filename = image_id_to_filename.get(img_id)
        if filename is None:
            continue
        img_path = image_dir / filename
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        for err in errs:
            bbox = err.get("bbox", [0, 0, 0, 0])
            score = err.get("score", 0.0)
            area = bbox[2] * bbox[3]
            label = f"{label_prefix} a={area:.0f}"
            if error_type == "fp":
                label = f"{label_prefix} s={score:.2f}"
            image = draw_bbox_on_image(image, bbox, color, label)

        out_path = save_dir / f"{error_type}_{img_id:06d}_{filename}"
        cv2.imwrite(str(out_path), image)
        saved += 1

    LOGGER.info("%s görseli kaydedildi: %d adet -> %s", error_type.upper(), saved, save_dir)
    return saved


def main() -> None:
    """Program giriş noktası."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_json(args.gt_json)
    preds = load_json(args.pred_json)

    # image_id → file_name mapping oluştur (görsel kaydetme için)
    image_id_to_filename: Dict[int, str] = {}
    for img_info in gt.get("images", []):
        image_id_to_filename[img_info["id"]] = img_info["file_name"]

    gt_by_img: Dict[int, List[Dict]] = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_img[ann["image_id"]].append(ann)

    pred_by_img: Dict[int, List[Dict]] = defaultdict(list)
    for p in preds:
        pred_by_img[p["image_id"]].append(p)

    fns, fps = match(gt_by_img, pred_by_img, args.conf_thres)

    # --- İstatistik Hesaplama ---
    fn_areas = [x["bbox"][2] * x["bbox"][3] for x in fns]
    fn_ratios = [x["bbox"][2] / max(1e-6, x["bbox"][3]) for x in fns]
    fn_centers = [center_of_bbox(x["bbox"]) for x in fns]

    fp_scores = [float(x.get("score", 0.0)) for x in fps]
    fp_centers = [center_of_bbox(x["bbox"]) for x in fps]

    # --- Grafikler ---
    plot_hist(fn_areas, "FN Area Distribution (Kaçırılan Pothole Boyutları)", "Alan (pixel²)", args.output_dir / "fn_area_hist.png")
    plot_hist(fn_ratios, "FN Aspect Ratio Distribution", "W/H Oranı", args.output_dir / "fn_aspect_hist.png")
    plot_hist(fp_scores, "FP Confidence Distribution (Yanlış Alarm Güvenleri)", "Confidence", args.output_dir / "fp_conf_hist.png")

    if fn_centers:
        xs, ys = zip(*fn_centers)
        plot_heatmap(list(xs), list(ys), "FN Spatial Heatmap (Kaçırılan Pothole Konumları)", args.output_dir / "fn_heatmap.png")
    if fp_centers:
        xs, ys = zip(*fp_centers)
        plot_heatmap(list(xs), list(ys), "FP Spatial Heatmap (Yanlış Alarm Konumları)", args.output_dir / "fp_heatmap.png")

    # --- FN/FP Görsel Kaydetme ---
    if args.image_dir is not None and args.image_dir.exists():
        save_error_images(fns, "fn", args.image_dir, args.output_dir, image_id_to_filename, args.max_save_images)
        save_error_images(fps, "fp", args.image_dir, args.output_dir, image_id_to_filename, args.max_save_images)
    else:
        LOGGER.info("--image-dir verilmedi veya mevcut değil, FN/FP görselleri kaydedilmedi.")

    # --- Rapor ---
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
        "kucuk_pothole_kacirma_orani": f"{(small_fn / total_fn * 100.0) if total_fn else 0.0:.1f}%",
        "kenar_bolgelerindeki_fn_orani": f"{(edge_fn / total_fn * 100.0) if total_fn else 0.0:.1f}%",
        "ust_bolgedeki_fp_orani": f"{(upper_fp / len(fps) * 100.0) if fps else 0.0:.1f}%",
        "totals": {"gt": total_gt, "fn": total_fn, "fp": len(fps)},
        "fn_boyut_dagilimi": {
            "small_lt_32sq": sum(1 for a in fn_areas if a < 32 * 32),
            "medium_32sq_96sq": sum(1 for a in fn_areas if 32 * 32 <= a <= 96 * 96),
            "large_gt_96sq": sum(1 for a in fn_areas if a > 96 * 96),
        },
        "oneriler": [
            "Küçük nesneler için multi-scale eğitim ve çözünürlük artışı",
            "Copy-paste augmentation oranını 0.2-0.3 bandında tut",
            "Üst %30 ROI için confidence azaltma/spatial filter uygula",
            "Recall hedefi için düşük threshold + dual-threshold postprocess kullan",
            "EMA kullanarak model stability artır",
            "Warmup scheduler ile ilk epoch'larda LR'yi kademeli artır",
        ],
    }

    out = args.output_dir / "error_analysis_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Hata analizi raporu kaydedildi: %s", out)


if __name__ == "__main__":
    main()
