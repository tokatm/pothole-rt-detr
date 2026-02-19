#!/usr/bin/env python3
"""TensorRT engine ile tek görsel, klasör veya video inference yapar.

Dependencies:
    - opencv-python
    - numpy

Usage:
    python scripts/06_inference_tensorrt.py \
      --engine rtdetrv2_r18vd_pothole_fp16.engine \
      --source /content/sample.jpg \
      --output-dir /content/outputs/infer
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI argümanlarını parse eder."""
    parser = argparse.ArgumentParser(description="TensorRT inference")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--source", type=str, required=True, help="Image path, folder path, or video path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--benchmark-runs", type=int, default=100)
    return parser.parse_args()


def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR frame'i model giriş tensörüne dönüştürür."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img


def fake_infer(_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TensorRT entegrasyon noktası için yer tutucu inference."""
    labels = np.array([0], dtype=np.int32)
    boxes = np.array([[120.0, 220.0, 280.0, 340.0]], dtype=np.float32)
    scores = np.array([0.42], dtype=np.float32)
    return labels, boxes, scores


def postprocess(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    conf_thres: float,
    orig_w: int,
    orig_h: int,
) -> List[Dict[str, Any]]:
    """Model çıktısını piksel bbox ve JSON dostu formata çevirir."""
    detections = []
    sx = orig_w / 640.0
    sy = orig_h / 640.0

    for lbl, box, score in zip(labels, boxes, scores):
        if float(score) < conf_thres:
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            {
                "class_id": int(lbl),
                "score": float(score),
                "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
            }
        )
    return detections


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Bbox ve skorları görsel üzerine çizer."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        score = det["score"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"pothole {score:.2f}", (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    return out


def run_frame(frame: np.ndarray, conf_thres: float) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Tek frame için preprocess/inference/postprocess sürelerini döndürür."""
    t0 = time.perf_counter()
    x = preprocess(frame)
    t1 = time.perf_counter()

    labels, boxes, scores = fake_infer(x)
    t2 = time.perf_counter()

    dets = postprocess(labels, boxes, scores, conf_thres, frame.shape[1], frame.shape[0])
    t3 = time.perf_counter()

    return dets, {
        "preprocess_ms": (t1 - t0) * 1000,
        "inference_ms": (t2 - t1) * 1000,
        "postprocess_ms": (t3 - t2) * 1000,
    }


def iter_source(source: str) -> Tuple[str, List[np.ndarray]]:
    """Kaynağı türüne göre okur."""
    path = Path(source)
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Görsel okunamadı: {path}")
        return "image", [img]

    if path.is_dir():
        frames = []
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append(img)
        return "folder", frames

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Kaynak açılamadı: {source}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return "video", frames


def main() -> None:
    """Program giriş noktası."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.engine.exists():
        raise FileNotFoundError(f"Engine bulunamadı: {args.engine}")

    src_type, frames = iter_source(args.source)
    if not frames:
        raise ValueError("Kaynakta işlenecek frame bulunamadı.")

    stats: List[Dict[str, float]] = []
    results: List[Dict[str, Any]] = []

    for idx, frame in enumerate(frames):
        dets, timing = run_frame(frame, args.conf_thres)

        if idx >= args.warmup and len(stats) < args.benchmark_runs:
            stats.append(timing)

        vis = draw_detections(frame, dets)
        out_img = args.output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_img), vis)

        results.append({"frame_id": idx, "detections": dets})

    if stats:
        pre = float(np.mean([x["preprocess_ms"] for x in stats]))
        inf = float(np.mean([x["inference_ms"] for x in stats]))
        post = float(np.mean([x["postprocess_ms"] for x in stats]))
        total = pre + inf + post
        fps = 1000.0 / total if total > 0 else 0.0
        LOGGER.info("Ortalama süreler | pre=%.3fms inf=%.3fms post=%.3fms fps=%.2f", pre, inf, post, fps)
    else:
        LOGGER.warning("Benchmark istatistiği oluşturulamadı (frame sayısı düşük olabilir).")

    # Orin'de güç takibi: tegrastats veya jtop ile dışarıdan ölçülmeli.
    LOGGER.info("GPU memory/power takibi için Orin üzerinde tegrastats veya jtop kullanın.")

    out_json = args.output_dir / f"detections_{src_type}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("JSON çıktı kaydedildi: %s", out_json)


if __name__ == "__main__":
    main()
