"""ADAS için recall-odaklı post-processing modülü.

Dependencies:
    - numpy

Usage:
    from src.recall_optimized_postprocess import RecallOptimizedPostprocessor
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np


@dataclass
class Detection:
    """Tek bir detection kaydı."""

    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    class_id: int = 0


class RecallOptimizedPostprocessor:
    """Dual-threshold + temporal smoothing + ROI filtreleme uygular."""

    def __init__(
        self,
        high_conf_threshold: float = 0.50,
        low_conf_threshold: float = 0.20,
        roi_y_min_ratio: float = 0.30,
        temporal_buffer_size: int = 5,
        temporal_iou_threshold: float = 0.4,
    ) -> None:
        """Postprocess sınıfını başlatır."""
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.roi_y_min_ratio = roi_y_min_ratio
        self.temporal_iou_threshold = temporal_iou_threshold
        self.history: Deque[List[Detection]] = deque(maxlen=temporal_buffer_size)

    @staticmethod
    def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        """[x1,y1,x2,y2] için IoU hesaplar."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _apply_spatial_filter(self, det: Detection, image_h: int) -> Detection:
        """Üst %30 bölge için confidence azaltımı uygular."""
        x1, y1, x2, y2 = det.bbox_xyxy
        cy = (y1 + y2) / 2.0
        if cy < self.roi_y_min_ratio * image_h:
            # Recall korunurken üst bölgedeki FP'leri baskılamak için score azaltılır.
            return Detection(det.bbox_xyxy, det.score * 0.5, det.class_id)
        return det

    def _temporal_boost(self, det: Detection) -> float:
        """Önceki frame geçmişine göre confidence boost katsayısı döndürür."""
        matches = 0
        for frame_dets in self.history:
            if any(self._iou_xyxy(det.bbox_xyxy, prev.bbox_xyxy) >= self.temporal_iou_threshold for prev in frame_dets):
                matches += 1

        if matches >= 2:
            return 1.15
        if matches == 1:
            return 1.08
        return 1.0

    def process(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> List[Dict]:
        """Tek frame detection çıktısını işler.

        Args:
            boxes_xyxy: Nx4 bbox array
            scores: N confidence array
            class_ids: N class id array
            image_shape: (height, width)

        Returns:
            JSON-serileştirilebilir detection listesi.
        """
        image_h, _ = image_shape
        candidates: List[Detection] = []

        for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
            base = Detection(tuple(float(v) for v in box.tolist()), float(score), int(class_id))
            spatial = self._apply_spatial_filter(base, image_h)

            if spatial.score < self.low_conf_threshold:
                continue

            boosted_score = min(1.0, spatial.score * self._temporal_boost(spatial))
            candidates.append(Detection(spatial.bbox_xyxy, boosted_score, spatial.class_id))

        output = []
        for det in candidates:
            severity = "confirmed" if det.score >= self.high_conf_threshold else "warning"
            output.append(
                {
                    "bbox": [det.bbox_xyxy[0], det.bbox_xyxy[1], det.bbox_xyxy[2], det.bbox_xyxy[3]],
                    "score": det.score,
                    "class_id": det.class_id,
                    "severity": severity,
                }
            )

        self.history.append(candidates)
        return output

    def reset(self) -> None:
        """Video akışı bittiğinde geçmiş buffer'ını temizler."""
        self.history.clear()
