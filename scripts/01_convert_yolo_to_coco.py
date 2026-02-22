#!/usr/bin/env python3
"""YOLO etiketlerini COCO JSON formatina donusturur.

Dependencies:
    - Pillow

Usage:
    python scripts/01_convert_yolo_to_coco.py \
        --dataset-root /content/dataset/combined_dataset \
        --output-dir /content/dataset/combined_dataset/annotations
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class CocoImage:
    """COCO image kaydi."""

    id: int
    file_name: str
    width: int
    height: int


@dataclass
class CocoAnnotation:
    """COCO annotation kaydi."""

    id: int
    image_id: int
    category_id: int
    bbox: List[float]
    area: float
    iscrowd: int = 0


def parse_args() -> argparse.Namespace:
    """CLI argumanlarini parse eder."""
    parser = argparse.ArgumentParser(description="YOLO to COCO converter")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fix-oversized-before-convert",
        dest="fix_oversized_before_convert",
        action="store_true",
        default=True,
        help="JSON donusumunden once oversized image fix scriptini calistir.",
    )
    parser.add_argument(
        "--no-fix-oversized-before-convert",
        dest="fix_oversized_before_convert",
        action="store_false",
        help="Oversized image fix adimini kapat.",
    )
    parser.add_argument("--oversized-warn-pixels", type=int, default=120_000_000)
    parser.add_argument("--oversized-target-max-pixels", type=int, default=80_000_000)
    parser.add_argument(
        "--mixup-before-convert",
        action="store_true",
        help="JSON donusumunden once offline mixup augmentation uret.",
    )
    parser.add_argument("--mixup-ratio", type=float, default=0.20)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--mixup-seed", type=int, default=42)
    parser.add_argument("--mixup-max-samples", type=int, default=1000)
    return parser.parse_args()


def yolo_to_coco_bbox(
    x_center: float, y_center: float, width: float, height: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Normalize YOLO bbox'i COCO pixel bbox formatina cevirir."""
    w_px = width * img_w
    h_px = height * img_h
    x_min = (x_center * img_w) - (w_px / 2)
    y_min = (y_center * img_h) - (h_px / 2)
    return max(0.0, x_min), max(0.0, y_min), max(0.0, w_px), max(0.0, h_px)


def convert_split(dataset_root: Path, split: str) -> Dict[str, List[Dict]]:
    """Belirli split'i YOLO formatindan COCO dict'e donusturur."""
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Eksik split dizini: {image_dir} veya {label_dir}")

    image_records: List[Dict] = []
    annotation_records: List[Dict] = []

    image_id = 1
    annotation_id = 1

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as exc:
            LOGGER.warning("Gorsel okunamadi, atlandi: %s (%s)", image_path, exc)
            continue

        image_records.append(
            CocoImage(id=image_id, file_name=image_path.name, width=width, height=height).__dict__
        )

        if not label_path.exists():
            LOGGER.warning("Label dosyasi yok, bos kabul edildi: %s", label_path)
            image_id += 1
            continue

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            LOGGER.warning("Bos label dosyasi, atlandi: %s", label_path)
            image_id += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                LOGGER.warning("Hatali satir formati, atlandi: %s -> '%s'", label_path, line)
                continue
            try:
                cls, x_c, y_c, w, h = map(float, parts)
                if int(cls) != 0:
                    LOGGER.warning("Beklenmeyen class id (%s), atlandi: %s", int(cls), line)
                    continue
                coco_bbox = yolo_to_coco_bbox(x_c, y_c, w, h, width, height)
                area = coco_bbox[2] * coco_bbox[3]
                annotation_records.append(
                    CocoAnnotation(
                        id=annotation_id,
                        image_id=image_id,
                        category_id=1,
                        bbox=[round(v, 3) for v in coco_bbox],
                        area=round(area, 3),
                    ).__dict__
                )
                annotation_id += 1
            except Exception as exc:
                LOGGER.warning("Label parse hatasi, atlandi: %s (%s)", line, exc)
                continue

        image_id += 1

    coco = {
        "images": image_records,
        "annotations": annotation_records,
        "categories": [{"id": 1, "name": "pothole"}],
    }
    return coco


def main() -> None:
    """Program giris noktasi."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.fix_oversized_before_convert:
        fix_script = Path(__file__).with_name("00_fix_oversized_images.py")
        cmd = [
            sys.executable,
            str(fix_script),
            "--dataset-root",
            str(args.dataset_root),
            "--warn-pixels",
            str(args.oversized_warn_pixels),
            "--target-max-pixels",
            str(args.oversized_target_max_pixels),
        ]
        LOGGER.info("Oversized pre-step calistiriliyor: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if args.mixup_before_convert:
        mixup_script = Path(__file__).with_name("00_mixup_augment.py")
        cmd = [
            sys.executable,
            str(mixup_script),
            "--dataset-root",
            str(args.dataset_root),
            "--ratio",
            str(args.mixup_ratio),
            "--alpha",
            str(args.mixup_alpha),
            "--seed",
            str(args.mixup_seed),
            "--max-samples",
            str(args.mixup_max_samples),
        ]
        LOGGER.info("Mixup pre-step calistiriliyor: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    for split in ("train", "val"):
        coco = convert_split(args.dataset_root, split)
        out_path = args.output_dir / f"{split}_annotations.json"
        out_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info(
            "%s donusumu tamamlandi | images=%d annotations=%d -> %s",
            split,
            len(coco["images"]),
            len(coco["annotations"]),
            out_path,
        )


if __name__ == "__main__":
    main()
