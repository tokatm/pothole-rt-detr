#!/usr/bin/env python3
"""Küçük pothole'lar için offline copy-paste augmentation.

RT-DETR'nin transform pipeline'ı CopyPaste'i native desteklemediğinden,
bu script eğitim öncesi çalıştırılarak dataset'e augmented örnekler ekler.
Küçük pothole bölgeleri (patch) kesip rastgele yol görselleri üzerine yapıştırır.

Amaç: Dataset'teki ~%60 küçük pothole problemiyle mücadele.
      Küçük nesne çeşitliliğini artırarak recall iyileştirme.

Dependencies:
    - opencv-python
    - numpy
    - Pillow

Usage:
    # Eğitim öncesi çalıştır — COCO dönüşümünden SONRA, eğitimden ÖNCE
    python scripts/00_copypaste_augment.py \
      --dataset-root /content/dataset/combined_dataset \
      --output-dir /content/dataset/combined_dataset \
      --paste-ratio 0.25 \
      --num-augmented 500
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI argümanlarını parse eder."""
    parser = argparse.ArgumentParser(
        description="Offline copy-paste augmentation for small potholes"
    )
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="combined_dataset kök dizini")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Augmented görsellerin kaydedileceği yer (images/train ve labels/train)")
    parser.add_argument("--paste-ratio", type=float, default=0.25,
                        help="Copy-paste oranı (0.2-0.3 arası önerilen)")
    parser.add_argument("--num-augmented", type=int, default=500,
                        help="Oluşturulacak augmented görsel sayısı")
    parser.add_argument("--max-paste-per-image", type=int, default=3,
                        help="Bir görsele yapıştırılacak max pothole sayısı")
    parser.add_argument("--small-threshold", type=float, default=10000.0,
                        help="Küçük pothole alan eşiği (pixel²)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_yolo_labels(label_path: Path) -> List[List[float]]:
    """YOLO format label dosyasını okur. Her satır: [class, x_c, y_c, w, h]."""
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append([float(p) for p in parts])
    return labels


def save_yolo_labels(label_path: Path, labels: List[List[float]]) -> None:
    """YOLO format label dosyasını kaydeder."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for lbl in labels:
        cls = int(lbl[0])
        coords = " ".join(f"{v:.6f}" for v in lbl[1:])
        lines.append(f"{cls} {coords}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_pothole_patches(
    image_dir: Path,
    label_dir: Path,
    small_threshold: float,
) -> List[Dict]:
    """Küçük pothole patch'lerini toplar.

    Returns:
        List of dicts: {"image_path", "patch_bbox_px" (x1,y1,x2,y2), "area"}
    """
    patches = []
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        labels = load_yolo_labels(label_path)
        if not labels:
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue

        for lbl in labels:
            _, x_c, y_c, w, h = lbl
            # Normalized → pixel
            pw = w * img_w
            ph = h * img_h
            area = pw * ph

            if area < small_threshold:
                x1 = max(0, int((x_c - w / 2) * img_w))
                y1 = max(0, int((y_c - h / 2) * img_h))
                x2 = min(img_w, int((x_c + w / 2) * img_w))
                y2 = min(img_h, int((y_c + h / 2) * img_h))

                if (x2 - x1) > 4 and (y2 - y1) > 4:
                    patches.append({
                        "image_path": img_path,
                        "bbox_px": (x1, y1, x2, y2),
                        "area": area,
                    })

    LOGGER.info("Toplam %d küçük pothole patch toplandı (alan < %.0f px²)", len(patches), small_threshold)
    return patches


def paste_patch_on_image(
    target_img: np.ndarray,
    patch_img: np.ndarray,
    target_region: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Patch'i hedef görsele yapıştırır. Alpha blending ile kenar yumuşatma uygular.

    Args:
        target_img: Hedef görsel (BGR)
        patch_img: Pothole patch'i (BGR)
        target_region: Yapıştırma bölgesi oranları (y_min_ratio, y_max_ratio, x_min_ratio, x_max_ratio)

    Returns:
        (modified_image, (x1, y1, x2, y2) pixel coordinates)
    """
    th, tw = target_img.shape[:2]
    ph, pw = patch_img.shape[:2]

    y_min_ratio, y_max_ratio, x_min_ratio, x_max_ratio = target_region

    # Pothole'lar genelde görselin merkez-alt bölgesinde
    # y: 0.3-0.8, x: 0.1-0.9 arasında rastgele konum seç
    max_y = int(y_max_ratio * th) - ph
    min_y = int(y_min_ratio * th)
    max_x = int(x_max_ratio * tw) - pw
    min_x = int(x_min_ratio * tw)

    if max_y <= min_y or max_x <= min_x:
        return target_img, (0, 0, 0, 0)

    paste_y = random.randint(min_y, max_y)
    paste_x = random.randint(min_x, max_x)

    # Alpha blending — kenarları yumuşat (doğal görünsün)
    result = target_img.copy()
    # Basit alpha mask: merkez opak, kenarlar şeffaf
    alpha = np.ones((ph, pw), dtype=np.float32)
    border = max(2, min(ph, pw) // 6)
    for i in range(border):
        factor = (i + 1) / border
        alpha[i, :] *= factor
        alpha[ph - 1 - i, :] *= factor
        alpha[:, i] *= factor
        alpha[:, pw - 1 - i] *= factor
    alpha = alpha[:, :, np.newaxis]

    roi = result[paste_y:paste_y + ph, paste_x:paste_x + pw]
    blended = (patch_img.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    result[paste_y:paste_y + ph, paste_x:paste_x + pw] = blended

    return result, (paste_x, paste_y, paste_x + pw, paste_y + ph)


def augment_dataset(
    dataset_root: Path,
    output_dir: Path,
    patches: List[Dict],
    num_augmented: int,
    max_paste_per_image: int,
    paste_ratio: float,
) -> int:
    """Copy-paste ile augmented görseller oluşturur.

    Returns:
        Oluşturulan augmented görsel sayısı.
    """
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    out_image_dir = output_dir / "images" / "train"
    out_label_dir = output_dir / "labels" / "train"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    # Hedef görselleri topla (üzerine yapıştıracağız)
    target_images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not patches or not target_images:
        LOGGER.warning("Yeterli patch veya hedef görsel yok.")
        return 0

    created = 0
    for i in range(num_augmented):
        # Rastgele bir hedef görsel seç
        target_path = random.choice(target_images)
        target_img = cv2.imread(str(target_path))
        if target_img is None:
            continue

        th, tw = target_img.shape[:2]

        # Mevcut label'ları oku
        existing_label_path = label_dir / f"{target_path.stem}.txt"
        existing_labels = load_yolo_labels(existing_label_path)

        # Kaç patch yapıştıracağımızı belirle
        num_paste = random.randint(1, max_paste_per_image)
        if random.random() > paste_ratio:
            num_paste = 1  # Düşük oran: genelde 1 patch

        new_labels = list(existing_labels)
        result_img = target_img.copy()

        for _ in range(num_paste):
            patch_info = random.choice(patches)
            src_img = cv2.imread(str(patch_info["image_path"]))
            if src_img is None:
                continue

            x1, y1, x2, y2 = patch_info["bbox_px"]
            patch = src_img[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Hafif boyut varyasyonu (0.8x - 1.2x)
            scale = random.uniform(0.8, 1.2)
            new_w = max(4, int(patch.shape[1] * scale))
            new_h = max(4, int(patch.shape[0] * scale))
            patch = cv2.resize(patch, (new_w, new_h))

            # Pothole'ların spatial dağılımına uygun bölgeye yapıştır
            # Dataset analizi: y: 0.3-0.8, x: 0.1-0.9
            result_img, (px1, py1, px2, py2) = paste_patch_on_image(
                result_img, patch, (0.3, 0.85, 0.1, 0.9)
            )

            if px2 > px1 and py2 > py1:
                # YOLO format: class x_center y_center width height (normalized)
                cx = ((px1 + px2) / 2.0) / tw
                cy = ((py1 + py2) / 2.0) / th
                bw = (px2 - px1) / tw
                bh = (py2 - py1) / th
                new_labels.append([0.0, cx, cy, bw, bh])

        # Kaydet
        aug_name = f"aug_cp_{i:05d}_{target_path.stem}{target_path.suffix}"
        cv2.imwrite(str(out_image_dir / aug_name), result_img)
        save_yolo_labels(out_label_dir / f"aug_cp_{i:05d}_{target_path.stem}.txt", new_labels)
        created += 1

    LOGGER.info("Copy-paste augmentation tamamlandı: %d görsel oluşturuldu", created)
    return created


def main() -> None:
    """Program giriş noktası."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    image_dir = args.dataset_root / "images" / "train"
    label_dir = args.dataset_root / "labels" / "train"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Dataset dizinleri bulunamadı: {image_dir}, {label_dir}")

    # 1. Küçük pothole patch'lerini topla
    patches = extract_pothole_patches(image_dir, label_dir, args.small_threshold)

    if not patches:
        LOGGER.warning("Küçük pothole patch bulunamadı. Augmentation atlanıyor.")
        return

    # 2. Augmented görseller oluştur
    created = augment_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        patches=patches,
        num_augmented=args.num_augmented,
        max_paste_per_image=args.max_paste_per_image,
        paste_ratio=args.paste_ratio,
    )

    LOGGER.info(
        "Toplam %d augmented görsel oluşturuldu. "
        "DİKKAT: COCO annotation dönüşümünü (01_convert_yolo_to_coco.py) "
        "tekrar çalıştırmanız gerekir.",
        created,
    )


if __name__ == "__main__":
    main()
