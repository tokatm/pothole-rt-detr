#!/usr/bin/env python3
"""Train split icin offline mixup augmentation uretir (YOLO format).

Dependencies:
    - Pillow
    - numpy

Usage:
    python scripts/00_mixup_augment.py \
      --dataset-root /content/dataset/combined_dataset \
      --ratio 0.2 \
      --alpha 0.4
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp"]


def parse_args() -> argparse.Namespace:
    """CLI argumanlarini parse eder."""
    parser = argparse.ArgumentParser(description="Offline mixup for YOLO train split")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.2, help="Train image sayisina gore mixup oranı")
    parser.add_argument("--alpha", type=float, default=0.4, help="Beta(alpha, alpha) parametresi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=1000)
    return parser.parse_args()


def find_image_path(image_dir: Path, stem: str) -> Path | None:
    """Stem icin mevcut image dosyasini bulur."""
    for ext in IMAGE_SUFFIXES + [e.upper() for e in IMAGE_SUFFIXES]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_train_pairs(dataset_root: Path) -> List[Tuple[Path, Path]]:
    """Etiketli train sample listesi dondurur."""
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("images/train veya labels/train bulunamadi.")

    pairs: List[Tuple[Path, Path]] = []
    for lf in sorted(label_dir.glob("*.txt")):
        img = find_image_path(image_dir, lf.stem)
        if img is None:
            continue
        lines = lf.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue
        pairs.append((img, lf))
    return pairs


def blend_images(base_img: Image.Image, mix_img: Image.Image, lam: float) -> Image.Image:
    """Iki goruntuyu aynı boyuta getirip mixup uygular."""
    mix_resized = mix_img.resize(base_img.size, Image.Resampling.BILINEAR)
    a = np.asarray(base_img.convert("RGB"), dtype=np.float32)
    b = np.asarray(mix_resized.convert("RGB"), dtype=np.float32)
    out = lam * a + (1.0 - lam) * b
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def merge_yolo_labels(label_a: Path, label_b: Path) -> Sequence[str]:
    """Iki YOLO label dosyasini birlestirir."""
    lines_a = [x.strip() for x in label_a.read_text(encoding="utf-8").splitlines() if x.strip()]
    lines_b = [x.strip() for x in label_b.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines_a + lines_b


def run_mixup(dataset_root: Path, ratio: float, alpha: float, seed: int, max_samples: int) -> int:
    """Offline mixup sample'lari uretir ve olusan sayiyi dondurur."""
    random.seed(seed)
    np.random.seed(seed)

    pairs = load_train_pairs(dataset_root)
    if len(pairs) < 2:
        LOGGER.warning("Mixup icin yeterli train sample yok.")
        return 0

    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"

    num_to_generate = min(max_samples, int(len(pairs) * ratio))
    generated = 0

    for i in range(num_to_generate):
        (img_a, lbl_a), (img_b, lbl_b) = random.sample(pairs, 2)
        lam = float(np.random.beta(alpha, alpha))

        with Image.open(img_a) as a, Image.open(img_b) as b:
            mixed = blend_images(a, b, lam)

        stem = f"mixup_{img_a.stem}_{img_b.stem}_{i:05d}"
        out_img = image_dir / f"{stem}.jpg"
        out_lbl = label_dir / f"{stem}.txt"

        if out_img.exists() or out_lbl.exists():
            continue

        mixed.save(out_img, quality=95)
        out_lbl.write_text("\n".join(merge_yolo_labels(lbl_a, lbl_b)) + "\n", encoding="utf-8")
        generated += 1

    LOGGER.info("Mixup augmentation tamamlandi | uretilen sample: %d", generated)
    return generated


def main() -> None:
    """Program giris noktasi."""
    args = parse_args()
    run_mixup(
        dataset_root=args.dataset_root,
        ratio=args.ratio,
        alpha=args.alpha,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

