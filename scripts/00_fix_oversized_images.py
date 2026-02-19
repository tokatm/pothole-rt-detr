#!/usr/bin/env python3
"""Oversized goruntuleri guvenli boyuta indirir.

Dependencies:
    - Pillow

Usage:
    python scripts/00_fix_oversized_images.py \
      --dataset-root /content/dataset/combined_dataset \
      --warn-pixels 120000000 \
      --target-max-pixels 80000000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Bu dataset guvenilir kabul edilerek Pillow guard devre disi birakilir.
Image.MAX_IMAGE_PIXELS = None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """CLI argumanlarini parse eder."""
    parser = argparse.ArgumentParser(description="Fix oversized images before COCO conversion")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--warn-pixels", type=int, default=120_000_000)
    parser.add_argument("--target-max-pixels", type=int, default=80_000_000)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Verilmezse <dataset-root>/backup_oversized_images kullanilir.",
    )
    return parser.parse_args()


def iter_images(images_root: Path) -> Iterable[Path]:
    """images/* altindaki tum goruntuleri gezer."""
    for split_dir in sorted([p for p in images_root.iterdir() if p.is_dir()]):
        for image_path in sorted(split_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield image_path


def compute_resized_shape(width: int, height: int, target_max_pixels: int) -> Tuple[int, int]:
    """Hedef piksel sinirina gore yeni (w, h) hesaplar."""
    pixels = width * height
    if pixels <= target_max_pixels:
        return width, height
    scale = (target_max_pixels / float(pixels)) ** 0.5
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return new_w, new_h


def backup_original(image_path: Path, dataset_root: Path, backup_root: Path) -> None:
    """Orijinal dosyayi backup klasorune kopyalar."""
    relative = image_path.relative_to(dataset_root)
    backup_path = backup_root / relative
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        backup_path.write_bytes(image_path.read_bytes())


def save_resized(image_path: Path, image: Image.Image) -> None:
    """Resized goruntuyu orijinal dosyanin ustune yazar."""
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(image_path, quality=95)
    else:
        image.save(image_path)


def fix_oversized_images(
    dataset_root: Path,
    warn_pixels: int,
    target_max_pixels: int,
    backup_dir: Path | None,
) -> int:
    """Oversized goruntuleri downscale eder ve degisen dosya sayisini dondurur."""
    images_root = dataset_root / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"images klasoru bulunamadi: {images_root}")

    backup_root = backup_dir or (dataset_root / "backup_oversized_images")
    changed = 0

    for image_path in iter_images(images_root):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                pixels = width * height
                if pixels <= warn_pixels:
                    continue
                new_w, new_h = compute_resized_shape(width, height, target_max_pixels)
                if (new_w, new_h) == (width, height):
                    continue
                backup_original(image_path, dataset_root, backup_root)
                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                save_resized(image_path, resized)
                changed += 1
                LOGGER.info(
                    "Downscale: %s | %dx%d -> %dx%d",
                    image_path,
                    width,
                    height,
                    new_w,
                    new_h,
                )
        except Exception as exc:
            LOGGER.warning("Gorsel islenemedi, atlandi: %s (%s)", image_path, exc)

    LOGGER.info("Oversized fix tamamlandi | degisen dosya sayisi: %d", changed)
    LOGGER.info("Backup dizini: %s", backup_root)
    return changed


def main() -> None:
    """Program giris noktasi."""
    args = parse_args()
    fix_oversized_images(
        dataset_root=args.dataset_root,
        warn_pixels=args.warn_pixels,
        target_max_pixels=args.target_max_pixels,
        backup_dir=args.backup_dir,
    )


if __name__ == "__main__":
    main()

