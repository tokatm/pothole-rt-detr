#!/usr/bin/env python3
"""Kucuk bbox iceren train ornekleri icin offline upscale sample uretir.

Amac:
    - Kucuk pothole (< min-side-px) iceren ornekleri buyuk olcekte tekrar
      uretmek (800-1024 gibi), boylece buyuk olcek etkisini sadece bu gruba
      uygulamak.
    - Istenirse orijinal kucuk ornekleri conversion asamasinda dislamak icin
      stem listesi yazmak.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from PIL import Image


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upscale small-object train samples (YOLO format)")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--min-side-px", type=float, default=32.0)
    parser.add_argument("--target-short-sides", type=str, default="800,832,896,960,1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-original-small",
        action="store_true",
        help="Orijinal kucuk-ornek stem listesini yazar (converter bu stemleri atlayabilir).",
    )
    return parser.parse_args()


def parse_target_short_sides(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("target-short-sides bos olamaz")
    return vals


def find_image_by_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_SUFFIXES + [e.upper() for e in IMAGE_SUFFIXES]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_yolo_boxes(label_path: Path) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x, y, w, h = map(float, parts)
        out.append((x, y, w, h))
    return out


def has_small_bbox(boxes: Iterable[Tuple[float, float, float, float]], w: int, h: int, min_side: float) -> bool:
    for _, _, bw, bh in boxes:
        if min(bw * w, bh * h) < min_side:
            return True
    return False


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    target_short_sides = parse_target_short_sides(args.target_short_sides)

    img_dir = args.dataset_root / "images" / "train"
    lbl_dir = args.dataset_root / "labels" / "train"
    ann_dir = args.dataset_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    skip_stems_file = ann_dir / "small_object_original_stems.txt"

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError("images/train veya labels/train bulunamadi.")

    generated = 0
    small_original_stems: Set[str] = set()

    for lbl in sorted(lbl_dir.glob("*.txt")):
        img_path = find_image_by_stem(img_dir, lbl.stem)
        if img_path is None:
            continue

        boxes = read_yolo_boxes(lbl)
        if not boxes:
            continue

        with Image.open(img_path) as img:
            w, h = img.size
            if not has_small_bbox(boxes, w, h, args.min_side_px):
                continue

            small_original_stems.add(lbl.stem)
            short_side = min(w, h)
            target_short = random.choice(target_short_sides)
            # Mevcutten kucuge dusurme yok.
            target_short = max(target_short, short_side)
            scale = target_short / float(short_side)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))

            out_stem = f"smallup_{lbl.stem}"
            out_img = img_dir / f"{out_stem}.jpg"
            out_lbl = lbl_dir / f"{out_stem}.txt"

            if out_img.exists() and out_lbl.exists():
                continue

            resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            # JPEG, RGBA desteklemez; alpha iceren goruntuleri RGB'ye cevir.
            if resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            resized.save(out_img, quality=95)
            # YOLO label normalize oldugu icin uniform resize'da label satirlari ayni kalir.
            out_lbl.write_text(lbl.read_text(encoding="utf-8"), encoding="utf-8")
            generated += 1

    if args.exclude_original_small:
        skip_stems_file.write_text(
            "\n".join(sorted(small_original_stems)) + ("\n" if small_original_stems else ""),
            encoding="utf-8",
        )
        LOGGER.info("Kucuk-orijinal stem listesi yazildi: %s (%d)", skip_stems_file, len(small_original_stems))
    else:
        # Eski listeyi temiz tut.
        if skip_stems_file.exists():
            skip_stems_file.unlink()

    LOGGER.info("Small-object upscale tamamlandi | uretilen sample: %d", generated)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
