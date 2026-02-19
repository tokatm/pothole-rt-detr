#!/usr/bin/env python3
"""RT-DETRv2 pothole eğitimi başlatır ve yönetir.

Dependencies:
    - torch
    - pyyaml

Usage:
    python scripts/02_train.py \
      --config configs/rtdetrv2/rtdetrv2_r18vd_pothole.yml \
      --rtdetr-root /content/RT-DETR/rtdetrv2_pytorch \
      --use-amp \
      -t /content/weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED_URL = (
    "https://github.com/lyuwenyu/storage/releases/download/v0.2/"
    "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)


def parse_args() -> argparse.Namespace:
    """Komut satırı argümanlarını parse eder."""
    parser = argparse.ArgumentParser(description="RT-DETRv2 training launcher")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rtdetr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/outputs/pothole_rtdetrv2"))
    parser.add_argument(
        "-t",
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained weight path veya URL. Verilmezse varsayılan URL'den indirilir.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="RT-DETR train.py komutuna --use-amp geçirir.",
    )
    parser.add_argument(
        "--no-sync-configs",
        action="store_true",
        help="Projedeki config dosyalarını RT-DETR klasörüne otomatik kopyalamayı kapatır.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=60)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Reproducibility için global seed ayarlar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> Dict[str, Any]:
    """YAML config dosyasını yükler."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config bulunamadı: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _download_weights(url: str, output_path: Path) -> Path:
    """Pretrained weight dosyasını URL'den indirir."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Pretrained weight indiriliyor: %s", url)
    urllib.request.urlretrieve(url, output_path)
    LOGGER.info("Weight indirildi: %s", output_path)
    return output_path


def maybe_validate_checkpoint(path_or_url: str | None, output_dir: Path) -> Path | None:
    """Checkpoint'i dosya yolu/URL olarak çözümler, gerekirse indirir."""
    candidate = path_or_url or DEFAULT_PRETRAINED_URL

    if candidate.startswith(("http://", "https://")):
        filename = Path(candidate).name or "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
        local_path = output_dir / "weights" / filename
        if local_path.exists():
            LOGGER.info("İndirilen weight zaten mevcut: %s", local_path)
            return local_path
        try:
            return _download_weights(candidate, local_path)
        except Exception as exc:
            LOGGER.error("Weight indirilemedi: %s", exc)
            answer = input("Scratch eğitim ile devam edilsin mi? [e/h]: ").strip().lower()
            if answer in {"e", "evet", "y", "yes"}:
                return None
            raise RuntimeError("Kullanıcı scratch eğitimi reddetti.") from exc

    local_path = Path(candidate)
    if local_path.exists():
        return local_path

    LOGGER.error("Pretrained weight yüklenemedi: %s", local_path)
    answer = input("Scratch eğitim ile devam edilsin mi? [e/h]: ").strip().lower()
    if answer in {"e", "evet", "y", "yes"}:
        return None
    raise FileNotFoundError("Kullanıcı scratch eğitimi reddetti.")


def sync_project_configs(rtdetr_root: Path) -> Path:
    """Proje config dosyalarını RT-DETR config ağacına kopyalar."""
    src_dataset = PROJECT_ROOT / "configs" / "dataset" / "pothole_detection.yml"
    src_model = PROJECT_ROOT / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_pothole.yml"
    if not src_dataset.exists() or not src_model.exists():
        raise FileNotFoundError("Proje config dosyaları bulunamadı.")

    dst_dataset = rtdetr_root / "configs" / "dataset" / "pothole_detection.yml"
    dst_model = rtdetr_root / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_pothole.yml"
    dst_dataset.parent.mkdir(parents=True, exist_ok=True)
    dst_model.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_dataset, dst_dataset)
    shutil.copy2(src_model, dst_model)
    LOGGER.info("Config senkronizasyonu tamamlandı: %s ve %s", dst_dataset, dst_model)
    return dst_model


def run_rtdetr_train(
    rtdetr_root: Path,
    config_path: Path,
    output_dir: Path,
    pretrained_path: Path | None,
    use_amp: bool,
) -> int:
    """Orijinal RT-DETR train.py scriptini çalıştırır."""
    train_py = rtdetr_root / "tools" / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"RT-DETR train script bulunamadı: {train_py}")

    cmd: List[str] = [
        sys.executable,
        str(train_py),
        "-c",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if pretrained_path is not None:
        cmd += ["-t", str(pretrained_path)]
    if use_amp:
        cmd += ["--use-amp"]

    LOGGER.info("Eğitim başlatılıyor: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(rtdetr_root), check=False)
    return proc.returncode


def compute_custom_score(metrics: Dict[str, float]) -> float:
    """Recall-ağırlıklı skor hesaplar: 0.6 * recall + 0.4 * mAP50."""
    recall = float(metrics.get("recall", 0.0))
    map50 = float(metrics.get("map50", 0.0))
    return 0.6 * recall + 0.4 * map50


def collect_epoch_metrics(epoch: int) -> Dict[str, float]:
    """Epoch metriklerini toplamak için hook.

    Not:
        RT-DETR log formatı değişebildiği için burada örnek bir şablon döndürülür.
        Gerçek projede output log/JSON dosyası parse edilip doldurulmalıdır.
    """
    _ = epoch
    return {
        "map50": 0.0,
        "map50_95": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "giou_loss": 0.0,
        "cls_loss": 0.0,
        "l1_loss": 0.0,
    }


def save_training_log(log_path: Path, rows: List[Dict[str, float]]) -> None:
    """Epoch bazlı eğitim logunu JSON formatında kaydeder."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Script giriş noktası."""
    args = parse_args()

    config_path = args.config
    if not args.no_sync_configs:
        synced_model_cfg = sync_project_configs(args.rtdetr_root)
        expected = PROJECT_ROOT / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_pothole.yml"
        if config_path.resolve() == expected.resolve() or str(config_path).endswith(
            "configs/rtdetrv2/rtdetrv2_r18vd_pothole.yml"
        ):
            config_path = synced_model_cfg

    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    torch.backends.cudnn.benchmark = True

    pretrained_path = maybe_validate_checkpoint(args.pretrained, args.output_dir)

    return_code = run_rtdetr_train(
        rtdetr_root=args.rtdetr_root,
        config_path=config_path,
        output_dir=args.output_dir,
        pretrained_path=pretrained_path,
        use_amp=args.use_amp,
    )
    if return_code != 0:
        raise RuntimeError(f"RT-DETR train.py hata kodu ile çıktı: {return_code}")

    total_epochs = int(cfg.get("epoches", 200))
    patience = int(args.early_stopping_patience)

    best_score = -1.0
    stale_epochs = 0
    training_rows: List[Dict[str, float]] = []

    for epoch in range(1, total_epochs + 1):
        metrics = collect_epoch_metrics(epoch)
        score = compute_custom_score(metrics)

        row = {"epoch": epoch, **metrics, "custom_score": score}
        training_rows.append(row)

        LOGGER.info(
            "Epoch %d | mAP50=%.4f mAP50:95=%.4f Recall=%.4f Precision=%.4f "
            "Loss[g=%.4f c=%.4f l1=%.4f] Score=%.4f",
            epoch,
            metrics["map50"],
            metrics["map50_95"],
            metrics["recall"],
            metrics["precision"],
            metrics["giou_loss"],
            metrics["cls_loss"],
            metrics["l1_loss"],
            score,
        )

        if score > best_score:
            best_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            LOGGER.info("Early stopping tetiklendi (patience=%d).", patience)
            break

    save_training_log(args.output_dir / "training_log.json", training_rows)
    LOGGER.info("Eğitim logu kaydedildi: %s", args.output_dir / "training_log.json")


if __name__ == "__main__":
    main()
