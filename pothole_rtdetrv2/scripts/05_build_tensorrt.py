#!/usr/bin/env python3
"""ONNX modelinden TensorRT engine oluşturur (Orin odaklı).

Dependencies:
    - tensorrt
    - pycuda

Usage:
    python scripts/05_build_tensorrt.py \
      --onnx rtdetrv2_r18vd_pothole.onnx \
      --engine rtdetrv2_r18vd_pothole_fp16.engine
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


class EntropyCalibrator:
    """INT8 calibration altyapı iskeleti (IInt8EntropyCalibrator2 için yer tutucu)."""

    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths



def parse_args() -> argparse.Namespace:
    """CLI argümanlarını parse eder."""
    parser = argparse.ArgumentParser(description="Build TensorRT engine")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("rtdetrv2_r18vd_pothole_fp16.engine"))
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    return parser.parse_args()


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    int8: bool,
    workspace_gb: float,
    calibrator: Optional[EntropyCalibrator],
) -> None:
    """TensorRT engine build sürecini çalıştırır."""
    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError("tensorrt modülü bulunamadı.") from exc

    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            raise ValueError("INT8 için calibrator gerekli.")
        LOGGER.info("INT8 calibration etkin.")

    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse hatası: {errors}")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine oluşturulamadı.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    LOGGER.info("Engine kaydedildi: %s", engine_path)


def main() -> None:
    """Program giriş noktası."""
    args = parse_args()

    calibrator = None
    if args.int8:
        if args.calib_dir is None or not args.calib_dir.exists():
            raise ValueError("INT8 için --calib-dir verilmelidir.")
        images = [p for p in args.calib_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        random.shuffle(images)
        calibrator = EntropyCalibrator(images[:100])

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        int8=args.int8,
        workspace_gb=args.workspace_gb,
        calibrator=calibrator,
    )

    LOGGER.warning(
        "Bu engine yalnızca build edilen cihazda çalışır. Orin'de çalıştırmak için engine'i Orin üzerinde build edin."
    )


if __name__ == "__main__":
    main()
