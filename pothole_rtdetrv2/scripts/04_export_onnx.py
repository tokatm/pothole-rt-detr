#!/usr/bin/env python3
"""RT-DETRv2 modelini ONNX formatına export eder ve doğrular.

Dependencies:
    - torch
    - onnx
    - onnxruntime
    - onnxsim (opsiyonel)

Usage:
    python scripts/04_export_onnx.py \
      --checkpoint /content/outputs/pothole_rtdetrv2/best.pth \
      --output /content/outputs/rtdetrv2_r18vd_pothole.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import torch


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI argümanlarını parse eder."""
    parser = argparse.ArgumentParser(description="Export RT-DETRv2 to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("rtdetrv2_r18vd_pothole.onnx"))
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Checkpoint'ten modeli yükler.

    Not:
        Bu şablonda kullanıcı kendi RT-DETR model oluşturucusunu entegre etmelidir.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = torch.nn.Identity()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    _ = ckpt
    model.eval().to(device)
    return model


def try_simplify_onnx(path: Path) -> None:
    """ONNX graph'i onnxsim ile sadeleştirmeyi dener."""
    try:
        from onnxsim import simplify

        model = onnx.load(str(path))
        simplified, ok = simplify(model)
        if ok:
            onnx.save(simplified, str(path))
            LOGGER.info("ONNX simplify başarılı.")
        else:
            LOGGER.warning("ONNX simplify doğrulaması başarısız.")
    except Exception as exc:
        LOGGER.warning("onnxsim çalışmadı (opsiyonel): %s", exc)


def run_ort_compare(onnx_path: Path, input_tensor: torch.Tensor, torch_output: torch.Tensor) -> float:
    """PyTorch ve ONNXRuntime çıktıları arasındaki max mutlak farkı döndürür."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"images": input_tensor.detach().cpu().numpy()})
    ort_arr = np.array(ort_out[0])
    torch_arr = torch_output.detach().cpu().numpy()
    return float(np.max(np.abs(ort_arr - torch_arr)))


def export_onnx(model: torch.nn.Module, output_path: Path, opset: int, device: str) -> Dict[str, float]:
    """ONNX export, checker ve numerik doğrulamayı yürütür."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        torch_out = model(dummy)
        if not isinstance(torch_out, torch.Tensor):
            torch_out = dummy

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["labels", "boxes", "scores"],
    )

    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)
    LOGGER.info("ONNX checker başarılı: %s", output_path)

    try_simplify_onnx(output_path)
    diff = run_ort_compare(output_path, dummy, torch_out)
    return {"max_abs_diff": diff}


def main() -> None:
    """Program giriş noktası."""
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA yok, CPU'ya düşülüyor.")
        device = "cpu"

    model = load_model(args.checkpoint, device)
    stats = export_onnx(model, args.output, args.opset, device)

    LOGGER.info("Max abs diff (PyTorch vs ONNXRuntime): %.8f", stats["max_abs_diff"])
    if stats["max_abs_diff"] >= 1e-4:
        LOGGER.warning("Numerik fark 1e-4 eşiğini aştı.")


if __name__ == "__main__":
    main()
