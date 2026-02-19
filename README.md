# RT-DETRv2 Pothole Detection (Recall-First)

Bu proje, **lyuwenyu/RT-DETR** PyTorch implementasyonunu temel alarak tek sınıf pothole detection pipeline'ı sunar. Hedef senaryo ADAS olduğu için kaçırılan çukurları azaltmaya odaklıdır.

## Mimari Özet
- Model: **RT-DETRv2-S (R18-vd)**
- Neden v2: v1'e göre daha iyi accuracy/speed dengesi, discrete sampling ve deploy uyumluluğu
- Neden S varyantı: Jetson Orin Super 8GB üzerinde pratik FPS ve RAM dengesi
- Görev: Tek sınıf (pothole), recall ağırlıklı optimizasyon

## Klasör Yapısı
```text

├── configs/
├── scripts/
├── src/
├── requirements.txt
└── README.md
```

## Kurulum
```bash
git clone https://github.com/lyuwenyu/RT-DETR.git
pip install -r requirements.txt
```

Pretrained weight (manuel indir):
- [rtdetrv2_r18vd_120e_coco_rerun_48.1.pth](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth)

## Dataset Hazırlığı (YOLO -> COCO)
```bash
python scripts/01_convert_yolo_to_coco.py \
  --dataset-root /content/dataset/combined_dataset \
  --output-dir /content/dataset/combined_dataset/annotations
```

## Eğitim
```bash
python scripts/02_train.py \
  --config configs/rtdetrv2/rtdetrv2_r18vd_pothole.yml \
  --rtdetr-root /content/RT-DETR/rtdetrv2_pytorch \
  --use-amp
```

Eğitimde:
- script, `configs/` dosyalarını otomatik olarak RT-DETR config klasörüne senkronize eder
- `-t` verilmezse varsayılan RT-DETRv2-S pretrained dosyası otomatik indirilir
- AMP için `--use-amp` kullanılabilir
- scheduler: MultiStepLR (RT-DETR sürümleriyle güvenli uyum)
- recall ağırlıklı best checkpoint skoru: `0.6 * recall + 0.4 * mAP50`
- early stopping: `patience=60`
- epoch bazlı log: `training_log.json`

## Evaluation ve Optimal Threshold
```bash
python scripts/03_evaluate.py \
  --gt-json /content/dataset/combined_dataset/annotations/val_annotations.json \
  --pred-json /content/outputs/predictions_val.json \
  --output-dir /content/outputs/eval
```

Çıktılar:
- threshold sweep CSV (0.1-0.9 arası)
- optimal threshold seçimi (safety-first)
- küçük/orta/büyük nesne recall analizi
- PR curve
- `evaluation_results.json`

## ONNX Export ve Doğrulama
```bash
python scripts/04_export_onnx.py \
  --checkpoint /content/outputs/best.pth \
  --output /content/outputs/rtdetrv2_r18vd_pothole.onnx
```

Notlar:
- opset 16
- onnx.checker ile model doğrulaması
- onnxruntime ile PyTorch fark kontrolü (`max_abs_diff < 1e-4` hedef)
- v2'nin discrete sampling yaklaşımı deploy tarafında avantaj sağlar

## TensorRT Deploy (Orin)
```bash
python scripts/05_build_tensorrt.py \
  --onnx /content/outputs/rtdetrv2_r18vd_pothole.onnx \
  --engine /content/outputs/rtdetrv2_r18vd_pothole_fp16.engine \
  --fp16
```

`trtexec` alternatifi:
```bash
trtexec --onnx=rtdetrv2_r18vd_pothole.onnx --saveEngine=rtdetrv2_fp16.engine --fp16 --workspace=4096
```

Uyarı: TensorRT engine cihaza özeldir. Orin üzerinde kullanmak için engine'i Orin'de build edin.

## Inference Kullanımı
Tek görsel/klasör/video:
```bash
python scripts/06_inference_tensorrt.py \
  --engine /content/outputs/rtdetrv2_r18vd_pothole_fp16.engine \
  --source /content/sample.jpg \
  --output-dir /content/outputs/infer
```

## Hata Analizi
```bash
python scripts/07_analyze_errors.py \
  --gt-json /content/dataset/combined_dataset/annotations/val_annotations.json \
  --pred-json /content/outputs/predictions_val.json \
  --output-dir /content/outputs/error_analysis
```

## Performans Tablosu (Doldurulacak)
| Ortam | FPS | Latency (ms) | mAP50 | Recall |
|---|---:|---:|---:|---:|
| A100 (train/eval) | TBD | TBD | TBD | TBD |
| Orin Super FP16 | TBD | TBD | TBD | TBD |

## Lisans
- RT-DETR: Apache-2.0
- Bu proje: Apache-2.0 ile uyumlu kullanım
- Ultralytics bağımlılığı yoktur, AGPL-3.0 riski içermez.

## Bilinen Sorunlar ve Troubleshooting
- `tensorrt`/`pycuda` kurulumları platforma duyarlıdır; Orin üzerinde JetPack sürümü ile eşleşme gerekir.
- ONNX export sırasında custom op hatası alırsanız RT-DETR branch sürümünüzü güncelleyin.
- Düşük recall durumunda threshold sweep sonuçlarından daha düşük güven eşiği seçin.
- BN istatistik kaynaklı dalgalanma görürseniz `freeze_norm=True` ayarını koruyun.
