# RT-DETRv2 Pothole Detection (Recall-First)

Bu proje, **lyuwenyu/RT-DETR** PyTorch implementasyonunu temel alarak tek sinif pothole detection pipeline'i sunar. Hedef senaryo ADAS oldugu icin kacirilan cukurlari azaltmaya odaklidir.

## Mimari Ozet
- Model: **RT-DETRv2-S (R18-vd)** — 20M params, 60 GFLOPs
- Neden v2: v1'e gore daha iyi accuracy/speed dengesi, discrete sampling ve TensorRT deploy uyumlulugu
- Neden S varyanti: Jetson Orin Super 8GB uzerinde ~36 FPS (T4'te 217 FPS). Onceki RT-DETR-X Orin'de sadece 12.97 FPS verdi — ADAS icin yetersiz.
- Gorev: Tek sinif (pothole), recall agirlikli optimizasyon
- COCO AP farki (48.1 vs 54.8) tek sinif fine-tuning'de daralir

## Klasor Yapisi
```text
pothole_rtdetrv2/
├── configs/
│   ├── dataset/
│   │   └── pothole_detection.yml          # Dataset config (multi-scale, augmentation policy)
│   └── rtdetrv2/
│       └── rtdetrv2_r18vd_pothole.yml     # Model + training config
├── scripts/
│   ├── 00_fix_oversized_images.py         # Oversized goruntu guvenlik/resize adimi
│   ├── 00_copypaste_augment.py            # Offline copy-paste augmentation (kucuk pothole'lar icin)
│   ├── 00_mixup_augment.py                # Offline mixup augmentation (opsiyonel)
│   ├── 01_convert_yolo_to_coco.py         # YOLO→COCO format donusturucu
│   ├── 02_train.py                        # Egitim baslatici
│   ├── 03_evaluate.py                     # Detayli evaluation (recall-odakli metrikler)
│   ├── 04_export_onnx.py                  # ONNX export
│   ├── 05_build_tensorrt.py               # TensorRT engine olusturma (Orin icin)
│   ├── 06_inference_tensorrt.py           # TensorRT ile inference
│   └── 07_analyze_errors.py              # FN/FP analiz araci (gorsel kaydetme dahil)
├── src/
│   └── recall_optimized_postprocess.py    # Recall-odakli post-processing
├── analysis/
│   └── recall_metrics_analysis.md         # Onceki egitim analiz raporu
├── requirements.txt
└── README.md
```

## Kurulum
```bash
# 1. Bu repo'yu clone et
git clone <this-repo-url>

# 2. RT-DETR orijinal repo'yu clone et
git clone https://github.com/lyuwenyu/RT-DETR.git

# 3. Bagimliliklari yukle
pip install -r requirements.txt
```

Pretrained weight (manuel indir):
- [rtdetrv2_r18vd_120e_coco_rerun_48.1.pth](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth)

## Dataset Hazirligi

### Adim 1: Copy-Paste Augmentation (Opsiyonel ama Onerilen)
Kucuk pothole'larin recall'unu artirmak icin offline augmentation:
```bash
python scripts/00_copypaste_augment.py \
  --dataset-root /content/dataset/combined_dataset \
  --output-dir /content/dataset/combined_dataset \
  --paste-ratio 0.25 \
  --num-augmented 500
```

### Adim 2: YOLO -> COCO Donusum
```bash
python scripts/01_convert_yolo_to_coco.py \
  --dataset-root /content/dataset/combined_dataset \
  --output-dir /content/dataset/combined_dataset/annotations \
  --min-side-px 32 \
  --mixup-before-convert \
  --mixup-ratio 0.20 \
  --mixup-alpha 0.4
```
DIKKAT:
- Copy-paste augmentation calistirdiysan, donusumu TEKRAR calistir.
- `01_convert_yolo_to_coco.py` varsayilan olarak once `00_fix_oversized_images.py`
  calistirir. Bu sayede DecompressionBombError riski JSON'dan once temizlenir.
- Mixup adimi opsiyoneldir; sadece `--mixup-before-convert` verilirse train split'e yeni
  sentetik sample uretir.

## Egitim
```bash
python scripts/02_train.py \
  --config configs/rtdetrv2/rtdetrv2_r18vd_pothole.yml \
  --rtdetr-root /content/RT-DETR/rtdetrv2_pytorch \
  --use-amp \
  -t /content/weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
```

### Egitim Config Ozellikleri
- **Diferansiyel LR**: Backbone 10x dusuk LR (0.00002), encoder/decoder 0.0002
- **Multi-scale training**: [480..800] arasi 13 farkli olcek — kucuk pothole recall icin kritik
- **Augmentation policy**: Son 10 epoch'ta (epoch 40+) augmentation kapatilir
- **LR scheduler**: MultiStepLR milestones=[15, 30] — erken plateau kirma
- **Warmup**: Lineer warmup, 2000 iterasyon (~5 epoch)
- **EMA**: Exponential Moving Average acik (decay=0.9999)
- **Weight decay**: 5e-4 (kucuk dataset overfitting onleme)
- **Matcher**: HungarianMatcher (bipartite matching) dahil
- **Recall agirlikli best checkpoint**: `0.6 * recall + 0.4 * mAP50`
- **Early stopping**: `patience=15`
- **Toplam epoch**: 50
- **num_queries**: 150 (FP azaltma + Orin hiz optimizasyonu)

## Evaluation ve Optimal Threshold
```bash
python scripts/03_evaluate.py \
  --gt-json /content/dataset/combined_dataset/annotations/val_annotations.json \
  --pred-json /content/outputs/predictions_val.json \
  --output-dir /content/outputs/eval
```

Ciktilar:
- Threshold sweep CSV (0.1-0.9 arasi, 0.05 adimlarla)
- Optimal threshold secimi (safety-first: Recall>=0.85'te en yuksek F1)
- Kucuk/orta/buyuk nesne recall analizi (COCO boyut kategorileri)
- Precision-Recall curve
- `evaluation_results.json`

## Hata Analizi
```bash
python scripts/07_analyze_errors.py \
  --gt-json /content/dataset/combined_dataset/annotations/val_annotations.json \
  --pred-json /content/outputs/predictions_val.json \
  --image-dir /content/dataset/combined_dataset/images/val \
  --output-dir /content/outputs/error_analysis
```

FN/FP gorselleri `fn_images/` ve `fp_images/` klasorlerine kaydedilir.

## ONNX Export ve Dogrulama
```bash
python scripts/04_export_onnx.py \
  --checkpoint /content/outputs/best.pth \
  --output /content/outputs/rtdetrv2_r18vd_pothole.onnx
```

Notlar:
- opset 16 (TensorRT uyumlulugu)
- onnx.checker ile model dogrulamasi
- onnxruntime ile PyTorch fark kontrolu (`max_abs_diff < 1e-4` hedef)
- v2'nin discrete sampling yaklasimi deploy tarafinda avantaj saglar (grid_sample yerine index_select)

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

Uyari: TensorRT engine cihaza ozeldir. Orin uzerinde kullanmak icin engine'i Orin'de build edin.

## Inference Kullanimi
Tek gorsel/klasor/video:
```bash
python scripts/06_inference_tensorrt.py \
  --engine /content/outputs/rtdetrv2_r18vd_pothole_fp16.engine \
  --source /content/sample.jpg \
  --output-dir /content/outputs/infer
```

## Performans Tablosu (Doldurulacak)
| Ortam | FPS | Latency (ms) | mAP50 | Recall |
|---|---:|---:|---:|---:|
| A100 (train/eval) | TBD | TBD | TBD | TBD |
| Orin Super FP16 | TBD | TBD | TBD | TBD |

## Lisans
- RT-DETR: Apache-2.0
- Bu proje: Apache-2.0 ile uyumlu kullanim
- Ultralytics bagimliligı yoktur, AGPL-3.0 riski icermez.
- Ticari kullanim tamamen serbesttir.

## Bilinen Sorunlar ve Troubleshooting
- `tensorrt`/`pycuda` kurulumlari platforma duyarlidir; Orin uzerinde JetPack surumu ile esleshme gerekir.
- ONNX export sirasinda custom op hatasi alirsaniz RT-DETR branch surumunuzu guncelleyin.
- Dusuk recall durumunda threshold sweep sonuclarindan daha dusuk guven esigi secin.
- BN istatistik kaynakli dalgalanma gorurseniz `freeze_norm=True` ayarini koruyun.
- Multi-scale training VRAM sorunlari cikarirsa `total_batch_size`'i 16'ya dusurun.
- Copy-paste augmentation sonrasi COCO donusumunu tekrar calistirmayi unutmayin.
