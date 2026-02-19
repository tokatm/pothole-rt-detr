# RT-DETRv2 Pothole Detection - Recall Metrics Analysis

## Executive Summary

Model peaked at **epoch 7** (mAP@50:95 = 0.183) and has been consistently degrading since.
By epoch 13, mAP dropped to 0.145 (21% relative decline). Training loss continues to decrease
while validation metrics worsen — this is **classic overfitting**. The current training
configuration needs significant changes to achieve the recall targets required for ADAS safety.

---

## 1. Epoch-by-Epoch Metric Tracking

| Epoch | mAP@50:95 | mAP@50 | mAP@75 | AP_s  | AP_m  | AP_l  | AR@100 | AR_s  | AR_m  | AR_l  | AR@50 | AR@75 |
|-------|-----------|--------|--------|-------|-------|-------|--------|-------|-------|-------|-------|-------|
| 7*    | **0.183** | -      | -      | -     | -     | -     | -      | -     | -     | -     | -     | -     |
| 9†    | ~0.174    | -      | -      | -     | -     | -     | -      | -     | 0.557‡| 0.766‡| 0.374‡| -     |
| 10    | 0.165     | 0.351  | 0.128  | 0.087 | 0.189 | 0.314 | 0.410  | 0.276 | 0.492 | 0.583 | 0.761 | 0.392 |
| 11    | 0.158     | 0.337  | 0.119  | 0.101 | 0.168 | 0.304 | 0.370  | 0.252 | 0.416 | 0.582 | 0.680 | 0.355 |
| 12    | 0.159     | 0.343  | 0.122  | 0.085 | 0.173 | 0.316 | 0.376  | 0.236 | 0.449 | 0.583 | 0.701 | 0.354 |
| 13    | 0.145     | 0.311  | 0.110  | 0.100 | 0.138 | 0.279 | 0.326  | 0.235 | 0.363 | 0.486 | 0.613 | 0.307 |
| 14    | 0.160     | 0.346  | 0.125  | 0.101 | 0.167 | 0.308 | 0.362  | 0.234 | 0.420 | 0.571 | 0.689 | 0.336 |

*Best checkpoint. †Partial results from logs. ‡These are AR_large, AR@50_all, AR@75_all from epoch 9 partial output.

---

## 2. Critical Issues Identified

### 2.1 Overfitting (Primary Problem)

**Evidence:**
- Training loss steadily decreasing: 10.15 (ep10) → 9.84 (ep10 avg) → 9.59 (ep14)
- Validation mAP peaked at epoch 7 and has not recovered in 7+ epochs
- Gap widens every epoch — model memorizes training set instead of generalizing

**Root causes:**
1. **Dataset likely too small** for 200-epoch training at constant lr=2e-4
2. **Backbone freeze_at=2 is insufficient** — the later ResNet stages may still overfit
3. **No regularization beyond weight_decay=1e-4** — no dropout, no label smoothing
4. **Augmentation pipeline is standard but not aggressive enough** for a small dataset

### 2.2 Recall Degradation (Most Critical for ADAS)

The project explicitly prioritizes recall (0.6*recall + 0.4*mAP50) for safety. But recall is collapsing:

| Metric        | Epoch 10 | Epoch 13 | Epoch 14 | Change  |
|---------------|----------|----------|----------|---------|
| AR@100 (all)  | 0.410    | 0.326    | 0.362    | -11.7%  |
| AR@50 (all)   | 0.761    | 0.613    | 0.689    | -9.5%   |
| AR (small)    | 0.276    | 0.235    | 0.234    | -15.2%  |
| AR (medium)   | 0.492    | 0.363    | 0.420    | -14.6%  |
| AR (large)    | 0.583    | 0.486    | 0.571    | -2.1%   |

Key observations:
- **Small and medium potholes lose recall fastest** — these are the most dangerous to miss
- Large object recall is relatively stable (~0.57-0.58)
- AR@50 >> AR@75 gap (0.689 vs 0.336 at ep14) indicates **poor localization quality**

### 2.3 Poor Localization Quality

- mAP@75 = 0.125 vs mAP@50 = 0.346 at epoch 14 → model detects potholes roughly but fails at precise bounding box regression
- GIoU loss remains high (~0.65) and barely improves across epochs
- This matters for ADAS: imprecise boxes → wrong distance estimation → incorrect braking decisions

### 2.4 Small Object Performance is Critically Low

- AP_small fluctuates between 0.085-0.101 across all epochs
- AR_small peaked at 0.276 (ep10) and declined to 0.234 (ep14)
- For pothole detection, small objects = distant potholes = early warning = most safety-critical

---

## 3. Loss Component Analysis

### Training loss trends (epoch averages):

| Loss Component | Epoch 10 | Epoch 12 | Epoch 14 | Trend |
|----------------|----------|----------|----------|-------|
| Total loss     | 9.841    | 9.720    | 9.588    | ↓ decreasing |
| VFL (cls)      | 0.628    | 0.614    | 0.616    | → plateau |
| BBox (L1)      | 0.234    | 0.229    | 0.222    | ↓ slow |
| GIoU           | 0.667    | 0.666    | 0.646    | → nearly flat |
| VFL_dn_2       | 0.398    | 0.395    | 0.394    | → plateau |
| BBox_dn_2      | 0.198    | 0.191    | 0.190    | ↓ slow |
| GIoU_dn_2      | 0.513    | 0.508    | 0.499    | ↓ very slow |
| VFL_enc        | 0.680    | 0.664    | 0.671    | → fluctuating |

**Interpretation:**
- **VFL loss plateaued at ~0.62** — the classification head is not learning further meaningful features. This suggests the feature representations from frozen backbone stages are insufficient for fine-grained pothole vs. non-pothole discrimination.
- **GIoU loss remains high at ~0.65** — the model struggles with precise box regression. This is consistent with the large mAP@50 vs mAP@75 gap.
- **Denoising (DN) losses decrease consistently** — DN training auxiliary tasks work as designed, but the main task is not benefiting proportionally.
- **Encoder loss fluctuates** — the HybridEncoder's feature quality isn't improving.

---

## 4. Configuration vs Results Assessment

### What's working:
- **Pretrained COCO weights** — the model did reach 0.183 mAP early, showing transfer learning helps
- **Frozen backbone (first 2 stages)** — prevents catastrophic forgetting of low-level features
- **Recall-weighted selection metric** — correct priority for ADAS safety
- **Dual-threshold postprocessing** — a good design for production recall optimization

### What's not working:

| Configuration | Current | Problem | Recommended |
|---------------|---------|---------|-------------|
| LR schedule | MultiStepLR, drop at ep160 | Constant 2e-4 is too high after epoch 7; overfitting before milestone | CosineAnnealingLR or reduce milestones to [40, 80] |
| Epochs | 200 | Way too many for this dataset size | 50-80 with proper LR schedule |
| Weight decay | 1e-4 | Too weak for small dataset | 5e-4 to 1e-3 |
| Batch size | 32 | Potentially too large for small dataset | 16 with gradient accumulation |
| Augmentation | Standard DETR pipeline | Not aggressive enough | Add MixUp, Mosaic, CopyPaste for small objects |
| freeze_at | 2 | Only 2 stages frozen; later stages overfit | Consider freeze_at=3 or progressive unfreezing |
| Input size | 640x640 fixed | No multi-scale training (scales: ~) | Enable multi-scale [480, 512, 544, 576, 608, 640] |
| num_queries | 150 | Reasonable for single-class, but may limit recall | Test 200-300 range |

---

## 5. Specific Recommendations

### 5.1 Immediate (stop training, relaunch)

1. **Revert to epoch 7 checkpoint** — this is the best model available
2. **Implement CosineAnnealingLR** or at minimum change milestones to `[30, 60]` with warmup
3. **Reduce total epochs to 80** — the model is fully converged by epoch 15 at current settings
4. **Increase weight_decay to 5e-4**

### 5.2 Short-term (data and augmentation)

5. **Enable multi-scale training**: Set `scales: [480, 512, 544, 576, 608, 640]` in the collate_fn — this is the single biggest change for small object recall
6. **Add MixUp/Mosaic augmentation** — RT-DETRv2 supports this; it dramatically helps small datasets
7. **Add CopyPaste augmentation** specifically for small potholes — cut pothole patches and paste onto random road backgrounds
8. **Verify dataset quality** — with mAP@50:95 only reaching 0.183, there may be annotation quality issues (mislabeled images, inconsistent box boundaries, duplicate annotations)

### 5.3 Medium-term (architecture)

9. **Test freeze_at=3** and use a lower learning rate (5e-5) for unfrozen layers
10. **Consider RT-DETRv2-M (R50)** if compute allows — the R18 backbone may lack capacity for fine-grained pothole features
11. **Add EMA (Exponential Moving Average)** — smooths parameter updates and improves generalization; RT-DETR supports this via `ema:` config key
12. **Increase num_queries to 300** for recall experiments — 150 may be limiting in dense scenes

### 5.4 Evaluation and post-processing

13. **The custom postprocessor (`recall_optimized_postprocess.py`) compensates at inference but cannot fix poor model quality** — prioritize improving the base model's recall at training time
14. **Monitor per-size recall during training** — the current `best_stat` only tracks overall mAP; add size-stratified recall to the selection metric
15. **The threshold sweep in `03_evaluate.py` may need recalibration** after retraining — the confidence distribution will change

---

## 6. Dataset Size Estimation

Based on the training logs:
- 255 iterations per epoch with batch_size=32 → **~8,160 training images**
- 49 iterations for validation with batch_size=32 → **~1,568 validation images**

For single-class detection with RT-DETRv2-S, this dataset size is **borderline**. The COCO pretrained backbone helps, but:
- 200 epochs × 255 iterations = 51,000 gradient updates on ~8K images
- The model sees each image ~200 times — extreme repetition leads to memorization
- **Recommendation: Reduce to 50-80 epochs or significantly increase data augmentation diversity**

---

## 7. Summary of Diagnosis

```
Root Cause Chain:
  Small dataset (~8K images)
  + Constant high LR (2e-4 for 200 epochs)
  + Weak regularization (wd=1e-4, no dropout/EMA)
  + Insufficient augmentation (no multi-scale, no MixUp)
  = Overfitting starting at epoch 7-8
  → Recall degradation (0.410 → 0.326 AR@100)
  → Especially poor small object detection (AP_small < 0.10)
  → Unusable for ADAS safety requirements
```

**Priority action: Stop current training. Use epoch 7 checkpoint. Redesign LR schedule and augmentation pipeline. Retrain for 80 epochs max.**
