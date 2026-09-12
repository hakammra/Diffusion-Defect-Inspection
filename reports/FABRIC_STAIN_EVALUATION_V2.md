# Fabric Stain Evaluation (V2 Protocol)

## Summary

Following diagnosis of the V1 evaluation, where reconstructed fabric images were noisy due to excessive diffusion steps ($t = 250$), multi-channel chromatic artifacts, and an unnormalized residual threshold, the **V2 evaluation protocol** was executed using the 2,000-step checkpoint with noise-robust inference and adaptive thresholding:

1. **Diffusion Distance**: $t_{\text{distance}} = 50$ (reduced from 250).
2. **Grayscale Luminance Residual**: RGB images projected to luminance before differencing, eliminating chromatic artifacts.
3. **Spatial Smoothing**: $5 \times 5$ average pooling on squared residuals to suppress isolated weave noise spikes.
4. **Per-Image Background Normalization**: Normalization by $(R - \text{median}) / \text{IQR}$ to cancel per-image baseline reconstruction noise floors.
5. **Score Statistic**: 99.9th percentile of the normalized residual map.
6. **Calibration**: Threshold frozen at the maximum V2 score among the 10 held-out normal validation images ($\tau = 10.100$).

---

## Comparison: V1 vs. V2

| Measurement | V1 Protocol ($t=250$, Raw p99.5) | V2 Protocol ($t=50$, Smoothed + IQR Norm) | Improvement |
|---|---:|---:|---|
| **Image AUROC** | **0.536** | **0.809** | **+0.273 (+51%)** |
| **Average Precision** | 0.938 | 0.978 | +0.040 |
| **Specificity** | 0.900 (9/10) | **1.000 (10/10)** | **Zero false alarms** |
| **Sensitivity (@ calibrated threshold)** | 0.270 (27/100) | **0.470 (47/100)** | **+74% more stains detected** |
| **True Positives / False Negatives** | 27 / 73 | 47 / 53 | +20 detected defects |
| **False Positives / True Negatives** | 1 / 9 | 0 / 10 | 0 false positives |
| **Balanced Accuracy** | 0.585 | **0.735** | +0.150 |
| **Inference Time** | 17.72 s/image | **3.71 s/image** | **4.8× faster** |

---

## Visual Observations

1. **Reconstruction Clarity**: At $t = 50$, reconstructions are clean, sharp, and preserve the natural fabric weave without the heavy, speckled, multicoloured noise seen at $t = 250$.
2. **Anomaly Isolation**: Stains produce distinct, localized response peaks in the normalized residual maps, while normal fabric regions remain suppressed below the threshold.
3. **Zero False Positives**: All 10 untouched normal test images remained safely below the validation-calibrated threshold (mean normal score: 6.44 vs threshold 10.10).

---

## Evidence

- Archive: `results/fabric_stain/fabric_stain_evaluation_v2_results.zip`
- Extracted Artifacts: `artifacts/fabric_stain_evaluation_v2/`
  - `evaluation_report_v2.json`
  - `evaluation_preview_v2.png`
  - `roc_curve_v2.png`
  - `calibration_v2.json`
  - `per_image_v2.csv`
  - `test_maps_v2_float16.npz`
