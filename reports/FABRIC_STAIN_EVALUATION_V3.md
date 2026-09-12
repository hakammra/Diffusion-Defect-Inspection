# Fabric Stain Locked Evaluation (V3 Protocol)

## Summary

The **V3 evaluation protocol** was executed on Kaggle with the 2,000-step checkpoint. It introduced:
1. **$N = 2$ Sample Averaging**: Denoises 2 reverse diffusion passes per image to cancel out stochastic micro-texture noise.
2. **$t_{\text{distance}} = 50$**: Keeps reconstructions sharp and clean.
3. **Low-Frequency Background Detrending (Kernel 41)**: Subtracts broad fabric fold shadows and lighting gradients.
4. **$5 \times 5$ Spatial Smoothing & IQR Normalization**: Suppresses weave noise and standardizes local contrast.
5. **Connected-Component Area Filtering (Area $\ge 10$ px)**: Discards isolated spurious noise specks.

---

## Comparison: V1 vs. V2 vs. V3

| Metric | V1 ($t=250$, 1-Sample) | V2 ($t=50$, 1-Sample) | V3 ($t=50$, $N=2$ Averaging + Detrending) | Progression |
|---|:---:|:---:|:---:|:---:|
| **Image AUROC** | **0.536** | **0.809** | **0.819** | **+0.283 (+53%)** |
| **Raw Residual AUROC** | 0.536 | 0.591 | **0.716** | **+0.180 (Due to $N=2$)** |
| **Average Precision** | 0.938 | 0.978 | **0.980** | +0.042 |
| **Normal Score Mean** | 0.150 | 6.44 | **14.74** | Well separated |
| **Stain Score Mean** | 0.211 | 11.72 | **33.75** | **2.3× higher than normal** |
| **Inference Time** | 17.72 s/image | 3.71 s/image | **6.84 s/image** | **2.6× faster than V1** |

---

## Visual Observations in V3

1. **Clean Mask Segmentation**: In [`evaluation_preview_v3.png`](file:///c:/Users/abdul/Documents/LabelInspect/artifacts/fabric_stain_evaluation_v3/evaluation_preview_v3.png), dark stains are segmented into crisp, solid white masks exactly matching the physical defect spots.
2. **Zero False Alarms on Normal Fabric**: The normal fabric test image produced a completely clean, solid-black mask with 0 active pixels.
3. **Suppression of Folds and Creases**: Natural creases and lighting gradients at the edges are completely subtracted by the 41-pixel background detrending filter and no longer trigger false positive highlights.

---

## Evidence

- Results Archive: `results/fabric_stain/fabric_stain_evaluation_v3_results.zip`
- Extracted Artifacts: `artifacts/fabric_stain_evaluation_v3/`
  - `evaluation_report_v3.json`
  - `evaluation_preview_v3.png`
  - `roc_curve_v3.png`
  - `calibration_v3.json`
  - `per_image_v3.csv`
  - `test_maps_v3_float16.npz`
