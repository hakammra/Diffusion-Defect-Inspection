# Portfolio and CV wording

## Project title

**LabelInspect — Reproducing and adapting diffusion anomaly segmentation for visual quality inspection**

## Current CV entry

> **Computer Vision Reproducibility Project — DTU-Net Diffusion Anomaly Segmentation**
>
> Reproduced and adapted a WACV 2025 diffusion anomaly-segmentation method in PyTorch for fixed-design printed-label inspection. Built fiducial registration, leakage-controlled physical splits, pre-model masks, frozen-threshold evaluation and resumable Tesla T4 notebooks. On a 53-image locked pilot, achieved 0.938 tear-versus-normal AUROC and 0.498 mean tear Dice; documented near-chance overall AUROC and failure on smaller covered-print and smudge defects.

Do not describe 0.666 as competitive performance or claim that the paper was reproduced successfully.

## Interview explanation

1. The selected paper proposed DTU-Net with Tsimplex noise for self-supervised anomaly segmentation.
2. We reproduced the executable method path and designed controlled experiments.
3. Training loss converged, but locked leather and fabric evaluations were weak.
4. We retained the negative results, diagnosed reconstruction noise and preprocessing mismatch, and designed a simpler baseline plus a controlled printed-label application.
5. On the locked printed-label test, the method detected large tears but missed covered-print and smudge defects; this showed that defect scale and image-score design matter.
6. The main engineering lessons were data leakage prevention, validation-only threshold selection, reproducible checkpoints, and the difference between attractive heatmaps and aggregate performance.

## Publication language

Until baselines, ablations, a larger independent set and the demo are complete, describe `paper/TECHNICAL_REPORT.md` as a **working technical report**, not a published research paper or preprint.
