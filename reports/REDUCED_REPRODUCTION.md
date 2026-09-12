# Reduced MVTec leather reproduction

## Outcome

The evaluation pipeline completed on the official MVTec AD leather test set, but the reduced run did not reproduce the paper's reported performance. This is a useful negative result and must not be described as a successful numerical reproduction.

| Measure | Reduced run |
| --- | ---: |
| Training normals | 196 |
| Normal validation images | 49 |
| Official test images | 124 |
| Optimizer steps | 2,000 |
| Partial-diffusion distance | 250 |
| Pixel AUROC | 0.6656 |
| Image AUROC, 99.5th-percentile score | 0.5978 |
| Mean Dice on defective images | 0.0857 |
| Mean IoU on defective images | 0.0469 |
| Normal test pixel false-positive rate | 0.0081 |

The threshold was calibrated only on held-out normal validation images at a target pixel false-positive rate of 0.005. Test masks were loaded only after the threshold was frozen.

## Interpretation

The residual maps respond to some defects, especially glue and color anomalies, but the reconstruction contains high-frequency error across normal leather texture. Thresholding therefore produces scattered false positives and weak overlap with the defect masks.

This result differs materially from the paper protocol:

- The paper reports 3,000 epochs, batch size 32, and EMA rate 0.9999. This run used 2,000 optimizer steps with batch size 2 and did not maintain an EMA model.
- The public repository contains multiple inconsistent leather configurations and no released checkpoint. Its main training script also hard-codes some model settings instead of consistently using the JSON values.
- The inspected architecture was the SPE + DMHA + HFF + refinement configuration. The paper's ablation table reports materially different behavior across architecture variants.
- The run used one reconstruction per image and a normal-only threshold. The paper discusses averaging and selecting a diffusion-step range using Dice, but does not provide enough detail to reproduce every selection without test-set tuning.

## Evidence

- `artifacts/mvtec_leather_training/pilot_report.json`
- `artifacts/mvtec_leather_training/training_history.csv`
- `artifacts/mvtec_leather_evaluation/calibration.json`
- `artifacts/mvtec_leather_evaluation/evaluation_report.json`
- `artifacts/mvtec_leather_evaluation/per_image.csv`
- `artifacts/mvtec_leather_evaluation/evaluation_preview.png`

## Claim boundary

Permitted description: "Implemented and evaluated a reduced reproduction of a WACV 2025 diffusion anomaly-segmentation method on MVTec AD leather; documented upstream configuration gaps and analyzed failure modes under constrained compute."

Do not claim that the paper's 99.90 leather AUROC was reproduced.
