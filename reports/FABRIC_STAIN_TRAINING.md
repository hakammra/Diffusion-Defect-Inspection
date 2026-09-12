# Fabric stain training status

The DTU-Net/Tsimplex fabric-stain run resumed from the verified 100-step pilot and completed 2,000 total optimizer steps on a Tesla T4.

| Measurement | Value |
|---|---:|
| Training normal images | 48 |
| Validation normal images held out | 10 |
| Locked test images accessed during training | 0 |
| Model parameters | 28,863,812 |
| Initial loss | 0.420431 |
| Final loss | 0.042479 |
| Final 20-step mean loss | 0.063013 |
| Continuation runtime | 522.4 seconds |
| Median step time | 0.2648 seconds |
| Peak GPU allocation | 0.781 GiB |

The loss decreased and stabilized, gradients remained finite, and the checkpoint is ready for evaluation. A decreasing noise-prediction loss confirms optimization but does not establish anomaly-detection performance.

The next locked evaluation uses ten validation normals for normal-only cutoff calibration, ten untouched test normals and a seed-fixed subset of 100 stain images. The selected stain cohort provides no pixel masks, so the evaluation reports image-level metrics and qualitative anomaly maps rather than Dice or IoU.
