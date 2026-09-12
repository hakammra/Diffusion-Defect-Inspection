# Fabric stain evaluation

## Locked result

The 2,000-step DTU-Net/Tsimplex checkpoint was evaluated once using the protocol fixed before inference: ten normal validation images for cutoff calibration, ten untouched normal test images, and a seed-fixed subset of 100 stain images. Complete photographs were resized to 224 × 224 and the image score was the 99.5th percentile of squared reconstruction residuals.

| Measurement | Result |
|---|---:|
| Image AUROC | 0.536 |
| Average precision | 0.938 |
| Sensitivity | 0.270 |
| Specificity | 0.900 |
| Balanced accuracy | 0.585 |
| True positives / false negatives | 27 / 73 |
| True negatives / false positives | 9 / 1 |
| Mean normal score | 0.1501 |
| Mean stain score | 0.2108 |
| Approximate inference time | 17.72 seconds/image |

The positive prevalence is 100/110 = 0.909, so average precision has a high baseline and should not be read as strong performance. AUROC is close to random ranking, and the normal-only cutoff misses 73% of stains. The model mainly responds to very large, dark stains.

The qualitative preview shows noisy reconstructions even for normal fabric. Consequently, residual maps contain reconstruction texture across most pixels rather than isolating the defect. A lower training loss did not translate into useful anomaly separation.

## Post-evaluation diagnosis

Alternative score summaries were checked only after seeing the locked result. The maximum residual yielded AUROC 0.6175, and the maximum 2 × 2 block-averaged residual yielded 0.665. These are exploratory diagnostics selected on the test result and must not replace the registered AUROC of 0.536.

Likely causes include the whole-image evaluation resize differing from the 512-pixel crop scale used for training, only 48 normal source images, 2,000 optimizer steps rather than the paper's much longer schedule, and the author's partial reconstruction remaining visibly noisy at `t_distance = 250`.

## Conclusion

This run is a valid negative result. It does not support a claim that the trained model reliably detects fabric stains. The current aggregate dataset also lacks pixel masks for this stain cohort, so it cannot support segmentation Dice or IoU.

A stronger next experiment should use a source-consistent fabric dataset with pixel masks, such as the dedicated AITEX data, train one fabric type at a time, and evaluate at the same patch scale used during training. The printed-label application remains the better novel application for the assignment.
