# Real printed-label validation

## Outcome

The normal-only calibration completed successfully using the frozen 2,000-step
printed-label checkpoint. It processed ten registered photographs from physical labels
`N09` and `N10`; no test images were loaded.

| Item | Value |
|---|---:|
| Partial-diffusion distance | 250 |
| Pixel residual threshold | 0.162425 |
| Target normal pixel FPR | 0.005 |
| Observed aggregate normal pixel FPR | 0.004999 |
| Image-score threshold | 0.213790 |
| Observed normal image FPR | 0.000 |
| Mean amortized inference time | 18.78 seconds/image |
| Peak allocated GPU memory | 0.481 GiB |

The pixel threshold is the higher 99.5th percentile of validation-normal residuals
inside the registered label region. Each image score is its 99.5th-percentile residual;
the maximum of the ten normal image scores sets the image cutoff.

## Review

The calibration calculation is internally consistent. The aggregate predicted
normal-pixel fraction matches its 0.5% target, and the strict image-level threshold
produces no validation image false positives.

Reconstruction remains visibly noisy. Scores vary materially across capture conditions:
the lowest image score is 0.13285 and the highest is 0.21379. The white-background
capture `N09_B5_49.png` sets the image threshold and has a 1.62% predicted-pixel
fraction, compared with the 0.50% aggregate target. This suggests sensitivity to
lighting or acquisition condition even after geometric registration.

These findings do not measure defect detection. The frozen settings may detect larger
smudges and tears while struggling with small missing-print regions if their residuals
are comparable to reconstruction texture. The locked test must report that outcome
without retuning on test images. A registered template-residual baseline should be
evaluated alongside DTU-Net so the application study remains informative if the
diffusion model underperforms.

The generated JSON labels the 18.78-second timing as a median, but its notebook formula
divides total batch time by image count, so it is an amortized mean. The notebook builder
has been corrected for future runs; the original validation artifact is retained.

## Frozen test configuration

- 2,000-step checkpoint.
- Four-fiducial perspective registration.
- Aspect-preserving resize and padding to 224 x 224.
- Partial-diffusion distance 250.
- Pixel threshold 0.162425.
- Image score: 99.5th percentile of label-region residuals.
- Image threshold 0.213790.
