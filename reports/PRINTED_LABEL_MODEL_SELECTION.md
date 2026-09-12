# Printed-label pre-test model selection

## Protocol

The frozen 2,000-step checkpoint was evaluated at partial-diffusion distances 50, 100
and 250. The comparison used the ten registered normal-validation photographs from
physical labels `N09` and `N10`, together with 30 controlled digital defects derived
from those images: ten missing-print regions, ten smudges and ten edge tears. No real
test image was loaded.

For each distance, its pixel and image thresholds were calibrated independently from
the ten unmodified validation normals. The selection rule was recorded before viewing
the comparison: highest mean synthetic-defect Dice, then image sensitivity, then the
shorter distance.

## Results

| Distance | Mean Dice | Mean IoU | Image sensitivity | Seconds/image |
|---:|---:|---:|---:|---:|
| 50 | **0.532** | **0.429** | 0.467 | **3.60** |
| 100 | 0.504 | 0.390 | **0.700** | 6.97 |
| 250 | 0.445 | 0.328 | 0.333 | 17.32 |

Distance 50 was selected by the predefined segmentation rule. It also reduced
inference time by approximately 4.8 times relative to distance 250.

At distance 50, smudges were the strongest synthetic category with mean Dice 0.843
and 100% image sensitivity. Missing print achieved mean Dice 0.546 but only 20% image
sensitivity under the conservative image-level cutoff. Edge tears achieved mean Dice
0.206 and 20% image sensitivity. The segmentation mask can therefore overlap a
missing-print region even when the image score does not exceed the maximum-normal
cutoff.

## Frozen real-test configuration

- Checkpoint: 2,000 optimizer steps.
- Four-fiducial registration and 224 x 224 padded input.
- Partial-diffusion distance: 50.
- Reconstruction seed: 230274 (`230224 + 50`).
- Pixel threshold: 0.0235065464.
- Image score: 99.5th percentile of residual pixels in the label region.
- Image threshold: 0.0308046471.
- Target validation-normal pixel FPR: 0.005.

These choices supersede the earlier distance-250 calibration. They must not be changed
after real test predictions are viewed.

## Limitations

The synthetic defects support parameter selection but do not estimate real-world
accuracy. The digital tear is a simple grey edge wedge, and all three categories are
cleaner than physical damage. Real missing-print, smudge and tear results may differ.
The comparison also uses one stochastic reconstruction stream per candidate. The
locked real test must report aggregate metrics, per-category results, and failures.
