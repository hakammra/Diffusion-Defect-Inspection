# Frozen tiled fabric-stain evaluation

## Protocol

The final run applied the inference method and operating threshold selected before this evaluation.
No setting was changed after the final scores were observed.

| Setting | Frozen value |
|---|---:|
| DTU-Net checkpoint | 2,000 optimizer steps |
| Partial-diffusion distance | 50 |
| Reconstruction samples | 1 |
| Source tile size | 512 x 512 pixels |
| Tile stride | 384 pixels |
| Tiles per image | 20 |
| Per-tile score | Residual-map 99.9th percentile |
| Image score | Mean of two highest tile scores |
| Decision threshold | 16.47188949584961 |

The final stain cohort contains 100 images from shuffled positions `[130:230]`. It is disjoint from
the 100 stains used by V1-V5 and the 30 stains used for tiled model selection. The ten normal
reference images were used in earlier exploratory evaluations, so normal specificity is not a fully
independent estimate. The source dataset provides no pixel masks; this is an image-level evaluation
with qualitative localization maps.

## Results

| Metric | Final result |
|---|---:|
| Image AUROC | **0.773** |
| Bootstrap 95% AUROC interval | 0.588-0.910 |
| Average precision | 0.961 |
| No-skill average precision | 0.909 |
| Sensitivity | **0.510 (51/100 stains)** |
| Sensitivity Wilson 95% interval | 0.413-0.606 |
| Specificity | **0.900 (9/10 normal references)** |
| Specificity Wilson 95% interval | 0.596-0.982 |
| Balanced accuracy | **0.705** |
| False positives | 1 |
| False negatives | 49 |
| Runtime on Tesla T4 | 8,700 seconds (2 h 25 min) |

The validation-to-final reduction from AUROC 0.847 to 0.773 and sensitivity 60% to 51% is a normal
generalization gap and shows why the frozen final evaluation was necessary. The final result remains
above random ranking, but the confidence interval is wide because only ten normal reference images
are available.

## Error analysis

The only false alarm was `test/good/57.jpg`, which contains a pronounced diagonal fold and shading.
The model treats this structural variation as anomalous. Several low-scoring stain-labelled images
appear visually indistinguishable from normal fabric or contain only very faint marks. This suggests
that stain visibility and source-label noise contribute to the 49 false negatives. Those samples were
retained; excluding difficult images after observing model scores would bias the result.

## Interpretation

This run supports a portfolio claim of a self-supervised fabric-stain detection prototype with a
frozen image-level evaluation. It does not establish pixel-accurate segmentation, chemical-stain
classification, or production-ready 90% specificity. A subsequent study should use additional
matched normal images, manually reviewed image labels, pixel annotations, and a separate untouched
test set.
