# Printed-label first locked real-test evaluation

## Outcome

The 2,000-step DTU-Net/Tsimplex model does **not** work as a general detector for
the three printed-label defect categories in this pilot. It detects large tears
well, but it misses every covered-print and smudge photograph at the threshold
fixed before the real test.

This is the first and only evaluation with this checkpoint and frozen
configuration on the locked real test set. The result must not be replaced by a
threshold selected from these test data.

## Protocol verification

- Author source commit: `dc4a9bd2a2a5b1c31223daab4bdfea3f6a5b2990`
- Checkpoint: 2,000 optimizer steps
- Test images seen by the model: 53
- Registered normal images: 10 from physical labels N11 and N12
- Registered anomaly images: 43 repeated captures of nine physical defects
- Partial-diffusion distance: `t = 50`
- Seed: 230274
- Pixel threshold: 0.0235065464
- Image score: 99.5th percentile of residual pixels in the label ROI
- Image threshold: 0.0308046471
- GPU: Tesla T4

The distance, seed, score definition and thresholds were selected using normal
validation photographs and synthetic validation defects. The real images and
their masks were not used to select them.

## Aggregate model-only results

| Metric | Result |
|---|---:|
| Image AUROC | 0.512 |
| Image average precision | 0.866 |
| Sensitivity | 0.256 (11/43) |
| Specificity | 0.900 (9/10) |
| Precision | 0.917 (11/12) |
| F1 | 0.400 |
| Balanced accuracy | 0.578 |
| Pixel AUROC | 0.614 |
| Pixel average precision | 0.200 |
| Mean Dice on defective images | 0.160 |
| Mean IoU on defective images | 0.113 |
| Mean inference time | 4.00 s/image |

The high aggregate average precision should not be read alone: anomalies are
43/53, giving a prevalence baseline of 0.811. AUROC is close to random because
covered-print scores are generally lower than normal scores and smudge scores
overlap them.

## Category results

| Category | Images | Physical defects | Image sensitivity | AUROC versus normals | Mean Dice | Pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Covered/missing-print simulation | 15 | 3 | 0.000 | 0.173 | 0.001 | 0.502 |
| Smudge | 15 | 3 | 0.000 | 0.480 | 0.028 | 0.572 |
| Tear | 13 | 3 | 0.846 | 0.938 | 0.498 | 0.840 |

All three tear labels were detected in a majority of their registered captures:
T01 in 2/3, T02 in 4/5 and T03 in 5/5. No capture from M01--M03 or S01--S03 was
detected. Normal label N11 produced one false positive in condition B5; N12
produced none.

## Registration-stage result

Two additional severe T01 photographs could not be registered because only two
reliable fiducials remained. A production inspection pipeline can correctly
reject these inputs before DTU-Net, raising the combined image sensitivity from
11/43 to 13/45 = 0.289 while specificity remains 0.900. This is a hybrid
pipeline result; the two rejects are not DTU-Net detections.

## Failure interpretation

The reconstruction preserves the white cover and ink marks instead of replacing
them with the expected clean printed content. Their residual scores therefore
remain similar to, or below, normal reconstruction variation. Large tears expose
the background over a wide area and produce a much stronger residual, which
explains the clear tear separation.

The current image score also uses a high residual quantile over the whole label.
That statistic is suitable for large damage but can dilute small local smudges.
Changing it now would be post-test tuning. A region-aware or connected-component
score may be developed using a new validation set, followed by evaluation on new
physical labels.

## Defensible claim

The experiment supports a limited claim: the adapted pipeline detects and
segments large tears on this fixed printed-label design. It does not support a
claim of reliable covered-print or smudge detection, general printed-material
inspection, or successful reproduction of the paper's reported performance.
