# Fabric stain V2-V5 audit

## What improved

The strongest verified change was reducing the partial-diffusion distance from 250 to 50.
On the same 10-normal/100-stain cohort, image AUROC increased from 0.536 in V1 to 0.809 in V2,
while measured latency decreased from 17.72 to 3.71 seconds per image. Two-sample averaging and
detrending raised the best observed ranking result to AUROC 0.819 in V3. V5 produced larger,
more contiguous masks and recovered edge responses that V4's 12-pixel margin removed.

| Version | Main change | AUROC | AP | Operating result | Approx. seconds/image |
|---|---|---:|---:|---|---:|
| V1 | `t=250`, raw residual | 0.536 | 0.938 | 27/100 stains; 9/10 normals | 17.72 |
| V2 | `t=50`, smoothing + IQR normalization | 0.809 | 0.978 | 47/100 stains; 10/10 normals | 3.71 |
| V3 | `t=50`, two samples + detrending | **0.819** | **0.980** | 46/100 stains; 9/10 normals | 6.84 |
| V4 | reflection padding + 12-pixel margin | 0.759 | 0.973 | strict: 60/100 stains; 9/10 normals | 7.47 |
| V5 | two-scale detrending, no margin, shape filter | 0.810 | 0.979 | stain rule: 68/100 stains; 9/10 normals | 6.84 |

The values above come from `evaluation_report*.json`. They supersede manually assembled summary
tables. The V5 artifact reports 53 strict detections, 68 stain-rule detections, and 69 total-defect
detections. It does not report 61 strict detections or 76 recovered stains.

## Interpretation limits

V2-V5 all evaluate the same 10 normal and 100 stain test images. Once V2 was inspected and V3-V5
were designed in response, that cohort became development data. V3-V5 must therefore be described
as exploratory/post-test improvements, not independent locked tests.

The source cohort has image-level labels but no pixel masks. A visually plausible predicted mask is
useful qualitative evidence, but its area, boundary, Dice, and IoU cannot be called accurate without
human ground-truth masks. The V5 aspect-ratio rule is also a heuristic: the data labels only
`normal` and `stain`, so the claimed chemical-stain/crease distinction has not been independently
validated.

Average precision is inflated by the evaluation prevalence: 100 of 110 images are stains, giving a
no-skill AP baseline of 0.909. AUROC, sensitivity, specificity, and balanced accuracy are more useful
for this cohort. Specificity is especially uncertain because it is based on only 10 normal images.

Scores such as the V1 residual mean and the V5 normalized mean are on different scales and should
not be compared numerically. The meaningful comparisons are ranking and operating metrics.

## Next experiment

Freeze V5 as the current candidate and stop adjusting it on the original 110-image cohort. Run
`notebooks/14_fabric_stain_tiled_validation_kaggle.ipynb`, which compares whole-image and tiled
inference on the 10 validation normals plus 30 stain images excluded from V1-V5. The tiled path uses
512-pixel source crops to match training, instead of shrinking each complete 1984x1488 image to
224x224. It selects one scoring rule and one threshold; those settings should then be run once on a
separate stain cohort.

The completed tiled validation selected the mean of the two highest tile scores. It achieved AUROC
0.847, 60% sensitivity, 100% specificity, and balanced accuracy 0.800 on 10 validation normals and
30 previously unused stains. The frozen cutoff is 16.47188949584961. The next notebook,
`notebooks/15_fabric_stain_tiled_final_evaluate_kaggle.ipynb`, applies those settings to 100 further
unseen stains. Its ten normal references were used in earlier exploratory versions, so the final
report records that limitation explicitly.

The frozen final run completed without changing the method or threshold. On 100 stain images
excluded from every preceding run and ten reused normal references, it obtained AUROC 0.773,
average precision 0.961, sensitivity 0.510, specificity 0.900, and balanced accuracy 0.705. See
`reports/FABRIC_STAIN_TILED_FINAL_EVALUATION.md` for confidence intervals and error analysis.

Further work with the largest likely payoff:

1. Continue training from 2,000 to 5,000 and 10,000 steps, choosing the checkpoint only on the
   development validation cohort.
2. Add more matched-domain normal images. The other defect-free folders use different dimensions
   and colour modes, so mixing them without a domain check can make the evaluation easier for the
   wrong reason.
3. Annotate 30-50 stain masks and report pixel AUROC, Dice, and IoU at a validation-selected cutoff.
4. Compare DTU-Net with a strong feature-memory baseline such as PatchCore under the same split.
5. After all decisions are frozen, evaluate a disjoint final cohort once and report bootstrap
   confidence intervals.
