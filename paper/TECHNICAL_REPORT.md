# Reproducing Diffusion-Based Anomaly Segmentation and Extending It to Printed-Label Inspection

**Hakam M.R.A. and Himasara W.V.M.J.**

Working technical report — 11 September 2026

## Abstract

This report studies the reproducibility and application of the self-supervised anomaly-segmentation method proposed by Kumar et al. at WACV 2025. We integrated a pinned version of the authors' Dynamic Transformer UNet and Tsimplex diffusion path, documented compatibility changes, and conducted reduced experiments on MVTec AD leather, a matched grayscale fabric-stain cohort, and a new fixed-design printed-label dataset. A 2,000-step leather run achieved 0.598 image AUROC, 0.666 pixel AUROC and 0.086 mean Dice. A fabric-stain run achieved 0.536 image AUROC and 0.270 sensitivity at 0.900 specificity. The printed-label model achieved 0.512 overall image AUROC and 0.256 sensitivity at 0.900 specificity on a locked 53-image test. Performance was category-dependent: tear-versus-normal AUROC was 0.938 with 0.498 mean tear Dice, while all covered-print and smudge captures were missed. These reduced experiments did not reproduce the paper's reported performance. We retain the negative results and identify large-tear inspection as the only supported capability of the current printed-label pilot.

## 1. Research questions

1. Can the inspected author implementation be executed reproducibly on an accessible Tesla T4 environment?
2. Does a reduced training schedule produce useful anomaly segmentation on MVTec AD leather?
3. Does the same implementation transfer to grayscale fabric-stain inspection using normal-only training?
4. Can a controlled printed-label application improve experimental validity through fixed acquisition, registration, masks and simple baselines?

## 2. Method attribution

Kumar et al. proposed the diffusion method, Tsimplex noise integration and Dynamic Transformer UNet. Our work covers source inspection, compatibility adaptation, experiment design, reduced training and evaluation, failure analysis, and the proposed printed-label application. We do not claim authorship of DTU-Net or the underlying diffusion method.

The inspected model configuration uses 224 × 224 three-channel inputs, 16 × 16 patches, embedding width 384, depth 12, six attention heads, SPE patch embedding, DMHA, DAFF/HFF components and output refinement. The model contains 28,863,812 parameters. Tsimplex uses six octaves, frequency 64 and persistence 0.9 with a 1,000-step cosine schedule.

## 3. Reproducibility controls

- Author source commit pinned to `dc4a9bd2a2a5b1c31223daab4bdfea3f6a5b2990`.
- Five required source files checked using recorded SHA-256 hashes.
- Deterministic dataset splits and random seeds.
- Anomalous test images excluded from optimization.
- Thresholds calibrated using held-out normal images.
- Resumable checkpoints containing model, optimizer, configuration and history.
- Locked test evaluation performed after protocol decisions were recorded.

## 4. Experiments

### 4.1 Integration check

The author model completed forward prediction, L2 diffusion loss, backpropagation, two optimizer updates and an eight-step partial reconstruction on a Tesla T4. This established executability only.

### 4.2 MVTec AD leather

The reduced experiment used 196 normal training images and 49 normal validation images. Training ran for 2,000 optimizer steps with batch size two. The official 124-image test set contained 32 normal and 92 defective images. A 99.5th-percentile validation residual calibrated the binary threshold before loading test masks.

### 4.3 Fabric stains

The downloaded aggregate dataset mixed sources, resolutions and colour modes. We selected a matched grayscale domain containing 68 normal and 398 stain images, excluded all processed images, and checked exact hashes for duplicates. The split used 48 training normals, ten validation normals, ten test normals and a seed-fixed subset of 100 stain test images. Because the cohort supplied no pixel masks, the primary metric was image AUROC.

### 4.4 Printed-label application

The application uses separately printed physical labels assigned to training, validation, normal-test, covered-print, smudge and tear groups. Four corner markers support perspective registration, with a category-blind three-marker fallback when a defect damages one corner. The training set contains 40 normal photographs from eight physical labels across five balanced capture conditions. DTU-Net/Tsimplex trained for 2,000 optimizer steps with batch size two using mild lighting and registration augmentation. The locked test contains ten normal photographs and 43 registered anomaly photographs representing nine physical defects. Masks were created before model inference. Two further severe-tear photographs could not be registered and are reported separately as preprocessing rejects.

## 5. Results

| Experiment | Image AUROC | Pixel AUROC | Mean Dice | Other result |
|---|---:|---:|---:|---|
| MVTec leather | 0.598 | 0.666 | 0.086 | 0.0081 normal pixel FPR |
| Fabric stains | 0.536 | — | — | 0.270 sensitivity, 0.900 specificity |
| Printed labels, all categories | 0.512 | 0.614 | 0.160 | 0.256 sensitivity, 0.900 specificity |
| Printed-label tears | 0.938 versus normals | 0.840 | 0.498 | 11/13 registered tear captures detected |

The leather experiment showed limited pixel-level ranking signal but poor mask overlap. The fabric experiment ranked normal and stained images only slightly above chance. Fabric average precision was 0.938, but the test prevalence was 100/110 = 0.909, making that value uninformative by itself. The printed-label run completed all 2,000 steps with finite losses and a final-20 mean of 0.0685. A pre-test comparison using only normal validation photographs and controlled digital defects selected distance 50, pixel threshold 0.02351 and image threshold 0.03080. On the locked real test, the model produced 11 true positives, nine true negatives, one false positive and 32 false negatives. All 11 model detections were tears. All three physical tear labels were detected in a majority of their registered captures, but no covered-print or smudge capture was detected. The aggregate average precision of 0.866 is only modestly above the anomaly prevalence of 43/53 = 0.811 and does not override the near-random overall AUROC.

## 6. Failure analysis

Normal and defective reconstructions remained visibly noisy at partial diffusion distance 250. Residual maps therefore responded to reconstruction texture as well as defects. The fabric experiment also resized complete 1984 × 1488 images to 224 × 224 after training on resized 512-pixel crops, introducing a scale mismatch. Only 48 normal source images were used for fabric training. Both experiments used 2,000 optimizer steps rather than the paper's substantially longer schedule, and the reduced checkpoint did not use a long-running EMA model.

Post-evaluation fabric score exploration found a best AUROC of 0.665 using a 2 × 2 block-pooled maximum, but this statistic was selected after viewing the test result and is reported only as diagnosis. It does not replace the locked AUROC of 0.536.

For printed labels, reconstruction retained the white covers and ink marks rather than restoring the expected clean print. Their residual distributions overlapped normal captures. Large tears exposed a wide region of background and created much stronger residuals. The whole-label 99.5th-percentile image score was therefore effective for large damage but insensitive to smaller local defects. Any new score or region-aware threshold must be developed on new validation data and tested on new physical labels; changing it against this completed test would invalidate an independent-test claim.

## 7. Threats to validity

- Reduced training cannot establish full paper reproducibility.
- Architecture and configuration details are inconsistent across the paper and repository entry points.
- One stochastic reconstruction per image introduces variance.
- The fabric test has only ten independent normal images.
- Fabric masks are unavailable, preventing segmentation metrics.
- Whole-image fabric resizing differs from training crop scale.
- The printed-label anomalies comprise only nine physical defects with repeated captures.
- White overlays simulate covered or missing print rather than a true printer ink-transfer failure.
- Two severe-tear photographs could not be registered for model inference.

## 8. Remaining work

The first locked DTU-Net evaluation is complete. The remaining application work is to run the registered template baseline under the same split, develop any region-aware extension using new validation defects, evaluate the extension on new physical labels, and package a camera demonstration. Current evidence supports large-tear detection on one fixed label design and does not support general printed-label anomaly detection.

## References

Kumar, K., Chakraborty, S., Mahapatra, D., Bozorgtabar, B., and Roy, S. “Self-Supervised Anomaly Segmentation via Diffusion Models with Dynamic Transformer UNet.” WACV, 2025.

MVTec Software GmbH. “MVTec Anomaly Detection Dataset.”

Saleem, S. “Fabric Defects Dataset.” Mendeley Data, Version 3, 2024. DOI: 10.17632/663j22s43c.3.
