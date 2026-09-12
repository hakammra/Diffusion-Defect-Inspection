# Research and implementation plan

## Scope

Implement printed-label defect inspection based on the selected WACV 2025 diffusion paper. Start with a working, inexpensive baseline. Keep paper reproduction and the new application as separately evaluated work.

## Milestones and completion criteria

| Stage | Evidence needed | Status |
| --- | --- | --- |
| 0. Local foundation | CPU baseline, unit checks, synthetic end-to-end artifacts | Implemented; inspect run artifacts |
| 1. GPU access | Saved accelerator name, CUDA check, successful optimization step | Passed on Kaggle Tesla T4 |
| 2. Author model integration | Correct input/output shapes, finite loss, successful backward pass, short reconstruction, documented patches | Passed on Colab Tesla T4; see `artifacts/author_integration/integration_report.json` |
| 3. Reduced reproduction | One benchmark category; controlled splits; baseline and diffusion metrics; paper-setting differences listed | Reduced evaluation completed; paper result not reproduced; differences documented |
| 4. Application data | Real fixed-design label photos, masks, capture protocol, independent splits | Complete; 53 registered test images locked, plus two registration rejects |
| 5. Application comparison | Template and diffusion results; region-aware extension ablation; failure review | First DTU-Net test complete; baseline and extension comparison pending |
| 6. Portfolio package | Reproducible commands, demo, video, contribution statement, actual metrics | Pending |

An additional fabric-stain experiment was completed as a secondary application benchmark. Its 2,000-step training run converged, but locked image-level evaluation achieved only 0.536 AUROC and 0.270 sensitivity at 0.900 specificity. See `reports/FABRIC_STAIN_TRAINING.md` and `reports/FABRIC_STAIN_EVALUATION.md`. This negative result does not replace the printed-label application.

## Data protocol

Synthetic images are pipeline checks only. For real evaluation, collect multiple physical printed labels and multiple capture sessions under normal lighting and pose variation. Keep all captures from a physical label in one split; avoid near-duplicate sessions across splits. Use normal images for the denoising training set. Keep normal validation images for a normal-only threshold protocol, or explicitly disclose a labelled-validation alternative.

Suggested initial collection target, subject to effort and pilot findings: 100-200 normal photographs for training; a separate normal validation set; a test set containing both normal images and missing-print, smudge, and tear examples. These counts are a starting plan, not an established sample-size justification. Annotate test defect regions and retain raw photographs.

Print one fixed label layout first. A wrong but visually valid serial number is a semantic/OCR problem and is outside this initial visual-defect task.

## Baselines and proposed extension

- B0: aligned template residual with one validation-calibrated threshold.
- B1: registration followed by B0; quantify registration failures.
- B2: the reproduced diffusion method using its reconstruction residual.
- E1: text/background region-aware scoring or thresholds. This is a proposed extension to test, not a promised improvement.

Compare methods under the same test split. Tune only on validation data. Report average per-image segmentation scores on defective images, pooled pixel scores, and false positives on normal images separately. Retain continuous maps for AUROC; do not compute ranking metrics solely from binary masks.

## Compute plan

Use the laptop for preprocessing, baseline development, and small checks. Use a Kaggle GPU notebook for model training if the account has accelerator access. Use a Colab GPU runtime as the fallback if Kaggle access is unavailable. Do not assume that a specific GPU or unlimited runtime is available. Save resumable checkpoints and record optimizer/RNG state, split manifest, configuration and package versions.

Do not install the upstream requirements blindly: the downloaded list contains platform-specific CUDA packages, a self-referential editable dependency, and experimental code with inconsistent entrypoints. Build an isolated runtime for the selected model path.

## Immediate next run

Preserve the first locked DTU-Net result: overall image AUROC 0.512, sensitivity 0.256 and specificity 0.900. The tear subset achieved 0.938 AUROC versus normals and 0.498 mean Dice, while all covered-print and smudge captures were missed. Next, run the registered template baseline under the same split and use new validation data for any region-aware scoring extension. Do not tune against the completed locked test and then report the result as independent.

## Attribution and CV claims

The method is by Kumar et al. Our future CV statement should identify a reproduction/application contribution, not invention of DTU-Net. Use quantitative improvements only after independent evaluation. Record Hakam's and Himasara's individual work accurately.

Example structure to fill after completion: "Reproduced and adapted a WACV 2025 diffusion-based anomaly-segmentation method for printed-label inspection; evaluated against template subtraction on [N] held-out images and built a [demo type] inference application." Do not fill in unmeasured metrics.
