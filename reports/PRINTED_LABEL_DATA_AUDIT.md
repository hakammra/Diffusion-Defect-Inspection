# Real printed-label training data audit

## Scope

This audit covers the first normal-only capture set for the proposed fixed-design
printed-label inspection application. It is a data-preparation result and does not
measure anomaly-detection performance.

## Raw data

- 40 JPEG photographs at 3000 x 4000 pixels.
- Eight physical labels (`N01` through `N08`).
- Five capture conditions per physical label (`B1` through `B5`).
- Eight photographs in every capture condition.
- No exact duplicates or near-duplicate pairs were detected by the audit.
- The lowest relative sharpness score belonged to `N02_B4_026.jpg`; its label text,
  barcode and fiducials remained readable, so it was retained.

The split is leakage-safe at this stage because all eight physical labels are assigned
only to training. Later validation and test photographs must use different physical
labels.

## Registration

The preprocessing code detects the four black fiducials and estimates a perspective
mapping to the 1063 x 650 reference-label coordinate system. All 40 photographs were
registered successfully. This removes most of the background and aligns corresponding
printed regions before model training.

The training notebook performs an aspect-preserving resize with padding and applies
only mild brightness, contrast and rotation jitter. It does not mirror labels because
mirrored text and barcodes are not valid normal examples.

## Prepared artifacts

- Local training archive: `output/datasets/printed_label_train_v1.zip`
- Registered-image manifest: `output/datasets/printed_label_train_v1/manifest.csv`
- Kaggle notebook: `notebooks/10_printed_label_train_kaggle.ipynb`

The generated archive and raw photographs are excluded from Git because they are
experiment data. The audit code and notebook remain reproducible from the private raw
images.

## Remaining data

Normal-only training and validation capture are complete. Detection accuracy cannot
be calculated until the following independently assigned physical labels are captured:

- `N11`-`N12`: normal test images.
- `M01`-`M03`: missing-print test images.
- `S01`-`S03`: smudge test images.
- `T01`-`T03`: tear test images.

Defect masks must be drawn without inspecting model predictions before the locked test.

## Validation capture update

The normal validation set was subsequently captured and audited:

- Ten 3000 x 4000 JPEG photographs.
- Two independent physical labels (`N09` and `N10`).
- Five balanced capture conditions per label.
- No exact or near duplicates detected.
- All ten images registered successfully to 1063 x 650 pixels.

The prepared validation archive is
`output/datasets/printed_label_validation_v1.zip`. It is excluded from Git together
with the raw data. The calibration notebook is
`notebooks/11_printed_label_validate_kaggle.ipynb`.

## Normal test capture update

Ten normal test photographs from independent physical labels `N11` and `N12` were
received. Each label has one image in capture conditions `B1` through `B5`. All files
are 3000 x 4000 JPEGs, no duplicates were detected, and all ten passed automated
fiducial registration. A duplicated terminal photo number was corrected from
`N12_B2_53.jpg` to `N12_B2_54.jpg`.

Only structural and automated registration checks were performed. Model inference and
visual outcome review remain locked until the final validation-only distance selection
is saved. Missing-print, smudge and tear test photographs are still outstanding.
