# Printed-label real-defect audit

## Scope

This audit was completed before running the trained anomaly model on the real
test set. It covers 45 photographs from nine defective physical labels and the
ten previously reserved normal test photographs. Five photographs were taken
for every physical label under capture conditions B1--B5.

## Raw data

| Category | Physical labels | Raw photographs | Registered for model-only evaluation |
|---|---:|---:|---:|
| Missing/covered print simulation | M01--M03 | 15 | 15 |
| Ink smudge | S01--S03 | 15 | 15 |
| Tear | T01--T03 | 15 | 13 |
| Normal test | N11--N12 | 10 | 10 |
| **Total** | **11** | **55** | **53** |

All raw files are full-resolution JPEG photographs. The 45 defect filenames are
unique, every defect label has B1--B5, and no duplicate images were found. The
softest image is `M01_B5_103.jpg`; it remains readable and is retained.

The M01--M03 samples use white paper overlays. They are therefore simulations
of covered or missing printed content, not examples of ink physically failing
to transfer during printing. Results must use the description
"missing/covered print simulation."

## Registration audit

Registration uses printed corner fiducials and produces a 1063 x 650 canonical
view. A category-blind three-marker fallback is allowed when one fiducial is
damaged. Reflections are rejected, candidate fiducials must have comparable
sizes, and a fallback must improve reference agreement by a material margin
before it replaces a four-marker result.

| Category | Four-marker | Four-marker retained at low content similarity | Three-marker fallback | Failed |
|---|---:|---:|---:|---:|
| Missing/covered print | 12 | 3 | 0 | 0 |
| Smudge | 15 | 0 | 0 | 0 |
| Tear | 5 | 0 | 8 | 2 |

The registered contact sheets were inspected visually. Every retained image is
upright, contains the intended label region, and preserves the visible defect.

`T01_B3_082.jpg` and `T01_B4_085.jpg` are excluded from model-only image and
segmentation metrics because the tear removes enough right-side geometry that
only two reliable fiducials remain. Their registration failures should also be
reported separately as two successful rejects by an end-to-end inspection
pipeline, because a production system must reject an item that cannot be
registered. They must not be counted as DTU-Net detections.

## Statistical interpretation

The 43 registered defect photographs represent nine physical defects, not 43
independent defects. The final report must provide both per-image metrics and
per-physical-label results aggregated across B1--B5. All captures of a physical
label remain in the test set, so there is no train/test leakage.

## Locked test preparation

Ground-truth masks were created and reviewed without viewing DTU-Net predictions.
The 53-image manifest, masks, and file hashes are frozen in
`printed_label_test_locked_v1.zip`.

The remaining steps are:

1. Run the model once with the already selected checkpoint, `t = 50`, seed
   230274, pixel threshold 0.0235065464, and image threshold 0.0308046471.
2. Report model-only metrics for the 53 registered images and separate
   end-to-end rejection accounting for the two registration failures.
