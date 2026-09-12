# Fabric stain experiment

## Why the full archive is not one training domain

The downloaded Fabric Defects Dataset is an aggregation of several sources. A local audit found different colour modes, camera domains and resolutions in the same top-level folders. Training on every defect-free file and testing on every defective file would let source and acquisition differences dominate the anomaly score.

The `processed` files in the hole, horizontal and vertical folders are derived mask-like outputs. They are excluded from the first experiment. Processed files also occur in the hole folder, not only horizontal and vertical.

## Selected matched cohort

The first controlled experiment uses the grayscale stain cohort because its normal and defective images share the same dimensions and appearance domain:

| Split | Images | Training access |
|---|---:|---|
| Train normal | 48 | Used for optimization |
| Validation normal | 10 | Reserved for threshold selection |
| Test normal | 10 | Locked until evaluation |
| Test stain | 398 | Locked until evaluation |

All selected images are grayscale and either 1984 × 1488 or 1488 × 1984 pixels. Exact SHA-256 duplicate checking found no duplicate files in the selected cohort. The pack contains no processed files.

Training samples random 512 × 512 normal crops and resizes them to the DTU-Net input size of 224 × 224. Grayscale values are repeated over three channels to match the verified author model configuration.

## Run the Kaggle pilot

1. Create a Kaggle notebook and enable a GPU.
2. Upload `output/datasets/fabric_stain_pilot.zip` as a private notebook input.
3. Import `notebooks/07_fabric_stain_pilot_train_kaggle.ipynb` into Kaggle.
4. Enable Internet so the notebook can retrieve the five pinned, hash-checked author source files.
5. Run all cells.
6. Download both `/kaggle/working/fabric_stain_latest.pt` and `/kaggle/working/fabric_stain_pilot_results.zip`.

The first notebook performs 100 optimizer steps. It checks the data path, model execution, finite loss and gradient stability and records GPU time and memory. It never loads test images into the training loop. Continue to a longer run only after inspecting this pilot.

The 100-step pilot passed on a Tesla T4: loss decreased from 0.4204 to 0.1712, the final 20-step mean was 0.1654, and peak allocated GPU memory was 0.675 GiB. `notebooks/08_fabric_stain_train_kaggle.ipynb` continues the same checkpoint to 2,000 total optimizer steps. Attach both the dataset ZIP and `fabric_stain_latest.pt`; the continuation refuses to start from step zero.

## Evaluation limits

This stain cohort contains image-level class labels but no pixel masks. It supports normal-versus-stain image AUROC and qualitative anomaly maps. A later segmentation evaluation must manually annotate a fixed subset of stain images before viewing model results on that subset.

This experiment tests one grayscale stain domain. It must not be described as a model that works on every fabric type or defect.
