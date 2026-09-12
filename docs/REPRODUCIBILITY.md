# GPU Reproduction Guide

This guide outlines the steps to run the training and evaluation notebooks using GPU resources (such as a Kaggle Tesla T4 / P100 or Google Colab).

---

## 1. Environment Requirements

- **GPU Accelerator**: NVIDIA GPU with >= 12 GB VRAM (Tesla T4, P100, or RTX 3080+).
- **Python**: 3.10 or newer.
- **Internet Access**: Must be enabled in the hosted session to download the pinned author commit and pretrained dependencies.

---

## 2. Notebook Execution Workflow

The notebooks in `notebooks/` are organized in logical execution order:

### Phase 1: Environment & Foundational Paper Reproduction
1. `00_gpu_check.ipynb`: Verifies CUDA runtime, device memory, and software dependencies.
2. `01_cpu_baseline.ipynb`: Quick CPU-only sanity check with template subtraction.
3. `02_author_model_check.ipynb`: Verifies the author's DTU-Net architecture and Tsimplex diffusion definitions using pinned commit `dc4a9bd2a2a5b1c31223daab4bdfea3f6a5b2990`.
4. `03_mvtec_leather_train.ipynb`: Trains DTU-Net on MVTec AD leather for 2,000 steps.
5. `04_mvtec_leather_evaluate.ipynb`: Evaluates pixel and image-level anomaly metrics on official leather test images.

### Phase 2: Printed Label Inspection Application
6. `05_printed_label_train.ipynb`: Trains the diffusion model on 40 perspective-registered normal label captures.
7. `06_printed_label_evaluate.ipynb`: Evaluates performance on the 53-image locked real test cohort (tears, smudges, covered print).

### Phase 3: Fabric Stain Anomaly Inspection
8. `07_fabric_stain_train.ipynb`: Trains the Vision Transformer diffusion model on 48 normal fabric patches.
9. `08_fabric_stain_evaluate_baseline.ipynb`: Baseline single-scale diffusion residual evaluation (V1).
10. `09_fabric_stain_evaluate_v9_dual_pathway.ipynb`: **Flagship V9 evaluation notebook** implementing the Dual-Pathway Hybrid Engine (DiT residual + weave-suppressed photometric bandpass + crease disambiguation).

---

## 3. Dataset Setup on Kaggle

When running on Kaggle, upload the relevant dataset zip archive as a private dataset and attach it to your notebook session:

- For Leather: Attach the standard MVTec AD leather subset.
- For Printed Labels: Upload and attach `printed_label_train_v1.zip` and test packages.
- For Fabric Stains: Upload and attach `fabric_stain_pilot.zip`.

Each notebook automatically discovers datasets under `/kaggle/input/` and saves evaluation reports, curves, and visualization archives under `/kaggle/working/`.
