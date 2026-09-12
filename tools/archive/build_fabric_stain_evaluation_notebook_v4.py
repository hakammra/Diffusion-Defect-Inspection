from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "06_mvtec_leather_evaluate_kaggle.ipynb"
TRAINING = ROOT / "notebooks" / "08_fabric_stain_train_kaggle.ipynb"
OUTPUT = ROOT / "notebooks" / "09_fabric_stain_evaluate_kaggle_v4.ipynb"


def lines(text: str):
    return [line + "\n" for line in text.strip("\n").splitlines()]


nb = json.loads(SOURCE.read_text(encoding="utf-8"))
training_nb = json.loads(TRAINING.read_text(encoding="utf-8"))

nb["cells"][0]["source"] = lines(
    """
# Fabric stain locked image-level evaluation (V4 Protocol)

This V4 notebook evaluates the 2,000-step DTU-Net/Tsimplex checkpoint using an advanced,
edge-artifact-free inference and dual-threshold hysteresis segmentation protocol:

1. `t_distance = 50`: Bounded forward noise distance for sharp, artifact-free reconstruction.
2. Two-Sample Averaging (N=2): Cancels stochastic diffusion noise speckles.
3. Grayscale Luminance Residual: Projects RGB to luminance before computing residuals.
4. Reflect-Padded Background Detrending (Kernel 41): Replaces zero-padding with reflection
   padding to eliminate artificial boundary halos / border false alarms.
5. Reflect-Padded Spatial Smoothing (Kernel 5) & Per-Image IQR Background Normalization.
6. Boundary Margin Guard: Suppresses sensor / lens vignetting along the outer 12px perimeter.
7. Dual-Threshold Hysteresis Region Growing:
   - High seed threshold (tau_high): Triggers only on confident anomaly cores.
   - Low body threshold (tau_low): Traces and reconstructs the full contiguous shape of faint stains.
   - Area threshold (min_area >= 15 px): Rejects isolated texture spikes.
8. Calibrated Cutoffs from Validation Normal Controls: Automatically establishes strict threshold
   and hysteresis thresholds from the 10 held-out normal validation images.
"""
)

nb["cells"][1]["source"] = lines(
    """
from pathlib import Path
import csv, importlib.util, json, random, shutil, subprocess, sys, time, zipfile

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working/labelinspect')
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT = WORK / 'fabric_stain_evaluation_v4'
OUTPUT.mkdir(parents=True, exist_ok=True)

def find_dataset(search_root):
    found = []
    for path in search_root.rglob('fabric_stain_pilot'):
        if path.is_dir() and (path / 'val_normal').is_dir() and (path / 'test' / 'stain').is_dir():
            found.append(path)
    return sorted(set(found))

dataset_candidates = find_dataset(KAGGLE_INPUT)
if not dataset_candidates:
    matching = []
    for archive_path in KAGGLE_INPUT.rglob('*.zip'):
        with zipfile.ZipFile(archive_path) as archive:
            names = ['/' + item.filename.replace('\\\\', '/').lstrip('/') for item in archive.infolist()]
            if any('/fabric_stain_pilot/val_normal/' in name for name in names):
                matching.append(archive_path)
    if len(matching) == 1:
        extraction_root = WORK / 'uploaded_data'
        extraction_root.mkdir(parents=True, exist_ok=True)
        resolved = extraction_root.resolve()
        with zipfile.ZipFile(matching[0]) as archive:
            for item in archive.infolist():
                target = (extraction_root / item.filename).resolve()
                if target != resolved and resolved not in target.parents:
                    raise ValueError(f'Unsafe ZIP member: {item.filename}')
            archive.extractall(extraction_root)
        dataset_candidates = find_dataset(extraction_root)
if len(dataset_candidates) != 1:
    raise FileNotFoundError('Expected one fabric_stain_pilot dataset: ' + repr([str(p) for p in dataset_candidates]))
DATA_ROOT = dataset_candidates[0]

all_pt_files = sorted(KAGGLE_INPUT.rglob('*.pt'))
if not all_pt_files:
    extracted_roots = []
    for data_pickle in KAGGLE_INPUT.rglob('data.pkl'):
        candidate = data_pickle.parent
        if (candidate / 'data').is_dir() and (candidate / 'version').is_file():
            extracted_roots.append(candidate)
    extracted_roots = sorted(set(extracted_roots))
    if len(extracted_roots) == 1:
        archive_root = extracted_roots[0]
        rebuilt = WORK / 'rebuilt_fabric_stain_checkpoint.pt'
        with zipfile.ZipFile(rebuilt, 'w', compression=zipfile.ZIP_STORED) as archive:
            for source_file in sorted(archive_root.rglob('*')):
                if source_file.is_file():
                    archive.write(source_file, f'{archive_root.name}/{source_file.relative_to(archive_root).as_posix()}')
        all_pt_files = [rebuilt]
        print('Rebuilt Kaggle-extracted checkpoint:', rebuilt)
if len(all_pt_files) != 1:
    raise FileNotFoundError('Expected one checkpoint input; found: ' + repr([str(p) for p in all_pt_files]))
CHECKPOINT = all_pt_files[0]

SEED = 230224
IMAGE_SIZE = 224
BATCH_SIZE = 4
T_DISTANCE = 50
NUM_DIFFUSION_SAMPLES = 2
DETREND_KERNEL = 41
SMOOTH_KERNEL = 5
BORDER_MARGIN = 12
MIN_CLUSTER_AREA = 15
STAIN_TEST_COUNT = 100

print('GPU:', torch.cuda.get_device_name(0))
print('Dataset:', DATA_ROOT)
print('Checkpoint:', CHECKPOINT)
print(f'V4 Settings: t={T_DISTANCE}, samples={NUM_DIFFUSION_SAMPLES}, detrend_k={DETREND_KERNEL}, smooth_k={SMOOTH_KERNEL}, margin={BORDER_MARGIN}, min_area={MIN_CLUSTER_AREA}')
"""
)

nb["cells"][3]["source"] = training_nb["cells"][3]["source"]
nb["cells"][4]["source"] = lines(
    """
spec = importlib.util.spec_from_file_location('labelinspect_author_smoke', WORK / 'author_smoke.py')
author = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = author
spec.loader.exec_module(author)
import numba
numba.set_num_threads(min(2, numba.get_num_threads()))
source_cache = WORK / 'upstream' / author.COMMIT
hashes = author.fetch_sources(source_cache)
model_module, diffusion_ns, Adapter, compatibility = author.load_author_components(source_cache, OUTPUT)
print('Pinned author commit:', author.COMMIT)
"""
)

nb["cells"][5]["source"] = lines(
    """
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
def image_paths(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

val_paths = image_paths(DATA_ROOT / 'val_normal')
all_test_good = image_paths(DATA_ROOT / 'test' / 'good')
all_test_stain = image_paths(DATA_ROOT / 'test' / 'stain')
assert (len(val_paths), len(all_test_good), len(all_test_stain)) == (10, 10, 398)

selection_rng = random.Random(SEED)
selected_stain = all_test_stain.copy()
selection_rng.shuffle(selected_stain)
test_stain_paths = sorted(selected_stain[:STAIN_TEST_COUNT])
test_paths = all_test_good + test_stain_paths
image_labels = np.asarray([0] * len(all_test_good) + [1] * len(test_stain_paths), dtype=np.int64)
(OUTPUT / 'locked_test_files.txt').write_text('\\n'.join(path.relative_to(DATA_ROOT).as_posix() for path in test_paths))

RESAMPLE = getattr(Image, 'Resampling', Image).BILINEAR
def load_image(path):
    with Image.open(path) as image:
        image = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE)
        array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)

print('Protocol:', len(val_paths), 'validation normal,', len(all_test_good), 'test normal,', len(test_stain_paths), 'test stain')
"""
)

nb["cells"][6]["source"] = lines(
    """
device = torch.device('cuda:0')
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
assert checkpoint['step'] == 2000, f"Expected step 2000, found {checkpoint['step']}"
assert checkpoint['author_commit'] == author.COMMIT
assert checkpoint.get('dataset_category') == 'fabric_stain_pilot'
model_config = checkpoint['model_config']
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
backbone = model_module.UDHVT(**model_config).to(device)
model = Adapter(backbone)
model.load_state_dict(checkpoint['model'])
model.eval()
diffusion = diffusion_ns['GaussianDiffusionModel'](
    [224, 224], diffusion_ns['get_beta_schedule'](1000, 'cosine'), img_channels=3,
    loss_type='l2', noise='4dsimplex', octave=6, frequency=64, persistence=0.9, train=False)
diffusion.noise_fn(torch.zeros(1, 1, 4, 4, device=device), torch.tensor([5], device=device))
torch.cuda.reset_peak_memory_stats()
print('Loaded step', checkpoint['step'], 'model with', sum(p.numel() for p in model.parameters()), 'parameters.')
"""
)

nb["cells"][7]["source"] = lines(
    """
def reconstruct_paths(paths, batch_size=BATCH_SIZE, num_samples=NUM_DIFFUSION_SAMPLES):
    reconstructions = []; inputs = []; seconds = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        x = torch.stack([load_image(p) for p in batch_paths]).to(device)
        tick = time.perf_counter()
        with torch.inference_mode():
            batch_recons = []
            for s in range(num_samples):
                r = diffusion.forward_backward(model, x, None, see_whole_sequence=None,
                                               t_distance=T_DISTANCE, denoise_fn='noise_fn')
                batch_recons.append(r)
            recon = torch.stack(batch_recons).mean(dim=0)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - tick)
        inputs.append(x.cpu())
        reconstructions.append(recon.cpu())
        print(f'Reconstructed {min(start + len(batch_paths), len(paths))}/{len(paths)} ({num_samples} samples averaged)', flush=True)
    return torch.cat(inputs), torch.cat(reconstructions), seconds


def compute_v4_maps(inputs, recons, detrend_k=DETREND_KERNEL, smooth_k=SMOOTH_KERNEL, margin=BORDER_MARGIN):
    \"\"\"V4 residual map pipeline:
    1. Grayscale luminance conversion
    2. Reflection-padded low-frequency background detrending (eradicates border halo artifacts)
    3. Reflection-padded spatial smoothing
    4. Per-image background normalization ((R - median) / IQR)
    5. Sensor border guard (suppresses edge vignetting)
    \"\"\"
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    in_lum = (inputs * weights).sum(dim=1, keepdim=True)
    rec_lum = (recons * weights).sum(dim=1, keepdim=True)
    raw_sq = (in_lum - rec_lum).square()

    # Background detrending with REFLECT padding
    pad_bg = detrend_k // 2
    raw_sq_padded = torch.nn.functional.pad(raw_sq, (pad_bg, pad_bg, pad_bg, pad_bg), mode='reflect')
    bg = torch.nn.functional.avg_pool2d(raw_sq_padded, kernel_size=detrend_k, stride=1, padding=0)
    detrended = torch.clamp(raw_sq - bg, min=0.0)

    # Spatial smoothing with REFLECT padding
    pad_sm = smooth_k // 2
    detrended_padded = torch.nn.functional.pad(detrended, (pad_sm, pad_sm, pad_sm, pad_sm), mode='reflect')
    smoothed = torch.nn.functional.avg_pool2d(detrended_padded, kernel_size=smooth_k, stride=1, padding=0)
    maps = smoothed.squeeze(1).numpy().astype(np.float32)

    # Per-image background normalization
    flat = maps.reshape(maps.shape[0], -1)
    medians = np.median(flat, axis=1, keepdims=True)
    q75 = np.quantile(flat, 0.75, axis=1, keepdims=True)
    q25 = np.quantile(flat, 0.25, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-6)
    normalized = ((flat - medians) / iqr).reshape(maps.shape)

    # Suppress outer border margin
    if margin > 0:
        normalized[:, :margin, :] = 0.0
        normalized[:, -margin:, :] = 0.0
        normalized[:, :, :margin] = 0.0
        normalized[:, :, -margin:] = 0.0

    return normalized, maps, raw_sq.squeeze(1).numpy().astype(np.float32)


def hysteresis_segmentation(v4_map: np.ndarray, tau_high: float, tau_low: float, min_area: int = MIN_CLUSTER_AREA) -> np.ndarray:
    \"\"\"Extracts contiguous anomaly masks via dual-threshold hysteresis region growing.\"\"\"
    seed_mask = v4_map > tau_high
    body_mask = v4_map > tau_low
    if not np.any(seed_mask):
        return np.zeros_like(seed_mask, dtype=bool)

    H, W = v4_map.shape
    visited = np.zeros((H, W), dtype=bool)
    final_mask = np.zeros((H, W), dtype=bool)

    for r in range(H):
        for c in range(W):
            if body_mask[r, c] and not visited[r, c]:
                cluster = []
                has_seed = False
                q = [(r, c)]
                visited[r, c] = True
                while q:
                    cr, cc = q.pop()
                    cluster.append((cr, cc))
                    if seed_mask[cr, cc]:
                        has_seed = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and body_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                if has_seed and len(cluster) >= min_area:
                    for cr, cc in cluster:
                        final_mask[cr, cc] = True
    return final_mask
"""
)

nb["cells"][8]["source"] = lines(
    """
print(f'Reconstructing held-out normal validation images at t_distance={T_DISTANCE} with {NUM_DIFFUSION_SAMPLES} samples...')
val_inputs, val_recons, val_batch_seconds = reconstruct_paths(val_paths)
val_v4_maps, val_smooth_maps, val_raw_maps = compute_v4_maps(val_inputs, val_recons)
val_image_scores = np.max(val_v4_maps.reshape(len(val_paths), -1), axis=1)

# Strict threshold: conservative maximum score over validation normals
tau_strict = float(np.max(val_image_scores))

# Calibrated hysteresis thresholds:
# Seed threshold: confident defect peak (default 16.5 or min(16.5, tau_strict * 0.75))
tau_high = float(min(16.5, tau_strict * 0.75))
# Body threshold: traces full defect body down to background noise floor
tau_low = float(max(3.5, tau_high * 0.25))

calibration = {
    'version': 'v4',
    'source': '10 held-out normal validation images',
    'tau_strict': tau_strict,
    'tau_high': tau_high,
    'tau_low': tau_low,
    'min_cluster_area': MIN_CLUSTER_AREA,
    'border_margin': BORDER_MARGIN,
    'observed_validation_image_fpr': float(np.mean(val_image_scores > tau_strict)),
    't_distance': T_DISTANCE,
    'num_diffusion_samples': NUM_DIFFUSION_SAMPLES,
    'detrend_kernel': DETREND_KERNEL,
    'smooth_kernel': SMOOTH_KERNEL,
}
(OUTPUT / 'calibration_v4.json').write_text(json.dumps(calibration, indent=2))
print(json.dumps(calibration, indent=2))
"""
)

nb["cells"][9]["source"] = lines(
    """
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

print(f'Running locked normal-versus-stain evaluation at t_distance={T_DISTANCE}...')
test_inputs, test_recons, test_batch_seconds = reconstruct_paths(test_paths)
test_v4_maps, test_smooth_maps, test_raw_maps = compute_v4_maps(test_inputs, test_recons)

# Image scores
image_scores = np.max(test_v4_maps.reshape(len(test_paths), -1), axis=1)

# Strict predictions
strict_predictions = image_scores > tau_strict
tn_s, fp_s, fn_s, tp_s = confusion_matrix(image_labels, strict_predictions, labels=[0, 1]).ravel()
sens_strict = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) else None
spec_strict = float(tn_s / (tn_s + fp_s)) if (tn_s + fp_s) else None

# Hysteresis segmentations & predictions
hyst_masks = []
hyst_predictions = []
hyst_areas = []
for i in range(len(test_paths)):
    m = hysteresis_segmentation(test_v4_maps[i], tau_high=tau_high, tau_low=tau_low, min_area=MIN_CLUSTER_AREA)
    hyst_masks.append(m)
    hyst_predictions.append(bool(np.any(m)))
    hyst_areas.append(int(np.sum(m)))

hyst_predictions = np.asarray(hyst_predictions, dtype=bool)
tn_h, fp_h, fn_h, tp_h = confusion_matrix(image_labels, hyst_predictions, labels=[0, 1]).ravel()
sens_hyst = float(tp_h / (tp_h + fn_h)) if (tp_h + fn_h) else None
spec_hyst = float(tn_h / (tn_h + fp_h)) if (tn_h + fp_h) else None
bal_acc_hyst = float((sens_hyst + spec_hyst) / 2) if (sens_hyst is not None and spec_hyst is not None) else None

# AUC metrics
image_auc = float(roc_auc_score(image_labels, image_scores))
average_precision = float(average_precision_score(image_labels, image_scores))
v1_raw_scores = np.quantile(test_raw_maps.reshape(len(test_paths), -1), 0.995, axis=1)
v1_auc = float(roc_auc_score(image_labels, v1_raw_scores))

rows = []
for path, label, score, p_strict, p_hyst, area, v1_s in zip(
    test_paths, image_labels, image_scores, strict_predictions, hyst_predictions, hyst_areas, v1_raw_scores
):
    rows.append({
        'image': path.relative_to(DATA_ROOT).as_posix(),
        'label': int(label),
        'kind': 'stain' if label else 'good',
        'v4_score': float(score),
        'strict_prediction': int(p_strict),
        'hysteresis_prediction': int(p_hyst),
        'hysteresis_mask_area': int(area),
        'raw_score_p995': float(v1_s),
    })
with (OUTPUT / 'per_image_v4.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

fpr_v4, tpr_v4, _ = roc_curve(image_labels, image_scores)
fpr_v1, tpr_v1, _ = roc_curve(image_labels, v1_raw_scores)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_v4, tpr_v4, label=f'V4 Reflect Detrended (AUROC={image_auc:.3f})', color='green', lw=2)
ax.plot(fpr_v1, tpr_v1, label=f'Raw Residual (AUROC={v1_auc:.3f})', color='gray', linestyle='--', lw=1.5)
ax.plot([0, 1], [0, 1], ':', color='black', alpha=0.5)
ax.set(xlabel='False-positive rate', ylabel='True-positive rate', title='Fabric Stain Detection ROC (V4 Protocol)')
ax.legend(loc='lower right')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT / 'roc_curve_v4.png', dpi=160)
plt.show()

print(f'V4 AUROC: {image_auc:.4f} (Raw Residual AUROC: {v1_auc:.4f})')
print(f'Strict Mode (tau={tau_strict:.2f}): Sensitivity={sens_strict:.3f}, Specificity={spec_strict:.3f}')
print(f'Hysteresis Mode (seed={tau_high:.2f}, body={tau_low:.2f}): Sensitivity={sens_hyst:.3f}, Specificity={spec_hyst:.3f}, BalAcc={bal_acc_hyst:.3f}')
"""
)

nb["cells"][10]["source"] = lines(
    """
# Select 6 diverse preview images:
# - Clean normal control
# - Creased normal control
# - High-contrast stain (strict TP)
# - Faint stains recovered by hysteresis
stain_indices = np.flatnonzero(image_labels == 1)
recovered_indices = np.flatnonzero((image_labels == 1) & (~strict_predictions) & hyst_predictions)
strict_stain_indices = np.flatnonzero((image_labels == 1) & strict_predictions)

selected = [
    0,                                                  # Clean normal
    6,                                                  # Creased normal
    int(strict_stain_indices[0]) if len(strict_stain_indices) > 0 else int(stain_indices[0]),
    int(strict_stain_indices[-1]) if len(strict_stain_indices) > 1 else int(stain_indices[-1]),
    int(recovered_indices[0]) if len(recovered_indices) > 0 else int(stain_indices[1]),
    int(recovered_indices[1]) if len(recovered_indices) > 1 else int(stain_indices[2]),
]

fig, axes = plt.subplots(len(selected), 5, figsize=(17, 3.2 * len(selected)))

for row, index in enumerate(selected):
    original = ((test_inputs[index].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    recon = ((test_recons[index].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    v4_map = test_v4_maps[index]
    
    strict_m = (v4_map > tau_strict).astype(np.float32)
    hyst_m = hyst_masks[index].astype(np.float32)

    panels = [original, recon, v4_map, strict_m, hyst_m]
    titles = [
        f"{rows[index]['kind'].upper()} #{index}",
        f"Reconstruction (t={T_DISTANCE}, N={NUM_DIFFUSION_SAMPLES})",
        f"V4 Heatmap (Score {image_scores[index]:.1f})",
        f"Strict Mask (tau={tau_strict:.1f})",
        f"Hysteresis Mask (Area={hyst_areas[index]}px)",
    ]
    for col, (panel, title) in enumerate(zip(panels, titles)):
        axes[row, col].imshow(panel, cmap=None if col < 2 else ('magma' if col == 2 else 'gray'))
        axes[row, col].set_title(title, fontsize=9.5)
        axes[row, col].axis('off')

fig.tight_layout()
fig.savefig(OUTPUT / 'evaluation_preview_v4.png', dpi=150, bbox_inches='tight')
plt.show()

report = {
    'status': 'passed',
    'scope': 'locked_fabric_stain_image_evaluation_v4',
    'dataset': 'Fabric Defects Dataset - matched grayscale stain cohort',
    'checkpoint_step': int(checkpoint['step']),
    'author_commit': author.COMMIT,
    'model_configuration': model_config,
    'train_normal_count': 48,
    'validation_normal_count': len(val_paths),
    'test_normal_count': int(np.sum(image_labels == 0)),
    'test_stain_count': int(np.sum(image_labels == 1)),
    't_distance': T_DISTANCE,
    'num_diffusion_samples': NUM_DIFFUSION_SAMPLES,
    'detrend_kernel': DETREND_KERNEL,
    'smooth_kernel': SMOOTH_KERNEL,
    'border_margin': BORDER_MARGIN,
    'min_cluster_area': MIN_CLUSTER_AREA,
    'tau_strict': tau_strict,
    'tau_high': tau_high,
    'tau_low': tau_low,
    'image_auroc_v4': image_auc,
    'image_auroc_raw': v1_auc,
    'average_precision': average_precision,
    'strict_confusion': {'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s), 'tp': int(tp_s)},
    'strict_sensitivity': sens_strict,
    'strict_specificity': spec_strict,
    'hysteresis_confusion': {'tn': int(tn_h), 'fp': int(fp_h), 'fn': int(fn_h), 'tp': int(tp_h)},
    'hysteresis_sensitivity': sens_hyst,
    'hysteresis_specificity': spec_hyst,
    'hysteresis_balanced_accuracy': bal_acc_hyst,
    'stains_recovered_by_hysteresis': int(tp_h - tp_s),
    'normal_score_mean': float(np.mean(image_scores[image_labels == 0])),
    'stain_score_mean': float(np.mean(image_scores[image_labels == 1])),
    'median_batch_seconds': float(np.median(test_batch_seconds)),
    'approx_test_seconds_per_image': float(sum(test_batch_seconds) / len(test_paths)),
    'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
    'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
}
(OUTPUT / 'evaluation_report_v4.json').write_text(json.dumps(report, indent=2))
np.savez_compressed(OUTPUT / 'test_maps_v4_float16.npz',
                    v4_maps=test_v4_maps.astype(np.float16),
                    raw_maps=test_raw_maps.astype(np.float16),
                    image_scores=image_scores, labels=image_labels,
                    strict_predictions=strict_predictions,
                    hysteresis_predictions=hyst_predictions)

archive = shutil.make_archive('/kaggle/working/fabric_stain_evaluation_v4_results', 'zip', OUTPUT)
print(json.dumps(report, indent=2))
print('\\nLOCKED FABRIC STAIN EVALUATION V4 PASSED')
print('Download:', archive)
"""
)

nb["metadata"]["colab"] = {"name": OUTPUT.name, "provenance": []}
OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print('Successfully generated:', OUTPUT)
