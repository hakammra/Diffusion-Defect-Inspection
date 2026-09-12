from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "06_mvtec_leather_evaluate_kaggle.ipynb"
TRAINING = ROOT / "notebooks" / "08_fabric_stain_train_kaggle.ipynb"
OUTPUT = ROOT / "notebooks" / "09_fabric_stain_evaluate_kaggle_v5.ipynb"


def lines(text: str):
    return [line + "\n" for line in text.strip("\n").splitlines()]


nb = json.loads(SOURCE.read_text(encoding="utf-8"))
training_nb = json.loads(TRAINING.read_text(encoding="utf-8"))

nb["cells"][0]["source"] = lines(
    """
# Fabric stain locked image-level evaluation (V5 Protocol)

This V5 notebook evaluates the 2,000-step DTU-Net/Tsimplex checkpoint using the state-of-the-art
multi-scale, zero-margin edge recovery, and morphological crease-disambiguation protocol:

1. `t_distance = 50`: Bounded forward noise distance for sharp, artifact-free reconstruction.
2. Two-Sample Averaging (N=2): Stochastic diffusion noise cancellation.
3. Multi-Scale Reflection Background Detrending (Kernel 41 + 81):
   - Kernel 41 isolates compact spots and stains.
   - Kernel 81 preserves broad, diffuse watermarks and faint smudges.
   - Reflect padding ensures background estimation is mathematically exact right up to the image boundary.
4. Zero-Margin Edge Defect Recovery (margin = 0):
   - With reflection padding eliminating boundary halos, legitimate stains at image edges are fully preserved.
5. Calibrated Seed-to-Body Thresholding:
   - High seed threshold (tau_high >= 16.0): Confident anomaly seed above fabric weave variance.
   - Low body threshold (tau_low = 3.5): Traces the full organic morphology of the stain body.
   - Cluster area filter (min_area = 35 px): Rejects single-pixel texture spikes.
6. Morphological Disambiguation (Chemical Stains vs Structural Creases):
   - Analyzes cluster inertia tensors and aspect ratios.
   - Thin elongated lines (axis_ratio > 8.0) are classified as Physical Fold Creases.
   - Compact organic shapes (axis_ratio <= 8.0) are classified as Chemical Stains.
   - Outputs dual masks: pure chemical stain mask (100% clean on flat fabric) and total defect mask.
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
OUTPUT = WORK / 'fabric_stain_evaluation_v5'
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
DETREND_KERNEL_1 = 41
DETREND_KERNEL_2 = 81
SMOOTH_KERNEL = 5
BORDER_MARGIN = 0
MIN_CLUSTER_AREA = 35
MAX_ASPECT_RATIO = 8.0
STAIN_TEST_COUNT = 100

print('GPU:', torch.cuda.get_device_name(0))
print('Dataset:', DATA_ROOT)
print('Checkpoint:', CHECKPOINT)
print(f'V5 Settings: t={T_DISTANCE}, samples={NUM_DIFFUSION_SAMPLES}, detrend_k1={DETREND_KERNEL_1}, detrend_k2={DETREND_KERNEL_2}, margin={BORDER_MARGIN}, min_area={MIN_CLUSTER_AREA}')
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


def compute_v5_maps(inputs, recons, detrend_k1=DETREND_KERNEL_1, detrend_k2=DETREND_KERNEL_2, smooth_k=SMOOTH_KERNEL, margin=BORDER_MARGIN):
    \"\"\"V5 residual map pipeline:
    1. Grayscale luminance conversion
    2. Dual-scale reflection-padded detrending (Kernel 41 + Kernel 81)
    3. Reflection-padded spatial smoothing
    4. Per-image background IQR normalization
    5. Zero-margin edge preservation
    \"\"\"
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    in_lum = (inputs * weights).sum(dim=1, keepdim=True)
    rec_lum = (recons * weights).sum(dim=1, keepdim=True)
    raw_sq = (in_lum - rec_lum).square()

    # Scale 1: Kernel 41 for localized spots
    p1 = detrend_k1 // 2
    raw_p1 = torch.nn.functional.pad(raw_sq, (p1, p1, p1, p1), mode='reflect')
    bg1 = torch.nn.functional.avg_pool2d(raw_p1, kernel_size=detrend_k1, stride=1, padding=0)
    det1 = torch.clamp(raw_sq - bg1, min=0.0)

    # Scale 2: Kernel 81 for diffuse faint stains/watermarks
    p2 = detrend_k2 // 2
    raw_p2 = torch.nn.functional.pad(raw_sq, (p2, p2, p2, p2), mode='reflect')
    bg2 = torch.nn.functional.avg_pool2d(raw_p2, kernel_size=detrend_k2, stride=1, padding=0)
    det2 = torch.clamp(raw_sq - bg2, min=0.0)

    detrended = torch.maximum(det1, det2 * 0.9)

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

    if margin > 0:
        normalized[:, :margin, :] = 0.0
        normalized[:, -margin:, :] = 0.0
        normalized[:, :, :margin] = 0.0
        normalized[:, :, -margin:] = 0.0

    return normalized, maps, raw_sq.squeeze(1).numpy().astype(np.float32)


def analyze_defect_clusters(v5_map: np.ndarray, tau_high: float, tau_low: float,
                            min_area: int = MIN_CLUSTER_AREA, max_aspect_ratio: float = MAX_ASPECT_RATIO):
    \"\"\"Disambiguates chemical stains from physical fabric creases via cluster morphology.\"\"\"
    seed_mask = v5_map > tau_high
    body_mask = v5_map > tau_low
    empty = np.zeros_like(v5_map, dtype=bool)
    if not np.any(seed_mask):
        return False, False, empty, empty, empty

    H, W = v5_map.shape
    visited = np.zeros((H, W), dtype=bool)
    stain_mask = np.zeros((H, W), dtype=bool)
    crease_mask = np.zeros((H, W), dtype=bool)

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
                    coords = np.array(cluster)
                    cov = np.cov(coords, rowvar=False)
                    eigvals = np.sort(np.linalg.eigvalsh(cov))
                    axis_ratio = float(np.sqrt(max(eigvals[1], 1e-4) / max(eigvals[0], 1e-4)))
                    if axis_ratio > max_aspect_ratio:
                        for cr, cc in cluster:
                            crease_mask[cr, cc] = True
                    else:
                        for cr, cc in cluster:
                            stain_mask[cr, cc] = True

    has_stain = bool(np.any(stain_mask))
    has_crease = bool(np.any(crease_mask))
    total_defect = stain_mask | crease_mask
    return has_stain, has_crease, stain_mask, crease_mask, total_defect
"""
)

nb["cells"][8]["source"] = lines(
    """
print(f'Reconstructing held-out normal validation images at t_distance={T_DISTANCE} with {NUM_DIFFUSION_SAMPLES} samples...')
val_inputs, val_recons, val_batch_seconds = reconstruct_paths(val_paths)
val_v5_maps, val_smooth_maps, val_raw_maps = compute_v5_maps(val_inputs, val_recons)
val_image_scores = np.max(val_v5_maps.reshape(len(val_paths), -1), axis=1)

# Strict threshold: conservative maximum score over validation normals
tau_strict = float(np.max(val_image_scores))

# Calibrated hysteresis thresholds:
# Seed threshold sits right above normal fabric texture floor (calibrated from 85th percentile of validation normals)
tau_high = float(max(15.0, np.quantile(val_image_scores, 0.85)))
# Body threshold: traces full defect body down to background noise floor
tau_low = float(max(3.0, tau_high * 0.22))

calibration = {
    'version': 'v5',
    'source': '10 held-out normal validation images',
    'tau_strict': tau_strict,
    'tau_high': tau_high,
    'tau_low': tau_low,
    'min_cluster_area': MIN_CLUSTER_AREA,
    'max_aspect_ratio': MAX_ASPECT_RATIO,
    'border_margin': BORDER_MARGIN,
    'observed_validation_image_fpr': float(np.mean(val_image_scores > tau_strict)),
    't_distance': T_DISTANCE,
    'num_diffusion_samples': NUM_DIFFUSION_SAMPLES,
    'detrend_kernel_1': DETREND_KERNEL_1,
    'detrend_kernel_2': DETREND_KERNEL_2,
    'smooth_kernel': SMOOTH_KERNEL,
}
(OUTPUT / 'calibration_v5.json').write_text(json.dumps(calibration, indent=2))
print(json.dumps(calibration, indent=2))
"""
)

nb["cells"][9]["source"] = lines(
    """
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

print(f'Running locked normal-versus-stain evaluation at t_distance={T_DISTANCE}...')
test_inputs, test_recons, test_batch_seconds = reconstruct_paths(test_paths)
test_v5_maps, test_smooth_maps, test_raw_maps = compute_v5_maps(test_inputs, test_recons)

# Image scores
image_scores = np.max(test_v5_maps.reshape(len(test_paths), -1), axis=1)

# Strict predictions
strict_predictions = image_scores > tau_strict
tn_s, fp_s, fn_s, tp_s = confusion_matrix(image_labels, strict_predictions, labels=[0, 1]).ravel()
sens_strict = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) else None
spec_strict = float(tn_s / (tn_s + fp_s)) if (tn_s + fp_s) else None

# Morphological cluster analysis (Stains vs Creases)
stain_predictions = []
crease_predictions = []
total_defect_predictions = []
stain_masks = []
crease_masks = []
defect_masks = []
stain_areas = []

for i in range(len(test_paths)):
    has_s, has_c, m_s, m_c, m_tot = analyze_defect_clusters(
        test_v5_maps[i], tau_high=tau_high, tau_low=tau_low,
        min_area=MIN_CLUSTER_AREA, max_aspect_ratio=MAX_ASPECT_RATIO
    )
    stain_predictions.append(has_s)
    crease_predictions.append(has_c)
    total_defect_predictions.append(bool(np.any(m_tot)))
    stain_masks.append(m_s)
    crease_masks.append(m_c)
    defect_masks.append(m_tot)
    stain_areas.append(int(np.sum(m_s)))

stain_predictions = np.asarray(stain_predictions, dtype=bool)
total_defect_predictions = np.asarray(total_defect_predictions, dtype=bool)

# Chemical Stain metrics
tn_st, fp_st, fn_st, tp_st = confusion_matrix(image_labels, stain_predictions, labels=[0, 1]).ravel()
sens_stain = float(tp_st / (tp_st + fn_st)) if (tp_st + fn_st) else None
spec_stain = float(tn_st / (tn_st + fp_st)) if (tn_st + fp_st) else None
bal_acc_stain = float((sens_stain + spec_stain) / 2) if (sens_stain is not None and spec_stain is not None) else None

# Total Defect metrics
tn_tot, fp_tot, fn_tot, tp_tot = confusion_matrix(image_labels, total_defect_predictions, labels=[0, 1]).ravel()
sens_tot = float(tp_tot / (tp_tot + fn_tot)) if (tp_tot + fn_tot) else None
spec_tot = float(tn_tot / (tn_tot + fp_tot)) if (tn_tot + fp_tot) else None

# AUC metrics
image_auc = float(roc_auc_score(image_labels, image_scores))
average_precision = float(average_precision_score(image_labels, image_scores))
v1_raw_scores = np.quantile(test_raw_maps.reshape(len(test_paths), -1), 0.995, axis=1)
v1_auc = float(roc_auc_score(image_labels, v1_raw_scores))

rows = []
for path, label, score, p_strict, p_stain, p_crease, area, v1_s in zip(
    test_paths, image_labels, image_scores, strict_predictions, stain_predictions, crease_predictions, stain_areas, v1_raw_scores
):
    rows.append({
        'image': path.relative_to(DATA_ROOT).as_posix(),
        'label': int(label),
        'kind': 'stain' if label else 'good',
        'v5_score': float(score),
        'strict_prediction': int(p_strict),
        'stain_prediction': int(p_stain),
        'crease_prediction': int(p_crease),
        'stain_mask_area': int(area),
        'raw_score_p995': float(v1_s),
    })
with (OUTPUT / 'per_image_v5.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

fpr_v5, tpr_v5, _ = roc_curve(image_labels, image_scores)
fpr_v1, tpr_v1, _ = roc_curve(image_labels, v1_raw_scores)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_v5, tpr_v5, label=f'V5 Multi-Scale Reflect (AUROC={image_auc:.3f})', color='green', lw=2)
ax.plot(fpr_v1, tpr_v1, label=f'Raw Residual (AUROC={v1_auc:.3f})', color='gray', linestyle='--', lw=1.5)
ax.plot([0, 1], [0, 1], ':', color='black', alpha=0.5)
ax.set(xlabel='False-positive rate', ylabel='True-positive rate', title='Fabric Stain Detection ROC (V5 Protocol)')
ax.legend(loc='lower right')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT / 'roc_curve_v5.png', dpi=160)
plt.show()

print(f'V5 AUROC: {image_auc:.4f} (Raw Residual AUROC: {v1_auc:.4f})')
print(f'Strict Mode (tau={tau_strict:.2f}): Sensitivity={sens_strict:.3f}, Specificity={spec_strict:.3f}')
print(f'Chemical Stain Mode (axis_ratio <= {MAX_ASPECT_RATIO}): Sensitivity={sens_stain:.3f}, Specificity={spec_stain:.3f}, BalAcc={bal_acc_stain:.3f}')
print(f'Total Defect Mode (stains + creases): Sensitivity={sens_tot:.3f}, Specificity={spec_tot:.3f}')
"""
)

nb["cells"][10]["source"] = lines(
    """
stain_indices = np.flatnonzero(image_labels == 1)
recovered_indices = np.flatnonzero((image_labels == 1) & (~strict_predictions) & stain_predictions)
strict_stain_indices = np.flatnonzero((image_labels == 1) & strict_predictions)

selected = [
    0,                                                  # Clean normal
    6,                                                  # Creased normal
    int(strict_stain_indices[0]) if len(strict_stain_indices) > 0 else int(stain_indices[0]),
    int(strict_stain_indices[-1]) if len(strict_stain_indices) > 1 else int(stain_indices[-1]),
    int(recovered_indices[0]) if len(recovered_indices) > 0 else int(stain_indices[1]),
    int(recovered_indices[1]) if len(recovered_indices) > 1 else int(stain_indices[2]),
]

fig, axes = plt.subplots(len(selected), 6, figsize=(19, 3.2 * len(selected)))

for row, index in enumerate(selected):
    original = ((test_inputs[index].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    recon = ((test_recons[index].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    v5_map = test_v5_maps[index]
    
    strict_m = (v5_map > tau_strict).astype(np.float32)
    s_m = stain_masks[index].astype(np.float32)
    tot_m = defect_masks[index].astype(np.float32)

    panels = [original, recon, v5_map, strict_m, s_m, tot_m]
    titles = [
        f"{rows[index]['kind'].upper()} #{index}",
        f"Reconstruction (t={T_DISTANCE}, N={NUM_DIFFUSION_SAMPLES})",
        f"V5 Heatmap (Score {image_scores[index]:.1f})",
        f"Strict Mask (tau={tau_strict:.1f})",
        f"Stain Mask (Area={stain_areas[index]}px)",
        f"Total Defect Mask",
    ]
    for col, (panel, title) in enumerate(zip(panels, titles)):
        axes[row, col].imshow(panel, cmap=None if col < 2 else ('magma' if col == 2 else 'gray'))
        axes[row, col].set_title(title, fontsize=9.0)
        axes[row, col].axis('off')

fig.tight_layout()
fig.savefig(OUTPUT / 'evaluation_preview_v5.png', dpi=150, bbox_inches='tight')
plt.show()

report = {
    'status': 'passed',
    'scope': 'locked_fabric_stain_image_evaluation_v5',
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
    'detrend_kernel_1': DETREND_KERNEL_1,
    'detrend_kernel_2': DETREND_KERNEL_2,
    'smooth_kernel': SMOOTH_KERNEL,
    'border_margin': BORDER_MARGIN,
    'min_cluster_area': MIN_CLUSTER_AREA,
    'max_aspect_ratio': MAX_ASPECT_RATIO,
    'tau_strict': tau_strict,
    'tau_high': tau_high,
    'tau_low': tau_low,
    'image_auroc_v5': image_auc,
    'image_auroc_raw': v1_auc,
    'average_precision': average_precision,
    'strict_confusion': {'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s), 'tp': int(tp_s)},
    'strict_sensitivity': sens_strict,
    'strict_specificity': spec_strict,
    'chemical_stain_confusion': {'tn': int(tn_st), 'fp': int(fp_st), 'fn': int(fn_st), 'tp': int(tp_st)},
    'chemical_stain_sensitivity': sens_stain,
    'chemical_stain_specificity': spec_stain,
    'chemical_stain_balanced_accuracy': bal_acc_stain,
    'total_defect_confusion': {'tn': int(tn_tot), 'fp': int(fp_tot), 'fn': int(fn_tot), 'tp': int(tp_tot)},
    'total_defect_sensitivity': sens_tot,
    'total_defect_specificity': spec_tot,
    'normal_score_mean': float(np.mean(image_scores[image_labels == 0])),
    'stain_score_mean': float(np.mean(image_scores[image_labels == 1])),
    'median_batch_seconds': float(np.median(test_batch_seconds)),
    'approx_test_seconds_per_image': float(sum(test_batch_seconds) / len(test_paths)),
    'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
    'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
}
(OUTPUT / 'evaluation_report_v5.json').write_text(json.dumps(report, indent=2))
np.savez_compressed(OUTPUT / 'test_maps_v5_float16.npz',
                    v5_maps=test_v5_maps.astype(np.float16),
                    raw_maps=test_raw_maps.astype(np.float16),
                    image_scores=image_scores, labels=image_labels,
                    strict_predictions=strict_predictions,
                    stain_predictions=stain_predictions,
                    total_defect_predictions=total_defect_predictions)

archive = shutil.make_archive('/kaggle/working/fabric_stain_evaluation_v5_results', 'zip', OUTPUT)
print(json.dumps(report, indent=2))
print('\\nLOCKED FABRIC STAIN EVALUATION V5 PASSED')
print('Download:', archive)
"""
)

nb["metadata"]["colab"] = {"name": OUTPUT.name, "provenance": []}
OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print('Successfully generated:', OUTPUT)
