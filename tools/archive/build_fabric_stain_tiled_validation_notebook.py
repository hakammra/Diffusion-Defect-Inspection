from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "09_fabric_stain_evaluate_kaggle_v5.ipynb"
OUTPUT = ROOT / "notebooks" / "14_fabric_stain_tiled_validation_kaggle.ipynb"


def lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


source = json.loads(SOURCE.read_text(encoding="utf-8"))

cells = [
    markdown(
        r"""
# Fabric stain tiled-inference validation

This is a **development/model-selection run**, not a final test. It compares the existing
whole-image resize with a tiled method that matches the model's 512-pixel training crop scale.

- Training remains self-supervised and uses normal fabric only.
- The 10 normal validation images calibrate false alarms.
- Thirty stain images that were not used in V1-V5 select the inference method and threshold.
- The original 100 test stains are excluded from this run.
- After this notebook, freeze the selected settings and run them once on a new final cohort.

Why tiling may help: the current evaluation compresses a 1984x1488 image to 224x224, which can
erase a small stain. Tiling crops the original at 512x512, as in training, before resizing each
crop to 224x224.
"""
    ),
    code(
        r"""
from pathlib import Path
import csv, importlib.util, json, random, shutil, sys, time, zipfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working/labelinspect')
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT = WORK / 'fabric_stain_tiled_validation'
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
            names = ['/' + item.filename.replace('\\', '/').lstrip('/') for item in archive.infolist()]
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
T_DISTANCE = 50
NUM_DIFFUSION_SAMPLES = 1
TILE_SOURCE_SIZE = 512
TILE_STRIDE = 384
TILE_BATCH_SIZE = 12
VALIDATION_STAIN_COUNT = 30
DETREND_KERNEL_1 = 41
DETREND_KERNEL_2 = 81
SMOOTH_KERNEL = 5
TILE_PIXEL_QUANTILE = 0.999

print('GPU:', torch.cuda.get_device_name(0))
print('Dataset:', DATA_ROOT)
print('Checkpoint:', CHECKPOINT)
"""
    ),
    # Reuse the tested author-source bootstrap cell from V5.
    source["cells"][3],
    code(
        r"""
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
    ),
    code(
        r"""
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
def image_paths(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

val_normal_paths = image_paths(DATA_ROOT / 'val_normal')
all_stain_paths = image_paths(DATA_ROOT / 'test' / 'stain')
assert len(val_normal_paths) == 10 and len(all_stain_paths) == 398

# V1-V5 used shuffled positions 0:100. These validation stains come from 100:130.
rng = random.Random(SEED)
shuffled_stains = all_stain_paths.copy()
rng.shuffle(shuffled_stains)
validation_stain_paths = sorted(shuffled_stains[100:100 + VALIDATION_STAIN_COUNT])
validation_paths = val_normal_paths + validation_stain_paths
labels = np.asarray([0] * len(val_normal_paths) + [1] * len(validation_stain_paths), dtype=np.int64)

(OUTPUT / 'development_validation_files.txt').write_text(
    '\n'.join(path.relative_to(DATA_ROOT).as_posix() for path in validation_paths)
)
print('Development set:', len(val_normal_paths), 'normal +', len(validation_stain_paths), 'previously unused stains')
"""
    ),
    code(
        r"""
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
    ),
    code(
        r"""
RESAMPLE = getattr(Image, 'Resampling', Image).BILINEAR

def pil_to_tensor(image):
    image = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE)
    array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)

def reconstruct_tensor_batch(batch):
    x = batch.to(device)
    with torch.inference_mode():
        samples = []
        for _ in range(NUM_DIFFUSION_SAMPLES):
            samples.append(diffusion.forward_backward(
                model, x, None, see_whole_sequence=None,
                t_distance=T_DISTANCE, denoise_fn='noise_fn'))
        recon = torch.stack(samples).mean(dim=0)
    return x.cpu(), recon.cpu()

def residual_maps(inputs, recons):
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    raw = ((inputs * weights).sum(1, keepdim=True) - (recons * weights).sum(1, keepdim=True)).square()

    def reflected_mean(x, kernel):
        pad = kernel // 2
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'), kernel, stride=1)

    det1 = torch.clamp(raw - reflected_mean(raw, DETREND_KERNEL_1), min=0)
    det2 = torch.clamp(raw - reflected_mean(raw, DETREND_KERNEL_2), min=0)
    detrended = torch.maximum(det1, 0.9 * det2)
    smoothed = reflected_mean(detrended, SMOOTH_KERNEL).squeeze(1).numpy().astype(np.float32)

    flat = smoothed.reshape(len(smoothed), -1)
    med = np.median(flat, axis=1, keepdims=True)
    q25 = np.quantile(flat, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(flat, 0.75, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-6)
    return ((flat - med) / iqr).reshape(smoothed.shape)

def positions(length, crop=TILE_SOURCE_SIZE, stride=TILE_STRIDE):
    if length <= crop:
        return [0]
    values = list(range(0, length - crop + 1, stride))
    if values[-1] != length - crop:
        values.append(length - crop)
    return values

def evaluate_path(path):
    with Image.open(path) as source_image:
        full = source_image.convert('RGB')
        width, height = full.size

        global_tensor = pil_to_tensor(full).unsqueeze(0)
        global_input, global_recon = reconstruct_tensor_batch(global_tensor)
        global_map = residual_maps(global_input, global_recon)[0]
        global_score = float(np.max(global_map))

        tile_tensors = []
        tile_boxes = []
        for top in positions(height):
            for left in positions(width):
                box = (left, top, min(left + TILE_SOURCE_SIZE, width), min(top + TILE_SOURCE_SIZE, height))
                tile_tensors.append(pil_to_tensor(full.crop(box)))
                tile_boxes.append(box)

    tile_maps = []
    tick = time.perf_counter()
    for start in range(0, len(tile_tensors), TILE_BATCH_SIZE):
        batch = torch.stack(tile_tensors[start:start + TILE_BATCH_SIZE])
        tile_inputs, tile_recons = reconstruct_tensor_batch(batch)
        tile_maps.extend(residual_maps(tile_inputs, tile_recons))
    elapsed = time.perf_counter() - tick

    tile_maps = np.asarray(tile_maps, dtype=np.float32)
    tile_scores = np.quantile(tile_maps.reshape(len(tile_maps), -1), TILE_PIXEL_QUANTILE, axis=1)
    top_two = np.sort(tile_scores)[-min(2, len(tile_scores)):]
    return {
        'global_score': global_score,
        'tile_max_score': float(np.max(tile_scores)),
        'tile_top2_mean_score': float(np.mean(top_two)),
        'tile_count': len(tile_scores),
        'seconds': elapsed,
        'tile_maps': tile_maps,
        'tile_boxes': tile_boxes,
        'image_size': (width, height),
    }
"""
    ),
    code(
        r"""
records = []
preview_cache = {}
run_tick = time.perf_counter()
for index, (path, label) in enumerate(zip(validation_paths, labels)):
    result = evaluate_path(path)
    records.append({
        'image': path.relative_to(DATA_ROOT).as_posix(),
        'label': int(label),
        'kind': 'stain' if label else 'good',
        'global_score': result['global_score'],
        'tile_max_score': result['tile_max_score'],
        'tile_top2_mean_score': result['tile_top2_mean_score'],
        'tile_count': result['tile_count'],
        'tile_seconds': result['seconds'],
    })
    if index in {0, 1, 10, 11, 12, 13}:
        preview_cache[index] = result
    print(f"{index + 1}/{len(validation_paths)} {path.name}: global={result['global_score']:.2f}, "
          f"tile-top2={result['tile_top2_mean_score']:.2f}, {result['tile_count']} tiles", flush=True)
run_seconds = time.perf_counter() - run_tick

with (OUTPUT / 'validation_per_image.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader(); writer.writerows(records)
print('Validation runtime (minutes):', run_seconds / 60)
"""
    ),
    code(
        r"""
def choose_threshold(y, scores, minimum_specificity=0.90):
    candidates = np.r_[-np.inf, np.unique(scores), np.inf]
    choices = []
    for threshold in candidates:
        pred = scores > threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced = (sensitivity + specificity) / 2
        if specificity >= minimum_specificity:
            choices.append((balanced, sensitivity, specificity, float(threshold), (tn, fp, fn, tp)))
    return max(choices, key=lambda item: (item[0], item[1], item[2], item[3]))

methods = ['global_score', 'tile_max_score', 'tile_top2_mean_score']
comparison = []
for method in methods:
    scores = np.asarray([row[method] for row in records], dtype=np.float64)
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    balanced, sensitivity, specificity, threshold, counts = choose_threshold(labels, scores)
    tn, fp, fn, tp = counts
    comparison.append({
        'method': method,
        'auroc': auc,
        'average_precision': ap,
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': balanced,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    })

# Select by AUROC first, then balanced accuracy. This choice is frozen for the later final run.
selected = max(comparison, key=lambda row: (row['auroc'], row['balanced_accuracy']))
with (OUTPUT / 'method_comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
    writer.writeheader(); writer.writerows(comparison)

selection = {
    'status': 'development_validation_completed',
    'final_test_claim': False,
    'checkpoint_step': int(checkpoint['step']),
    'validation_normal_count': int(np.sum(labels == 0)),
    'validation_stain_count': int(np.sum(labels == 1)),
    'stain_slice': '[100:130] after seed-230224 shuffle; disjoint from V1-V5 [0:100]',
    't_distance': T_DISTANCE,
    'num_diffusion_samples': NUM_DIFFUSION_SAMPLES,
    'tile_source_size': TILE_SOURCE_SIZE,
    'tile_stride': TILE_STRIDE,
    'tile_batch_size': TILE_BATCH_SIZE,
    'tile_pixel_quantile': TILE_PIXEL_QUANTILE,
    'comparison': comparison,
    'selected_method': selected['method'],
    'frozen_threshold': selected['threshold'],
    'run_seconds': run_seconds,
    'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
    'next_step': 'Run the selected method and frozen threshold once on a disjoint final stain cohort.',
}
(OUTPUT / 'model_selection.json').write_text(json.dumps(selection, indent=2))
print(json.dumps(selection, indent=2))
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, method in zip(axes[0], methods):
    normal = [row[method] for row in records if row['label'] == 0]
    stain = [row[method] for row in records if row['label'] == 1]
    ax.boxplot([normal, stain], labels=['normal', 'stain'], showfliers=True)
    selected_row = next(row for row in comparison if row['method'] == method)
    ax.axhline(selected_row['threshold'], color='red', linestyle='--', label='selected cutoff')
    ax.set_title(f"{method}\nAUROC={selected_row['auroc']:.3f}")
    ax.grid(alpha=.2); ax.legend(fontsize=8)

preview_indices = sorted(preview_cache)[:3]
for ax, index in zip(axes[1], preview_indices):
    result = preview_cache[index]
    maps = result['tile_maps']
    best = int(np.argmax(np.quantile(maps.reshape(len(maps), -1), TILE_PIXEL_QUANTILE, axis=1)))
    ax.imshow(maps[best], cmap='magma')
    ax.set_title(f"{records[index]['kind']} {Path(records[index]['image']).name}\nbest tile response")
    ax.axis('off')

fig.tight_layout()
fig.savefig(OUTPUT / 'validation_summary.png', dpi=160, bbox_inches='tight')
plt.show()

archive = shutil.make_archive('/kaggle/working/fabric_stain_tiled_validation_results', 'zip', OUTPUT)
print('Download:', archive)
print('This output selects a candidate. It is not the final test result.')
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": source.get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": 5,
}
notebook["metadata"]["colab"] = {
    "name": OUTPUT.name,
    "provenance": [],
    "gpuType": "T4",
}
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("Generated:", OUTPUT)
