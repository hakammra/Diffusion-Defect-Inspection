from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "14_fabric_stain_tiled_validation_kaggle.ipynb"
OUTPUT = ROOT / "notebooks" / "15_fabric_stain_tiled_final_evaluate_kaggle.ipynb"


def lines(text: str) -> list[str]:
    return (text.strip("\n") + "\n").splitlines(keepends=True)


nb = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))

nb["cells"][0]["source"] = lines(
    r"""
# Fabric stain frozen tiled final evaluation

This notebook applies the method selected in the preceding development-validation run exactly once:

- DTU-Net checkpoint: 2,000 steps
- partial diffusion distance: `t=50`
- one reconstruction sample
- 512x512 source tiles with stride 384
- per-tile score: residual-map 99.9th percentile
- image score: mean of the two highest tile scores
- frozen decision cutoff: **16.47188949584961**

The 100 stain images in this run are disjoint from the 100 stains used by V1-V5 and the 30 stains
used for tiled model selection. The ten normal reference images were evaluated by earlier exploratory
versions, so stain sensitivity is the cleanest independent result; normal specificity must be reported
with that limitation. No setting may be changed after viewing this notebook's output.
"""
)

setup = "".join(nb["cells"][1]["source"])
setup = setup.replace(
    "OUTPUT = WORK / 'fabric_stain_tiled_validation'",
    "OUTPUT = WORK / 'fabric_stain_tiled_final_evaluation'",
)
setup = setup.replace("VALIDATION_STAIN_COUNT = 30", "FINAL_STAIN_COUNT = 100")
setup += "\nFROZEN_METHOD = 'tile_top2_mean_score'\nFROZEN_THRESHOLD = 16.47188949584961\n"
nb["cells"][1]["source"] = lines(setup)

nb["cells"][4]["source"] = lines(
    r"""
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
def image_paths(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

normal_reference_paths = image_paths(DATA_ROOT / 'test' / 'good')
all_stain_paths = image_paths(DATA_ROOT / 'test' / 'stain')
assert len(normal_reference_paths) == 10 and len(all_stain_paths) == 398

rng = random.Random(SEED)
shuffled_stains = all_stain_paths.copy()
rng.shuffle(shuffled_stains)

# [0:100] was used in V1-V5; [100:130] was tiled development validation.
# This final stain cohort is the previously unseen slice [130:230].
final_stain_paths = sorted(shuffled_stains[130:130 + FINAL_STAIN_COUNT])
final_paths = normal_reference_paths + final_stain_paths
labels = np.asarray([0] * len(normal_reference_paths) + [1] * len(final_stain_paths), dtype=np.int64)

(OUTPUT / 'frozen_final_files.txt').write_text(
    '\n'.join(path.relative_to(DATA_ROOT).as_posix() for path in final_paths)
)
print('Frozen evaluation:', len(normal_reference_paths), 'normal references +', len(final_stain_paths), 'unseen stains')
print('Frozen method:', FROZEN_METHOD, 'cutoff:', FROZEN_THRESHOLD)
"""
)

nb["cells"][6]["source"] = lines(
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
    order = np.argsort(tile_scores)
    top_two = tile_scores[order[-min(2, len(order)):]]
    best = int(order[-1])
    return {
        'score': float(np.mean(top_two)),
        'tile_count': len(tile_scores),
        'seconds': elapsed,
        'best_tile_score': float(tile_scores[best]),
        'best_tile_map': tile_maps[best].astype(np.float16),
        'best_tile_box': tile_boxes[best],
    }
"""
)

nb["cells"][7]["source"] = lines(
    r"""
records = []
best_maps = []
best_boxes = []
run_tick = time.perf_counter()
for index, (path, label) in enumerate(zip(final_paths, labels)):
    result = evaluate_path(path)
    prediction = bool(result['score'] > FROZEN_THRESHOLD)
    records.append({
        'image': path.relative_to(DATA_ROOT).as_posix(),
        'label': int(label),
        'kind': 'stain' if label else 'good',
        'tile_top2_mean_score': result['score'],
        'frozen_threshold': FROZEN_THRESHOLD,
        'prediction': int(prediction),
        'correct': int(prediction == bool(label)),
        'best_tile_score': result['best_tile_score'],
        'best_tile_box': ','.join(map(str, result['best_tile_box'])),
        'tile_count': result['tile_count'],
        'seconds': result['seconds'],
    })
    best_maps.append(result['best_tile_map'])
    best_boxes.append(result['best_tile_box'])
    print(f"{index + 1}/{len(final_paths)} {path.name}: score={result['score']:.2f}, "
          f"prediction={'stain' if prediction else 'normal'}", flush=True)
run_seconds = time.perf_counter() - run_tick

with (OUTPUT / 'final_per_image.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader(); writer.writerows(records)
np.savez_compressed(OUTPUT / 'best_tile_maps_float16.npz', maps=np.asarray(best_maps), labels=labels)
print('Evaluation runtime (minutes):', run_seconds / 60)
"""
)

nb["cells"][8]["source"] = lines(
    r"""
scores = np.asarray([row['tile_top2_mean_score'] for row in records], dtype=np.float64)
predictions = scores > FROZEN_THRESHOLD
tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
sensitivity = float(tp / (tp + fn))
specificity = float(tn / (tn + fp))
balanced_accuracy = float((sensitivity + specificity) / 2)
image_auc = float(roc_auc_score(labels, scores))
average_precision = float(average_precision_score(labels, scores))

def wilson_interval(successes, total, z=1.959963984540054):
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return [float(center-radius), float(center+radius)]

# Stratified bootstrap for AUROC uncertainty.
bootstrap_rng = np.random.default_rng(SEED)
normal_scores = scores[labels == 0]
stain_scores = scores[labels == 1]
bootstrap_auc = []
for _ in range(2000):
    sampled_normal = bootstrap_rng.choice(normal_scores, size=len(normal_scores), replace=True)
    sampled_stain = bootstrap_rng.choice(stain_scores, size=len(stain_scores), replace=True)
    sampled_scores = np.r_[sampled_normal, sampled_stain]
    sampled_labels = np.r_[np.zeros(len(sampled_normal), dtype=int), np.ones(len(sampled_stain), dtype=int)]
    bootstrap_auc.append(roc_auc_score(sampled_labels, sampled_scores))
auc_ci = [float(x) for x in np.quantile(bootstrap_auc, [0.025, 0.975])]

report = {
    'status': 'frozen_final_evaluation_completed',
    'method_changed_after_evaluation': False,
    'checkpoint_step': int(checkpoint['step']),
    'method': FROZEN_METHOD,
    'threshold': FROZEN_THRESHOLD,
    'test_normal_count': int(np.sum(labels == 0)),
    'test_stain_count': int(np.sum(labels == 1)),
    'stain_test_independent_of_v1_v5_and_tiled_validation': True,
    'normal_reference_previously_used_by_exploratory_versions': True,
    'image_auroc': image_auc,
    'image_auroc_bootstrap_95_ci': auc_ci,
    'average_precision': average_precision,
    'no_skill_average_precision': float(np.mean(labels)),
    'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'sensitivity': sensitivity,
    'sensitivity_wilson_95_ci': wilson_interval(int(tp), int(tp + fn)),
    'specificity': specificity,
    'specificity_wilson_95_ci': wilson_interval(int(tn), int(tn + fp)),
    'balanced_accuracy': balanced_accuracy,
    't_distance': T_DISTANCE,
    'num_diffusion_samples': NUM_DIFFUSION_SAMPLES,
    'tile_source_size': TILE_SOURCE_SIZE,
    'tile_stride': TILE_STRIDE,
    'tile_pixel_quantile': TILE_PIXEL_QUANTILE,
    'run_seconds': run_seconds,
    'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
    'segmentation_metrics_reported': False,
    'segmentation_limitation': 'The source has no pixel-level masks.',
}
(OUTPUT / 'final_evaluation_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
"""
)

nb["cells"][9]["source"] = lines(
    r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
normal_scores = scores[labels == 0]
stain_scores = scores[labels == 1]
axes[0].boxplot([normal_scores, stain_scores], labels=['normal reference', 'unseen stain'])
axes[0].axhline(FROZEN_THRESHOLD, color='red', linestyle='--', label='frozen cutoff')
axes[0].set_title(f'Frozen tiled score\nAUROC={image_auc:.3f}')
axes[0].set_ylabel('Top-two tile mean score')
axes[0].grid(alpha=.2); axes[0].legend()

axes[1].hist(normal_scores, bins=10, alpha=.7, label='normal reference')
axes[1].hist(stain_scores, bins=20, alpha=.7, label='unseen stain')
axes[1].axvline(FROZEN_THRESHOLD, color='red', linestyle='--')
axes[1].set_title('Frozen-score distributions')
axes[1].set_xlabel('Top-two tile mean score'); axes[1].legend()
fig.tight_layout()
fig.savefig(OUTPUT / 'final_score_summary.png', dpi=160, bbox_inches='tight')
plt.show()

categories = {
    'true normal': np.flatnonzero((labels == 0) & (~predictions)),
    'false alarm': np.flatnonzero((labels == 0) & predictions),
    'detected stain': np.flatnonzero((labels == 1) & predictions),
    'missed stain': np.flatnonzero((labels == 1) & (~predictions)),
}
selected = []
for name, indices in categories.items():
    for index in indices[:2]:
        selected.append((name, int(index)))

fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3.4 * len(selected)))
if len(selected) == 1:
    axes = np.asarray([axes])
for row, (category, index) in enumerate(selected):
    with Image.open(final_paths[index]) as image:
        full = image.convert('RGB')
        tile = full.crop(best_boxes[index]).resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE)
        full_preview = full.copy()
        full_preview.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
    response = np.asarray(best_maps[index], dtype=np.float32)
    panels = [full_preview, tile, response]
    titles = [
        f"{category}: {final_paths[index].name}\nscore={scores[index]:.2f}",
        f"highest-response 512px tile",
        "tile residual heatmap (qualitative)",
    ]
    for col, (panel, title) in enumerate(zip(panels, titles)):
        axes[row, col].imshow(panel, cmap='magma' if col == 2 else None)
        axes[row, col].set_title(title, fontsize=9)
        axes[row, col].axis('off')
fig.tight_layout()
fig.savefig(OUTPUT / 'final_examples.png', dpi=150, bbox_inches='tight')
plt.show()

archive = shutil.make_archive('/kaggle/working/fabric_stain_tiled_final_evaluation_results', 'zip', OUTPUT)
print('Download:', archive)
print('The method and threshold were frozen before this run.')
"""
)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

nb["metadata"]["colab"] = {"name": OUTPUT.name, "provenance": [], "gpuType": "T4"}
OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Generated:", OUTPUT)
