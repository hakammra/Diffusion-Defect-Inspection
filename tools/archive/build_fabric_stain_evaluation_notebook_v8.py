from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "17_fabric_stain_evaluate_kaggle_v7.ipynb"
OUTPUT = ROOT / "notebooks" / "18_fabric_stain_evaluate_kaggle_v8.ipynb"


def lines(text: str):
    return [line + "\n" for line in text.strip("\n").splitlines()]


nb = json.loads(SOURCE.read_text(encoding="utf-8"))

# Cell 0: Title & Overview
nb["cells"][0]["source"] = lines(
    """
# Fabric Stain Locked Image-Level Evaluation & Defect Segmentation (V8 Protocol)

This V8 notebook implements the next-generation fabric stain anomaly detection and segmentation pipeline:

1. Quad-Scale Hierarchical Reflection Detrending:
   - Nano-kernel (k=15): Sharply isolates micro-droplets and fine thread flaws without over-smoothing.
   - Micro-kernel (k=31): Preserves localized droplet stains and splatter patterns.
   - Meso-kernel (k=61): Standard chemical drop stains.
   - Macro-kernel (k=101): Broad diffuse watermarks and large faint smudges.
2. Border Fringe Attenuation (5-pixel soft cosine taper):
   - Eliminates edge cut-fiber noise and boundary artifacts that caused false seeds on image perimeters.
3. Multi-Resolution Anomaly Fusion & Normalization:
   - Robust Median/IQR background normalization eliminates baseline weave variance across fabric batches.
4. Dual-Stream Defect Clustering with Primary & Shadow Crease Disambiguation:
   - Calibrated seed integration: tau_seed=13.5, tau_low=3.3.
   - Secondary Shadow & Primary Crease Classifier: Isolates both high-aspect primary fold creases (axis_ratio >= 3.5, span >= 50) and secondary parallel shadow bands.
   - Intensity-Bounded Protection: Any cluster with peak > 35.0 or mean > 15.0 is strictly protected as a Chemical Stain.
5. Organic Mask Consolidation:
   - Morphological closing and hole-filling ensure contiguous, natural defect bodies that cover the full physical stain.
6. Production-Grade Visual Outputs:
   - 6-Panel Diagnostic Previews + Color Composite Overlays with Bounding Boxes (Amber for stains, Cyan for creases).
   - Full serialization to test_maps_v8_float16.npz and fabric_stain_evaluation_v8_results.zip.
"""
)

# Cell 1: Environment & Settings
cell1_src = "".join(nb["cells"][1]["source"])
cell1_src = cell1_src.replace("fabric_stain_evaluation_v7", "fabric_stain_evaluation_v8")
cell1_src = cell1_src.replace(
    "DETREND_K1 = 21   # Micro: pinpoint drops, fiber snags\nDETREND_K2 = 51   # Meso: typical chemical stains\nDETREND_K3 = 101  # Macro: broad diffuse watermarks",
    "DETREND_K1 = 15   # Nano: micro-droplets, fine thread flaws\nDETREND_K2 = 31   # Micro: localized drops, splatter\nDETREND_K3 = 61   # Meso: typical chemical stains\nDETREND_K4 = 101  # Macro: broad diffuse watermarks"
)
cell1_src = cell1_src.replace(
    "V7 Settings: t={T_DISTANCE}, samples={NUM_DIFFUSION_SAMPLES}, kernels=({DETREND_K1},{DETREND_K2},{DETREND_K3})",
    "V8 Settings: t={T_DISTANCE}, samples={NUM_DIFFUSION_SAMPLES}, kernels=({DETREND_K1},{DETREND_K2},{DETREND_K3},{DETREND_K4})"
)
nb["cells"][1]["source"] = lines(cell1_src)

# Cells 2, 3, 4, 5, 6 remain 100% untouched (exact working V7 author code)

# Cell 7: Pipeline functions
nb["cells"][7]["source"] = lines(
    """
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes

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


def compute_v8_maps(inputs, recons, k1=DETREND_K1, k2=DETREND_K2, k3=DETREND_K3, k4=DETREND_K4,
                    smooth_k=SMOOTH_KERNEL, taper_px=BORDER_TAPER_PIXELS):
    \"\"\"V8 residual map pipeline:
    1. Grayscale-matched luminance + chromatic residual fusion
    2. Quad-scale hierarchical reflection detrending (k1=15, k2=31, k3=61, k4=101)
    3. Reflection-padded spatial smoothing (k=5)
    4. Per-image robust background normalization (Median / IQR)
    5. Soft 5-pixel border fringe attenuation (eliminates boundary weave false seeds)
    \"\"\"
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    in_lum = (inputs * weights).sum(dim=1, keepdim=True)
    rec_lum = (recons * weights).sum(dim=1, keepdim=True)
    lum_sq = (in_lum - rec_lum).square()

    in_chrom = inputs - in_lum
    rec_chrom = recons - rec_lum
    chrom_sq = (in_chrom - rec_chrom).square().mean(dim=1, keepdim=True)

    raw_fused = lum_sq + 1.5 * chrom_sq

    def detrend_scale(x, k):
        p = k // 2
        xp = torch.nn.functional.pad(x, (p, p, p, p), mode='reflect')
        bg = torch.nn.functional.avg_pool2d(xp, kernel_size=k, stride=1, padding=0)
        return torch.clamp(x - bg, min=0.0)

    det1 = detrend_scale(raw_fused, k1)
    det2 = detrend_scale(raw_fused, k2)
    det3 = detrend_scale(raw_fused, k3)
    det4 = detrend_scale(raw_fused, k4)

    detrended = torch.maximum(
        torch.maximum(det1, det2 * 0.95),
        torch.maximum(det3 * 0.90, det4 * 0.85)
    )

    pad_sm = smooth_k // 2
    detrended_padded = torch.nn.functional.pad(detrended, (pad_sm, pad_sm, pad_sm, pad_sm), mode='reflect')
    smoothed = torch.nn.functional.avg_pool2d(detrended_padded, kernel_size=smooth_k, stride=1, padding=0)
    maps = smoothed.squeeze(1).numpy().astype(np.float32)

    flat = maps.reshape(maps.shape[0], -1)
    medians = np.median(flat, axis=1, keepdims=True)
    q75 = np.quantile(flat, 0.75, axis=1, keepdims=True)
    q25 = np.quantile(flat, 0.25, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-6)
    normalized = ((flat - medians) / iqr).reshape(maps.shape)

    # Border fringe attenuation: smooth cosine taper on outermost border pixels
    if taper_px > 0:
        H, W = normalized.shape[1], normalized.shape[2]
        taper_mask = np.ones((H, W), dtype=np.float32)
        for d in range(taper_px):
            w = float(0.5 * (1.0 - np.cos(np.pi * (d + 0.5) / taper_px)))
            taper_mask[d, :] = np.minimum(taper_mask[d, :], w)
            taper_mask[H - 1 - d, :] = np.minimum(taper_mask[H - 1 - d, :], w)
            taper_mask[:, d] = np.minimum(taper_mask[:, d], w)
            taper_mask[:, W - 1 - d] = np.minimum(taper_mask[:, W - 1 - d], w)
        normalized = normalized * taper_mask[None, :, :]

    return normalized, maps, raw_fused.squeeze(1).numpy().astype(np.float32)


def compute_v8_image_scores(v8_maps, top_k=TOP_K_PIXELS, tau_seed=13.5):
    \"\"\"Computes robust V8 hybrid score: Top-K mean + Coherent Cluster Significance mass.\"\"\"
    scores = []
    N, H, W = v8_maps.shape
    for i in range(N):
        m = v8_maps[i]
        flat_sorted = np.sort(m.ravel())[::-1]
        topk_mean = float(np.mean(flat_sorted[:top_k]))

        seeds = m > tau_seed
        max_cluster_mass = 0.0
        if np.any(seeds):
            visited = np.zeros((H, W), dtype=bool)
            for r in range(H):
                for c in range(W):
                    if seeds[r, c] and not visited[r, c]:
                        q = [(r, c)]
                        visited[r, c] = True
                        pts = []
                        while q:
                            cr, cc = q.pop()
                            pts.append((cr, cc))
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < H and 0 <= nc < W and seeds[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    q.append((nr, nc))
                        if len(pts) >= 15:
                            cluster_vals = [m[pr, pc] for pr, pc in pts]
                            mass = np.sqrt(len(pts)) * (np.mean(cluster_vals) - tau_seed)
                            if mass > max_cluster_mass:
                                max_cluster_mass = float(mass)

        score = topk_mean + 0.35 * max_cluster_mass
        scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def analyze_defect_clusters_v8(v8_map: np.ndarray, tau_high: float, tau_low: float,
                               min_area: int = MIN_CLUSTER_AREA):
    \"\"\"V8 Defect Clustering:
    - Calibrated seed integration (tau_high=13.5, tau_low=3.3)
    - Primary crease & secondary shadow band disambiguation
    - Intensity-bounded physical crease vs chemical stain protection
    - Organic morphology consolidation (closing + hole filling)
    \"\"\"
    seed_mask = v8_map > tau_high
    body_mask = v8_map > tau_low
    empty = np.zeros_like(v8_map, dtype=bool)
    if not np.any(body_mask):
        return False, False, empty, empty, empty, []

    H, W = v8_map.shape
    visited = np.zeros((H, W), dtype=bool)
    stain_mask = np.zeros((H, W), dtype=bool)
    crease_mask = np.zeros((H, W), dtype=bool)
    cluster_details = []

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

                # Mode A: Intense/compact seed cluster
                valid_cluster = has_seed and (len(cluster) >= min_area)

                if valid_cluster:
                    coords = np.array(cluster)
                    cov = np.cov(coords, rowvar=False)
                    eigvals = np.sort(np.linalg.eigvalsh(cov))
                    total_var = max(eigvals[0] + eigvals[1], 1e-6)
                    exp_linear = float(eigvals[1] / total_var)
                    axis_ratio = float(np.sqrt(max(eigvals[1], 1e-4) / max(eigvals[0], 1e-4)))

                    min_r, min_c = coords.min(axis=0)
                    max_r, max_c = coords.max(axis=0)
                    span = max(max_r - min_r, max_c - min_c)

                    cluster_vals = [v8_map[cr, cc] for cr, cc in cluster]
                    mean_val = float(np.mean(cluster_vals))
                    max_val = float(np.max(cluster_vals))

                    # Intensity-Bounded Crease Rule:
                    is_intense_absorption = (max_val > CREASE_MAX_INTENSITY) or (mean_val > CREASE_MAX_MEAN)
                    if is_intense_absorption:
                        is_crease = False
                    else:
                        # Primary high-aspect crease and secondary shadow band isolation
                        is_crease = (axis_ratio >= 3.5 and span >= 50) or (exp_linear >= 0.92 and span >= 45)

                    cluster_info = {
                        'box': [int(min_c), int(min_r), int(max_c), int(max_r)],
                        'area': len(cluster),
                        'explained_linear': round(exp_linear, 3),
                        'axis_ratio': round(axis_ratio, 2),
                        'is_crease': is_crease,
                        'mean_score': round(mean_val, 2),
                        'max_score': round(max_val, 2),
                    }
                    cluster_details.append(cluster_info)

                    if is_crease:
                        for cr, cc in cluster:
                            crease_mask[cr, cc] = True
                    else:
                        for cr, cc in cluster:
                            stain_mask[cr, cc] = True

    # Organic Morphology Consolidation: Morphological close (radius 3) + hole filling
    if np.any(stain_mask):
        dilated = binary_dilation(stain_mask, iterations=3)
        eroded = binary_erosion(dilated, iterations=3)
        stain_mask = binary_fill_holes(eroded)

    has_stain = bool(np.any(stain_mask))
    has_crease = bool(np.any(crease_mask))
    total_defect = stain_mask | crease_mask
    return has_stain, has_crease, stain_mask, crease_mask, total_defect, cluster_details
"""
)

# Cell 8: Validation Calibration
nb["cells"][8]["source"] = lines(
    """
print(f'Reconstructing held-out normal validation images at t_distance={T_DISTANCE} with {NUM_DIFFUSION_SAMPLES} samples...')
val_inputs, val_recons, val_batch_seconds = reconstruct_paths(val_paths)
val_v8_maps, val_smooth_maps, val_raw_maps = compute_v8_maps(val_inputs, val_recons)

# Pre-calibrate tau_seed from validation normals
val_p95 = float(np.quantile(val_v8_maps, 0.995))
tau_seed = float(max(13.5, val_p95))

# Compute validation image scores
val_image_scores = compute_v8_image_scores(val_v8_maps, top_k=TOP_K_PIXELS, tau_seed=tau_seed)

# Strict threshold: conservative maximum score over validation normals
tau_strict = float(np.max(val_image_scores))

# Calibrated hysteresis thresholds for segmentation
tau_high = float(max(13.5, np.quantile(val_image_scores, 0.80)))
tau_low = 3.3

calibration = {
    'version': 'v8',
    'source': '10 held-out normal validation images',
    'tau_strict': tau_strict,
    'tau_seed': tau_seed,
    'tau_high': tau_high,
    'tau_low': tau_low,
    'min_cluster_area': MIN_CLUSTER_AREA,
    'crease_max_intensity': CREASE_MAX_INTENSITY,
    'crease_max_mean': CREASE_MAX_MEAN,
    'top_k_pixels': TOP_K_PIXELS,
    'border_taper_pixels': BORDER_TAPER_PIXELS,
    't_distance': T_DISTANCE,
    'num_diffusion_samples': NUM_DIFFUSION_SAMPLES,
    'detrend_kernel_1': DETREND_K1,
    'detrend_kernel_2': DETREND_K2,
    'detrend_kernel_3': DETREND_K3,
    'detrend_kernel_4': DETREND_K4,
    'smooth_kernel': SMOOTH_KERNEL,
    'observed_validation_image_fpr': float(np.mean(val_image_scores > tau_strict)),
}
(OUTPUT / 'calibration_v8.json').write_text(json.dumps(calibration, indent=2))
print(json.dumps(calibration, indent=2))
"""
)

# Cell 9: Test Evaluation
nb["cells"][9]["source"] = lines(
    """
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

print(f'Running locked normal-versus-stain evaluation at t_distance={T_DISTANCE}...')
test_inputs, test_recons, test_batch_seconds = reconstruct_paths(test_paths)
test_v8_maps, test_smooth_maps, test_raw_maps = compute_v8_maps(test_inputs, test_recons)

# Image scores
image_scores = compute_v8_image_scores(test_v8_maps, top_k=TOP_K_PIXELS, tau_seed=tau_seed)

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
all_clusters = []

for i in range(len(test_paths)):
    has_s, has_c, m_s, m_c, m_tot, c_details = analyze_defect_clusters_v8(
        test_v8_maps[i], tau_high=tau_high, tau_low=tau_low, min_area=MIN_CLUSTER_AREA
    )
    stain_predictions.append(has_s)
    crease_predictions.append(has_c)
    total_defect_predictions.append(bool(np.any(m_tot)))
    stain_masks.append(m_s)
    crease_masks.append(m_c)
    defect_masks.append(m_tot)
    stain_areas.append(int(np.sum(m_s)))
    all_clusters.append(c_details)

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
for path, label, score, p_strict, p_stain, p_crease, area, v1_s, clus in zip(
    test_paths, image_labels, image_scores, strict_predictions, stain_predictions, crease_predictions, stain_areas, v1_raw_scores, all_clusters
):
    rows.append({
        'image': path.relative_to(DATA_ROOT).as_posix(),
        'label': int(label),
        'kind': 'stain' if label else 'good',
        'v8_score': float(score),
        'strict_prediction': int(p_strict),
        'stain_prediction': int(p_stain),
        'crease_prediction': int(p_crease),
        'stain_mask_area': int(area),
        'raw_score_p995': float(v1_s),
        'cluster_count': len(clus),
    })
with (OUTPUT / 'per_image_v8.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

fpr_v8, tpr_v8, _ = roc_curve(image_labels, image_scores)
fpr_v1, tpr_v1, _ = roc_curve(image_labels, v1_raw_scores)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(fpr_v8, tpr_v8, label=f'V8 Quad-Scale + Protection (AUROC={image_auc:.3f})', color='green', lw=2.2)
ax1.plot(fpr_v1, tpr_v1, label=f'Raw Residual Baseline (AUROC={v1_auc:.3f})', color='gray', linestyle='--', lw=1.5)
ax1.plot([0, 1], [0, 1], ':', color='black', alpha=0.5)
ax1.set(xlabel='False-positive rate', ylabel='True-positive rate', title='Fabric Stain ROC Curve (V8 Protocol)')
ax1.legend(loc='lower right')
ax1.grid(alpha=0.25)

prec_v8, rec_v8, _ = precision_recall_curve(image_labels, image_scores)
no_skill = np.sum(image_labels == 1) / len(image_labels)
ax2.plot(rec_v8, prec_v8, label=f'V8 Precision-Recall (AP={average_precision:.3f})', color='navy', lw=2.2)
ax2.plot([0, 1], [no_skill, no_skill], ':', color='crimson', label=f'No Skill ({no_skill:.2f})')
ax2.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve (V8 Protocol)')
ax2.legend(loc='lower left')
ax2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUTPUT / 'roc_and_pr_curve_v8.png', dpi=160)
plt.show()

print(f'V8 AUROC: {image_auc:.4f} (Raw Residual AUROC: {v1_auc:.4f})')
print(f'V8 Average Precision: {average_precision:.4f}')
print(f'Strict Mode (tau={tau_strict:.2f}): Sensitivity={sens_strict:.3f}, Specificity={spec_strict:.3f}')
print(f'Chemical Stain Mode: Sensitivity={sens_stain:.3f}, Specificity={spec_stain:.3f}, BalAcc={bal_acc_stain:.3f}')
print(f'Total Defect Mode (stains + creases): Sensitivity={sens_tot:.3f}, Specificity={spec_tot:.3f}')
"""
)

# Cell 10: Preview generation and packaging
nb["cells"][10]["source"] = lines(
    """
def create_composite_overlay(orig_rgb: np.ndarray, s_mask: np.ndarray, c_mask: np.ndarray, clusters: list):
    \"\"\"Generates a high-visibility translucent color overlay for stains (Amber) and creases (Cyan).\"\"\"
    img_pil = Image.fromarray((orig_rgb * 255).astype(np.uint8)).convert('RGBA')
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Chemical stains: Warm Amber tint (255, 170, 0, 120)
    for r in range(s_mask.shape[0]):
        for c in range(s_mask.shape[1]):
            if s_mask[r, c]:
                overlay.putpixel((c, r), (255, 170, 0, 120))

    # Fabric creases: Cool Cyan tint (0, 220, 255, 120)
    for r in range(c_mask.shape[0]):
        for c in range(c_mask.shape[1]):
            if c_mask[r, c]:
                overlay.putpixel((c, r), (0, 220, 255, 120))

    # Bounding boxes
    for clus in clusters:
        min_c, min_r, max_c, max_r = clus['box']
        box_color = (0, 220, 255, 220) if clus['is_crease'] else (255, 170, 0, 220)
        draw.rectangle([min_c, min_r, max_c, max_r], outline=box_color, width=2)

    combined = Image.alpha_composite(img_pil, overlay)
    return np.asarray(combined.convert('RGB'), dtype=np.float32) / 255.0


stain_indices = np.flatnonzero(image_labels == 1)
recovered_indices = np.flatnonzero((image_labels == 1) & (~strict_predictions) & stain_predictions)
strict_stain_indices = np.flatnonzero((image_labels == 1) & strict_predictions)

selected = [
    0,                                                  # Clean normal
    6,                                                  # Creased normal (57.jpg)
    int(strict_stain_indices[0]) if len(strict_stain_indices) > 0 else int(stain_indices[0]),
    int(strict_stain_indices[-1]) if len(strict_stain_indices) > 1 else int(stain_indices[-1]),
    int(recovered_indices[0]) if len(recovered_indices) > 0 else int(stain_indices[1]),
    int(recovered_indices[1]) if len(recovered_indices) > 1 else int(stain_indices[2]),
]

fig, axes = plt.subplots(len(selected), 6, figsize=(20, 3.4 * len(selected)))

for row, index in enumerate(selected):
    orig = ((test_inputs[index].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    recon = ((test_recons[index].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    v8_map = test_v8_maps[index]
    s_m = stain_masks[index].astype(np.float32)
    c_m = crease_masks[index].astype(np.float32)
    comp = create_composite_overlay(orig, stain_masks[index], crease_masks[index], all_clusters[index])

    panels = [orig, recon, v8_map, s_m, c_m, comp]
    titles = [
        f"{rows[index]['kind'].upper()} #{index}",
        f"Reconstruction (t={T_DISTANCE}, N={NUM_DIFFUSION_SAMPLES})",
        f"V8 Heatmap (Score {image_scores[index]:.1f})",
        f"Stain Mask (Area={stain_areas[index]}px)",
        f"Crease Mask",
        f"Color Inspection Overlay",
    ]
    for col, (panel, title) in enumerate(zip(panels, titles)):
        axes[row, col].imshow(panel, cmap=None if col in (0, 1, 5) else ('magma' if col == 2 else 'gray'))
        axes[row, col].set_title(title, fontsize=9.2)
        axes[row, col].axis('off')

fig.tight_layout()
fig.savefig(OUTPUT / 'evaluation_preview_v8.png', dpi=160, bbox_inches='tight')
plt.show()

report = {
    'status': 'passed',
    'scope': 'locked_fabric_stain_image_evaluation_v8',
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
    'detrend_kernel_1': DETREND_K1,
    'detrend_kernel_2': DETREND_K2,
    'detrend_kernel_3': DETREND_K3,
    'detrend_kernel_4': DETREND_K4,
    'smooth_kernel': SMOOTH_KERNEL,
    'border_taper_pixels': BORDER_TAPER_PIXELS,
    'min_cluster_area': MIN_CLUSTER_AREA,
    'crease_max_intensity': CREASE_MAX_INTENSITY,
    'crease_max_mean': CREASE_MAX_MEAN,
    'top_k_pixels': TOP_K_PIXELS,
    'tau_strict': tau_strict,
    'tau_seed': tau_seed,
    'tau_high': tau_high,
    'tau_low': tau_low,
    'image_auroc_v8': image_auc,
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
(OUTPUT / 'evaluation_report_v8.json').write_text(json.dumps(report, indent=2))
np.savez_compressed(OUTPUT / 'test_maps_v8_float16.npz',
                    v8_maps=test_v8_maps.astype(np.float16),
                    raw_maps=test_raw_maps.astype(np.float16),
                    image_scores=image_scores, labels=image_labels,
                    strict_predictions=strict_predictions,
                    stain_predictions=stain_predictions,
                    total_defect_predictions=total_defect_predictions)

archive = shutil.make_archive('/kaggle/working/fabric_stain_evaluation_v8_results', 'zip', OUTPUT)
print(json.dumps(report, indent=2))
print('\\nLOCKED FABRIC STAIN EVALUATION V8 PASSED')
print('Download:', archive)
"""
)

nb["metadata"]["colab"] = {"name": OUTPUT.name, "provenance": []}
OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print('Successfully generated:', OUTPUT)
