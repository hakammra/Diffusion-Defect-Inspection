"""Local evaluation and verification script for Fabric Stain V4 pipeline."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
V3_RESULTS = ROOT / "artifacts" / "fabric_stain_evaluation_v3"
NPZ_PATH = V3_RESULTS / "test_maps_v3_float16.npz"
CSV_PATH = V3_RESULTS / "per_image_v3.csv"
DATA_ZIP = ROOT / "output" / "datasets" / "fabric_stain_pilot.zip"
OUT_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v4"


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2
        start = end
    positives = labels == 1
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(~positives))
    u = float(np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2)
    return u / (n_pos * n_neg)


def pool_reflect(x: np.ndarray, k: int) -> np.ndarray:
    """Computes exact 2D uniform filter with REFLECT padding across batch x (N, H, W)."""
    p = k // 2
    N, H, W = x.shape
    padded = np.pad(x, ((0, 0), (p, p), (p, p)), mode="reflect")
    padded_cs = np.pad(padded, ((0, 0), (1, 0), (1, 0)), mode="constant")
    np.cumsum(padded_cs, axis=1, out=padded_cs)
    np.cumsum(padded_cs, axis=2, out=padded_cs)
    return (
        padded_cs[:, k : k + H, k : k + W]
        - padded_cs[:, 0:H, k : k + W]
        - padded_cs[:, k : k + H, 0:W]
        + padded_cs[:, 0:H, 0:W]
    ) / (k * k)


def compute_v4_maps(raw_sq: np.ndarray, detrend_k: int = 41, smooth_k: int = 5, margin: int = 12):
    """Computes V4 reflection-padded residual maps and scores."""
    # 1. Background detrending with reflect padding
    bg = pool_reflect(raw_sq, detrend_k)
    detrended = np.maximum(raw_sq - bg, 0.0)

    # 2. Smoothing with reflect padding
    smoothed = pool_reflect(detrended, smooth_k)

    # 3. Per-image background normalization
    flat = smoothed.reshape(len(smoothed), -1)
    medians = np.median(flat, axis=1, keepdims=True)
    q75 = np.quantile(flat, 0.75, axis=1, keepdims=True)
    q25 = np.quantile(flat, 0.25, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-6)
    v4_maps = ((flat - medians) / iqr).reshape(smoothed.shape)

    # 4. Suppress boundary margin to avoid lens/sensor edge vignetting
    if margin > 0:
        v4_maps[:, :margin, :] = 0.0
        v4_maps[:, -margin:, :] = 0.0
        v4_maps[:, :, :margin] = 0.0
        v4_maps[:, :, -margin:] = 0.0

    scores = np.max(v4_maps.reshape(len(v4_maps), -1), axis=1)
    return v4_maps, scores


def hysteresis_segmentation(v4_map: np.ndarray, tau_high: float, tau_low: float, min_area: int = 15) -> np.ndarray:
    """Extracts contiguous anomaly masks via dual-threshold hysteresis region growing."""
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


def colorize_heatmap(residual_map: np.ndarray, vmin: float = 0.0, vmax: float = 35.0) -> Image.Image:
    """Applies a high-contrast colormap to 2D residual map."""
    norm = np.clip((residual_map - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    # Magma-like gradient
    r = np.clip(norm * 2.2, 0, 1)
    g = np.clip(norm * 1.4 - 0.2, 0, 1)
    b = np.clip(norm * 3.0 - 1.2, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(NPZ_PATH)
    raw_maps = data["raw_maps"].astype(np.float32)
    labels = data["labels"].astype(np.int64)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    v4_maps, v4_scores = compute_v4_maps(raw_maps, detrend_k=41, smooth_k=5, margin=12)

    auroc = binary_auroc(labels, v4_scores)

    calibration = json.loads((OUT_DIR / "calibration_v4.json").read_text())
    tau_strict = float(calibration["tau_strict"])
    tau_high = float(calibration["tau_high"])
    tau_low = float(calibration["tau_low"])
    min_area = int(calibration["min_cluster_area"])

    # Compute masks
    hyst_detected = []
    strict_detected = []
    mask_areas = []

    for i in range(len(v4_maps)):
        m_hyst = hysteresis_segmentation(v4_maps[i], tau_high, tau_low, min_area=min_area)
        m_strict = (v4_maps[i] > tau_strict)
        area = int(np.sum(m_hyst))
        hyst_detected.append(bool(np.any(m_hyst)))
        strict_detected.append(bool(np.any(m_strict)))
        mask_areas.append(area)

    hyst_detected = np.array(hyst_detected)
    strict_detected = np.array(strict_detected)

    normal_hyst_fp = int(np.sum(hyst_detected[labels == 0]))
    stain_hyst_tp = int(np.sum(hyst_detected[labels == 1]))
    normal_strict_fp = int(np.sum(strict_detected[labels == 0]))
    stain_strict_tp = int(np.sum(strict_detected[labels == 1]))

    print(f"=== Fabric Stain V4 Verification ===")
    print(f"V4 AUROC: {auroc:.4f}")
    print(f"Strict (tau={tau_strict}): Stain TP={stain_strict_tp}/100, Normal FP={normal_strict_fp}/10")
    print(f"Hysteresis (high={tau_high}, low={tau_low}): Stain TP={stain_hyst_tp}/100, Normal FP={normal_hyst_fp}/10")

    # Generate multi-sample visual preview
    # Select 6 interesting samples:
    # 2 normals (one clean, one crease), 2 strict detections, 2 faint stains recovered by hysteresis
    clean_normal_idx = 0
    crease_normal_idx = 6
    strict_sample_idx = 10
    strict_sample_2 = 14
    # Find faint stains that were missed by strict but recovered by hysteresis
    recovered_indices = np.flatnonzero((labels == 1) & (~strict_detected) & hyst_detected)
    rec1 = int(recovered_indices[0]) if len(recovered_indices) > 0 else 11
    rec2 = int(recovered_indices[1]) if len(recovered_indices) > 1 else 12

    preview_indices = [clean_normal_idx, crease_normal_idx, strict_sample_idx, strict_sample_2, rec1, rec2]

    zip_archive = zipfile.ZipFile(DATA_ZIP)
    preview_rows = []

    for idx in preview_indices:
        r = rows[idx]
        img_rel = f"fabric_stain_pilot/{r['image']}"
        with zip_archive.open(img_rel) as s:
            orig_im = Image.open(io.BytesIO(s.read())).convert("RGB").resize((224, 224), Image.BILINEAR)

        # Heatmap
        heat_im = colorize_heatmap(v4_maps[idx], vmin=0.0, vmax=max(30.0, float(v4_scores[idx]) * 0.8))

        # Strict Mask
        strict_m = v4_maps[idx] > tau_strict
        strict_im = Image.fromarray((strict_m.astype(np.uint8) * 255)).convert("RGB")

        # Hysteresis Mask
        hyst_m = hysteresis_segmentation(v4_maps[idx], tau_high, tau_low, min_area=min_area)
        hyst_im = Image.fromarray((hyst_m.astype(np.uint8) * 255)).convert("RGB")

        # Overlay hysteresis mask in cyan on original
        overlay_arr = np.array(orig_im).astype(np.float32)
        overlay_arr[hyst_m, 0] = overlay_arr[hyst_m, 0] * 0.3
        overlay_arr[hyst_m, 1] = 255
        overlay_arr[hyst_m, 2] = 255
        overlay_im = Image.fromarray(overlay_arr.astype(np.uint8))

        tag = "Normal" if labels[idx] == 0 else ("Strict TP" if strict_detected[idx] else "Recovered")
        preview_rows.append((orig_im, heat_im, strict_im, hyst_im, overlay_im, f"#{idx} {tag} (Score {v4_scores[idx]:.1f})"))

    # Compose visual grid: 6 rows x 5 columns
    cell_w, cell_h = 224, 224
    header_h = 30
    total_w = cell_w * 5 + 60
    total_h = (cell_h + header_h + 10) * len(preview_rows) + 60

    grid_im = Image.new("RGB", (total_w, total_h), (18, 24, 38))
    draw = ImageDraw.Draw(grid_im)

    col_titles = ["Original", "V4 Reflect Heatmap", f"Strict Mask (tau={tau_strict:.1f})", "Hysteresis Full Mask", "Defect Overlay"]
    for c, title in enumerate(col_titles):
        x = 20 + c * (cell_w + 10)
        draw.text((x + 10, 15), title, fill=(200, 220, 240))

    y_off = 50
    for row_idx, (im1, im2, im3, im4, im5, label_txt) in enumerate(preview_rows):
        draw.text((20, y_off - 18), label_txt, fill=(255, 255, 255))
        for col_idx, im in enumerate([im1, im2, im3, im4, im5]):
            x = 20 + col_idx * (cell_w + 10)
            grid_im.paste(im, (x, y_off))
        y_off += cell_h + header_h + 10

    preview_path = OUT_DIR / "evaluation_preview_v4.png"
    grid_im.save(preview_path, quality=92)
    print(f"Saved verification preview to: {preview_path}")

    # Save summary report
    v4_report = {
        "status": "verified",
        "protocol": "Fabric Stain V4 Reflection Detrending + Hysteresis Segmentation",
        "auroc": auroc,
        "tau_strict": tau_strict,
        "tau_high": tau_high,
        "tau_low": tau_low,
        "min_area": min_area,
        "stain_strict_tp": stain_strict_tp,
        "stain_hysteresis_tp": stain_hyst_tp,
        "faint_stains_recovered": int(stain_hyst_tp - stain_strict_tp),
        "normal_strict_fp": normal_strict_fp,
        "normal_hysteresis_fp": normal_hyst_fp,
    }
    (OUT_DIR / "v4_verification_summary.json").write_text(json.dumps(v4_report, indent=2))
    print(json.dumps(v4_report, indent=2))


if __name__ == "__main__":
    run()
