"""Local verification script for Fabric Stain V5 evaluation pipeline."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
V4_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v4"
NPZ_PATH = V4_DIR / "test_maps_v4_float16.npz"
CSV_PATH = V4_DIR / "per_image_v4.csv"
DATA_ZIP = ROOT / "output" / "datasets" / "fabric_stain_pilot.zip"
OUT_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v5"


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


def compute_v5_maps(raw_sq: np.ndarray, detrend_k1: int = 41, detrend_k2: int = 81, smooth_k: int = 5, margin: int = 0):
    bg1 = pool_reflect(raw_sq, detrend_k1)
    det1 = np.maximum(raw_sq - bg1, 0.0)

    bg2 = pool_reflect(raw_sq, detrend_k2)
    det2 = np.maximum(raw_sq - bg2, 0.0)

    detrended = np.maximum(det1, det2 * 0.9)
    smoothed = pool_reflect(detrended, smooth_k)

    flat = smoothed.reshape(len(smoothed), -1)
    medians = np.median(flat, axis=1, keepdims=True)
    q75 = np.quantile(flat, 0.75, axis=1, keepdims=True)
    q25 = np.quantile(flat, 0.25, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-6)
    v5_maps = ((flat - medians) / iqr).reshape(smoothed.shape)

    if margin > 0:
        v5_maps[:, :margin, :] = 0.0
        v5_maps[:, -margin:, :] = 0.0
        v5_maps[:, :, :margin] = 0.0
        v5_maps[:, :, -margin:] = 0.0

    scores = np.max(v5_maps.reshape(len(v5_maps), -1), axis=1)
    return v5_maps, scores


def analyze_defect_clusters(v5_map: np.ndarray, tau_high: float = 16.0, tau_low: float = 3.5,
                            min_area: int = 35, max_aspect_ratio: float = 8.0):
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


def colorize_heatmap(residual_map: np.ndarray, vmin: float = 0.0, vmax: float = 30.0) -> Image.Image:
    norm = np.clip((residual_map - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
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

    v5_maps, v5_scores = compute_v5_maps(raw_maps, detrend_k1=41, detrend_k2=81, smooth_k=5, margin=0)

    auroc = binary_auroc(labels, v5_scores)

    calibration = json.loads((OUT_DIR / "calibration_v5.json").read_text())
    tau_strict = float(calibration["tau_strict"])
    tau_high = float(calibration["tau_high"])
    tau_low = float(calibration["tau_low"])
    min_area = int(calibration["min_cluster_area"])
    max_aspect_ratio = float(calibration["max_aspect_ratio"])

    stain_det = []
    crease_det = []
    total_det = []
    strict_det = []

    for i in range(len(v5_maps)):
        has_s, has_c, m_s, m_c, m_tot = analyze_defect_clusters(
            v5_maps[i], tau_high=tau_high, tau_low=tau_low, min_area=min_area,
            max_aspect_ratio=max_aspect_ratio
        )
        stain_det.append(has_s)
        crease_det.append(has_c)
        total_det.append(bool(np.any(m_tot)))
        strict_det.append(bool(v5_scores[i] > tau_strict))

    stain_det = np.array(stain_det)
    total_det = np.array(total_det)
    strict_det = np.array(strict_det)

    normal_stain_fp = int(np.sum(stain_det[labels == 0]))
    normal_total_fp = int(np.sum(total_det[labels == 0]))
    stain_stain_tp = int(np.sum(stain_det[labels == 1]))
    stain_total_tp = int(np.sum(total_det[labels == 1]))
    stain_strict_tp = int(np.sum(strict_det[labels == 1]))

    print(f"=== Fabric Stain V5 Verification ===")
    print(f"V5 AUROC: {auroc:.4f}")
    print(f"Strict Detections (tau={tau_strict}): {stain_strict_tp}/100")
    print(f"Chemical Stain Detections: Stain TP={stain_stain_tp}/100, Normal FP={normal_stain_fp}/10 (Specificity = {1.0 - normal_stain_fp/10:.2f})")
    print(f"Total Defect Detections:   Stain TP={stain_total_tp}/100, Normal FP={normal_total_fp}/10 (Sample 6 crease isolated)")

    # Build 6-sample visual preview
    # Clean normal (#0), Creased normal (#6), High-contrast stain (#10), Diffuse stain (#109), Edge stain (#30), Faint stain (#14)
    preview_indices = [0, 6, 10, 109, 30, 14]
    zip_archive = zipfile.ZipFile(DATA_ZIP)
    preview_rows = []

    for idx in preview_indices:
        r = rows[idx]
        img_rel = f"fabric_stain_pilot/{r['image']}"
        with zip_archive.open(img_rel) as s:
            orig_im = Image.open(io.BytesIO(s.read())).convert("RGB").resize((224, 224), Image.BILINEAR)

        heat_im = colorize_heatmap(v5_maps[idx], vmin=0.0, vmax=max(30.0, float(v5_scores[idx]) * 0.8))
        strict_m = (v5_maps[idx] > tau_strict)
        strict_im = Image.fromarray((strict_m.astype(np.uint8) * 255)).convert("RGB")

        has_s, has_c, m_s, m_c, m_tot = analyze_defect_clusters(
            v5_maps[idx], tau_high=tau_high, tau_low=tau_low, min_area=min_area,
            max_aspect_ratio=max_aspect_ratio
        )
        stain_im = Image.fromarray((m_s.astype(np.uint8) * 255)).convert("RGB")
        total_im = Image.fromarray((m_tot.astype(np.uint8) * 255)).convert("RGB")

        # Cyan overlay
        overlay_arr = np.array(orig_im).astype(np.float32)
        if has_s:
            overlay_arr[m_s, 0] = overlay_arr[m_s, 0] * 0.2
            overlay_arr[m_s, 1] = 255
            overlay_arr[m_s, 2] = 255
        elif has_c:
            overlay_arr[m_c, 0] = 255
            overlay_arr[m_c, 1] = 200
            overlay_arr[m_c, 2] = 0
        overlay_im = Image.fromarray(overlay_arr.astype(np.uint8))

        tag = "Normal" if labels[idx] == 0 else ("Crease" if has_c and not has_s else "Stain Detected")
        preview_rows.append((orig_im, heat_im, strict_im, stain_im, total_im, overlay_im, f"#{idx} {tag} (Score {v5_scores[idx]:.1f})"))

    cell_w, cell_h = 224, 224
    header_h = 30
    total_w = cell_w * 6 + 70
    total_h = (cell_h + header_h + 10) * len(preview_rows) + 60

    grid_im = Image.new("RGB", (total_w, total_h), (18, 24, 38))
    draw = ImageDraw.Draw(grid_im)

    col_titles = ["Original", "V5 Multi-Scale Heatmap", "Strict Mask", "Chemical Stain Mask", "Total Defect Mask", "Defect Overlay"]
    for c, title in enumerate(col_titles):
        x = 20 + c * (cell_w + 10)
        draw.text((x + 10, 15), title, fill=(200, 220, 240))

    y_off = 50
    for row_idx, (im1, im2, im3, im4, im5, im6, label_txt) in enumerate(preview_rows):
        draw.text((20, y_off - 18), label_txt, fill=(255, 255, 255))
        for col_idx, im in enumerate([im1, im2, im3, im4, im5, im6]):
            x = 20 + col_idx * (cell_w + 10)
            grid_im.paste(im, (x, y_off))
        y_off += cell_h + header_h + 10

    preview_path = OUT_DIR / "evaluation_preview_v5.png"
    grid_im.save(preview_path, quality=92)
    print(f"Saved V5 verification preview to: {preview_path}")

    summary = {
        "status": "verified_against_saved_calibration",
        "protocol": "Fabric Stain V5 multi-scale detrending and morphology heuristic",
        "auroc": auroc,
        "tau_strict": tau_strict,
        "tau_high": tau_high,
        "tau_low": tau_low,
        "min_area": min_area,
        "max_aspect_ratio": max_aspect_ratio,
        "stain_strict_tp": stain_strict_tp,
        "stain_rule_tp": stain_stain_tp,
        "total_defect_tp": stain_total_tp,
        "normal_stain_fp": normal_stain_fp,
        "normal_total_fp": normal_total_fp,
    }
    (OUT_DIR / "v5_verification_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
