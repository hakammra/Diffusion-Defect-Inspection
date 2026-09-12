from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
INPUT_NPZ = ROOT / "artifacts" / "fabric_stain_evaluation_v2" / "test_maps_v2_float16.npz"
OUTPUT_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v3"


def fast_box_blur_2d(img: np.ndarray, r: int) -> np.ndarray:
    N, H, W = img.shape
    pad_h = np.pad(img, ((0, 0), (r + 1, r), (0, 0)), mode="edge")
    cs_h = np.cumsum(pad_h, axis=1)
    blur_h = (cs_h[:, 2 * r + 1 :, :] - cs_h[:, :H, :]) / (2 * r + 1)

    pad_w = np.pad(blur_h, ((0, 0), (0, 0), (r + 1, r)), mode="edge")
    cs_w = np.cumsum(pad_w, axis=2)
    blur = (cs_w[:, :, 2 * r + 1 :] - cs_w[:, :, :W]) / (2 * r + 1)
    return blur


def filter_connected_components(binary_mask: np.ndarray, min_area: int = 10) -> np.ndarray:
    H, W = binary_mask.shape
    visited = np.zeros((H, W), dtype=bool)
    cleaned = np.zeros((H, W), dtype=bool)

    for r in range(H):
        for c in range(W):
            if binary_mask[r, c] and not visited[r, c]:
                component = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop()
                    component.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and binary_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                if len(component) >= min_area:
                    for cr, cc in component:
                        cleaned[cr, cc] = True
    return cleaned


def compute_roc(y_true: np.ndarray, y_score: np.ndarray):
    desc = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc]
    y_true_sorted = y_true[desc]
    distinct = np.where(np.diff(y_score_sorted))[0]
    idxs = np.r_[distinct, y_true_sorted.size - 1]
    tps = np.cumsum(y_true_sorted)[idxs]
    fps = 1 + idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    auroc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auroc


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray):
    desc = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc]
    cum_tp = np.cumsum(y_true_sorted)
    precisions = cum_tp / (np.arange(len(y_true_sorted)) + 1)
    pos_indices = np.where(y_true_sorted == 1)[0]
    if len(pos_indices) == 0:
        return 0.0
    return float(np.mean(precisions[pos_indices]))


def draw_roc_curves(curves_dict: dict[str, tuple[np.ndarray, np.ndarray, float]], out_path: Path):
    width, height = 720, 600
    margin = 80
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    for step in range(11):
        x = margin + int(step * plot_w / 10)
        y = margin + int(step * plot_h / 10)
        draw.line([(x, margin), (x, height - margin)], fill="#E5E7EB", width=1)
        draw.line([(margin, y), (width - margin, y)], fill="#E5E7EB", width=1)
        v = step / 10.0
        draw.text((x - 8, height - margin + 8), f"{v:.1f}", fill="#4B5563")
        draw.text((margin - 30, height - margin - int(step * plot_h / 10) - 6), f"{v:.1f}", fill="#4B5563")

    draw.line([(margin, height - margin), (width - margin, margin)], fill="#9CA3AF", width=2)
    draw.rectangle([(margin, margin), (width - margin, height - margin)], outline="#111827", width=2)
    draw.text((width // 2 - 60, height - margin + 30), "False Positive Rate", fill="#111827")
    draw.text((15, height // 2), "TPR", fill="#111827")
    draw.text((margin, 25), "Fabric Stain ROC Progression: V1 vs V2 vs V3", fill="#111827")

    color_map = {
        "V1 Baseline (t=250, Raw p99.5)": "#DC2626",
        "V2 Protocol (t=50, Smooth+IQR)": "#2563EB",
        "V3 Detrended (t=50, Detrend+Smooth)": "#D97706",
        "V3 Detrended + Area Filter (t=50)": "#059669",
    }

    legend_y = margin + 20
    for name, (fpr, tpr, auroc) in curves_dict.items():
        color = color_map.get(name, "#4B5563")
        points = []
        for fx, ty in zip(fpr, tpr):
            px = margin + int(fx * plot_w)
            py = height - margin - int(ty * plot_h)
            points.append((px, py))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)

        draw.rectangle([(width - margin - 320, legend_y), (width - margin - 295, legend_y + 12)], fill=color)
        draw.text((width - margin - 285, legend_y), f"{name}: {auroc:.3f}", fill="#1F2937")
        legend_y += 24

    img.save(out_path)


def run():
    if not INPUT_NPZ.is_file():
        raise FileNotFoundError(f"Missing residual maps file: {INPUT_NPZ}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(INPUT_NPZ)
    raw_maps = data["raw_maps"].astype(np.float32)  # (110, 224, 224)
    v2_maps = data["v2_maps"].astype(np.float32)  # (110, 224, 224)
    labels = data["labels"].astype(np.int64)

    num_normals = int(np.sum(labels == 0))
    num_stains = int(np.sum(labels == 1))
    print(f"Loaded test maps: {num_normals} normal and {num_stains} stain.")

    print("Computing V3 Low-Frequency Detrending (Kernel 41)...")
    bg = fast_box_blur_2d(raw_maps, 20)
    detrended = np.maximum(raw_maps - bg, 0)
    smooth = fast_box_blur_2d(detrended, 2)  # 5x5 smoothing

    flat = smooth.reshape(110, -1)
    med = np.median(flat, axis=1, keepdims=True)
    q75 = np.quantile(flat, 0.75, axis=1, keepdims=True)
    q25 = np.quantile(flat, 0.25, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-6)
    v3_maps = ((flat - med) / iqr).reshape(110, 224, 224)

    print("Computing V3 Connected-Component Area Filtered Scores (min_area=10)...")
    v3_area_scores = []
    v3_clean_masks = []
    pixel_threshold = 5.0
    min_area = 10

    for i in range(len(v3_maps)):
        bin_mask = v3_maps[i] > pixel_threshold
        cleaned_mask = filter_connected_components(bin_mask, min_area=min_area)
        v3_clean_masks.append(cleaned_mask)
        if np.any(cleaned_mask):
            score = float(np.mean(v3_maps[i][cleaned_mask]))
        else:
            score = 0.0
        v3_area_scores.append(score)
    v3_area_scores = np.array(v3_area_scores)

    # Strategies to compare
    strategies = {
        "V1 Baseline (t=250, Raw p99.5)": np.load(ROOT / "artifacts" / "fabric_stain_evaluation" / "test_maps_float16.npz")["image_scores"],
        "V2 Protocol (t=50, Smooth+IQR)": data["image_scores"],
        "V3 Detrended (t=50, Detrend+Smooth)": np.max(v3_maps, axis=(1, 2)),
        "V3 Detrended + Area Filter (t=50)": v3_area_scores,
    }

    results = []
    curves_for_plot = {}

    for name, img_scores in strategies.items():
        fpr, tpr, auroc = compute_roc(labels, img_scores)
        ap = compute_average_precision(labels, img_scores)

        normal_scores = np.sort(img_scores[labels == 0])
        stain_scores = img_scores[labels == 1]

        thresh_100 = float(normal_scores[-1])
        tp_100 = int(np.sum(stain_scores > thresh_100))
        fp_100 = int(np.sum(normal_scores > thresh_100))
        sens_100 = tp_100 / num_stains

        record = {
            "strategy": name,
            "auroc": round(auroc, 4),
            "average_precision": round(ap, 4),
            "threshold_spec100": round(thresh_100, 4),
            "sensitivity_spec100": round(sens_100, 4),
            "specificity_spec100": 1.0,
            "tp_spec100": tp_100,
            "fp_spec100": fp_100,
        }
        results.append(record)
        curves_for_plot[name] = (fpr, tpr, auroc)

    # Save summary table
    with (OUTPUT_DIR / "comparison_table_v3.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save per-image CSV
    v2_per_image = ROOT / "artifacts" / "fabric_stain_evaluation_v2" / "per_image_v2.csv"
    orig_rows = []
    if v2_per_image.is_file():
        with v2_per_image.open("r") as f:
            orig_rows = list(csv.DictReader(f))

    v3_thresh = results[3]["threshold_spec100"]
    per_image_rows = []
    for idx in range(len(labels)):
        img_name = orig_rows[idx]["image"] if idx < len(orig_rows) else f"image_{idx}"
        kind = "stain" if labels[idx] == 1 else "good"
        score = float(v3_area_scores[idx])
        pred = int(score > v3_thresh)
        per_image_rows.append(
            {
                "image": img_name,
                "label": int(labels[idx]),
                "kind": kind,
                "v2_score": float(data["image_scores"][idx]),
                "v3_clean_score": round(score, 4),
                "v3_prediction_spec100": pred,
            }
        )

    with (OUTPUT_DIR / "per_image_v3.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    # Plot ROC curves
    draw_roc_curves(curves_for_plot, OUTPUT_DIR / "roc_curve_v3.png")

    # Select 1 normal image and 3 diverse stain images for preview
    stain_indices = np.where(labels == 1)[0]
    sample_indices = [0, int(stain_indices[0]), int(stain_indices[len(stain_indices) // 2]), int(stain_indices[-1])]
    thumb_w, thumb_h = 224, 224
    sheet = Image.new("RGB", (thumb_w * 4, thumb_h * len(sample_indices)), "black")
    draw_sheet = ImageDraw.Draw(sheet)

    for row_idx, idx in enumerate(sample_indices):
        # Panel 1: raw residual map
        r_raw = (raw_maps[idx] / (np.max(raw_maps[idx]) + 1e-6) * 255).clip(0, 255).astype(np.uint8)
        im_raw = Image.fromarray(r_raw).convert("RGB")

        # Panel 2: V2 normalized map
        v2_norm = (v2_maps[idx] - v2_maps[idx].min()) / (v2_maps[idx].max() - v2_maps[idx].min() + 1e-6)
        im_v2 = Image.fromarray((v2_norm * 255).astype(np.uint8)).convert("RGB")

        # Panel 3: V3 detrended map
        v3_norm = (v3_maps[idx] - v3_maps[idx].min()) / (v3_maps[idx].max() - v3_maps[idx].min() + 1e-6)
        im_v3 = Image.fromarray((v3_norm * 255).astype(np.uint8)).convert("RGB")

        # Panel 4: V3 clean connected component mask (spurious parts removed!)
        clean_mask = (v3_clean_masks[idx].astype(np.uint8) * 255)
        im_clean = Image.fromarray(clean_mask).convert("RGB")

        sheet.paste(im_raw, (0, row_idx * thumb_h))
        sheet.paste(im_v2, (thumb_w, row_idx * thumb_h))
        sheet.paste(im_v3, (thumb_w * 2, row_idx * thumb_h))
        sheet.paste(im_clean, (thumb_w * 3, row_idx * thumb_h))

        kind = "Good" if labels[idx] == 0 else "Stain"
        draw_sheet.text((10, row_idx * thumb_h + 10), f"{kind} Raw Res", fill="white")
        draw_sheet.text((thumb_w + 10, row_idx * thumb_h + 10), "V2 Map (Has Spurious)", fill="white")
        draw_sheet.text((thumb_w * 2 + 10, row_idx * thumb_h + 10), "V3 Detrended", fill="white")
        draw_sheet.text((thumb_w * 3 + 10, row_idx * thumb_h + 10), "V3 Clean Mask (Area>=10)", fill="white")

    sheet.save(OUTPUT_DIR / "v3_clean_preview.png")

    report = {
        "status": "passed",
        "description": "V3 Fabric Stain Detrending and Area Filtering Analysis",
        "baseline_v1_auroc": results[0]["auroc"],
        "v2_auroc": results[1]["auroc"],
        "v3_detrended_auroc": results[2]["auroc"],
        "v3_clean_area_auroc": results[3]["auroc"],
        "strategies": results,
    }
    with (OUTPUT_DIR / "offline_evaluation_report_v3.json").open("w") as f:
        json.dump(report, f, indent=2)

    # Save cleaned maps
    np.savez_compressed(OUTPUT_DIR / "test_maps_v3_float16.npz",
                        v3_maps=v3_maps.astype(np.float16),
                        clean_masks=np.array(v3_clean_masks, dtype=bool),
                        scores=v3_area_scores,
                        labels=labels)

    print("\n=== V1 vs V2 vs V3 Evaluation Results ===")
    print(f"{'Strategy':<40} | {'AUROC':<7} | {'Sens @100% Spec':<16}")
    print("-" * 68)
    for r in results:
        print(f"{r['strategy']:<40} | {r['auroc']:<7.3f} | {r['sensitivity_spec100']*100:<15.1f}%")
    print(f"\nArtifacts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
