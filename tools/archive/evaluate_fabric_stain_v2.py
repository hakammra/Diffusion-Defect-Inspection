from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
INPUT_NPZ = ROOT / "artifacts" / "fabric_stain_evaluation" / "test_maps_float16.npz"
OUTPUT_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v2"


def compute_roc(y_true: np.ndarray, y_score: np.ndarray):
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    auroc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auroc


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray):
    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]
    cum_tp = np.cumsum(y_true_sorted)
    precisions = cum_tp / (np.arange(len(y_true_sorted)) + 1)
    pos_indices = np.where(y_true_sorted == 1)[0]
    if len(pos_indices) == 0:
        return 0.0
    return float(np.mean(precisions[pos_indices]))


def box_filter_2d(images: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    padded = np.pad(images, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    windows = sliding_window_view(padded, (k, k), axis=(1, 2))
    return np.mean(windows, axis=(-2, -1))


def morph_open_2d(images: np.ndarray, k: int = 3) -> np.ndarray:
    pad = k // 2
    padded = np.pad(images, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    windows = sliding_window_view(padded, (k, k), axis=(1, 2))
    eroded = np.min(windows, axis=(-2, -1))

    padded_eroded = np.pad(eroded, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    windows_dilated = sliding_window_view(padded_eroded, (k, k), axis=(1, 2))
    return np.max(windows_dilated, axis=(-2, -1))


def iqr_normalize(maps: np.ndarray) -> np.ndarray:
    flat = maps.reshape(maps.shape[0], -1)
    medians = np.median(flat, axis=1, keepdims=True)
    q75 = np.quantile(flat, 0.75, axis=1, keepdims=True)
    q25 = np.quantile(flat, 0.25, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-6)
    normalized = (flat - medians) / iqr
    return normalized.reshape(maps.shape)


def draw_roc_curves(curves_dict: dict[str, tuple[np.ndarray, np.ndarray, float]], out_path: Path):
    width, height = 700, 600
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
    draw.text((margin, 25), "Fabric Stain Detection ROC Curves (V1 vs V2 Strategies)", fill="#111827")

    color_map = {
        "V1 Baseline (Raw p99.5)": "#DC2626",
        "V2 Smooth 5x5 + IQR (p99.9)": "#2563EB",
        "V2 Morph Opening 3x3 + IQR (Max)": "#059669",
        "V2 Raw + IQR (p99.9)": "#D97706",
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

        draw.rectangle([(width - margin - 275, legend_y), (width - margin - 250, legend_y + 12)], fill=color)
        draw.text((width - margin - 240, legend_y), f"{name}: {auroc:.3f}", fill="#1F2937")
        legend_y += 24

    img.save(out_path)


def run():
    if not INPUT_NPZ.is_file():
        raise FileNotFoundError(f"Missing residual maps file: {INPUT_NPZ}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(INPUT_NPZ)
    raw_scores = data["scores"].astype(np.float32)  # (110, 224, 224)
    labels = data["labels"].astype(np.int64)  # (110,)

    norm_mask = labels == 0
    stain_mask = labels == 1
    num_normals = int(np.sum(norm_mask))
    num_stains = int(np.sum(stain_mask))

    print(f"Loaded test maps for {num_normals} normal and {num_stains} stain images.")

    print("Computing filtered and normalized maps...")
    smooth3 = box_filter_2d(raw_scores, 3)
    smooth5 = box_filter_2d(raw_scores, 5)
    opened3 = morph_open_2d(raw_scores, 3)

    raw_iqr = iqr_normalize(raw_scores)
    smooth5_iqr = iqr_normalize(smooth5)
    opened3_iqr = iqr_normalize(opened3)

    strategies = {
        "V1 Baseline (Raw p99.5)": np.quantile(raw_scores.reshape(110, -1), 0.995, axis=1),
        "V1 Max (Raw Max)": np.max(raw_scores.reshape(110, -1), axis=1),
        "V2 Smooth 3x3 (p99.9)": np.quantile(smooth3.reshape(110, -1), 0.999, axis=1),
        "V2 Smooth 5x5 (p99.9)": np.quantile(smooth5.reshape(110, -1), 0.999, axis=1),
        "V2 Morph Opening 3x3 (Max)": np.max(opened3.reshape(110, -1), axis=1),
        "V2 Raw + IQR (p99.9)": np.quantile(raw_iqr.reshape(110, -1), 0.999, axis=1),
        "V2 Smooth 5x5 + IQR (p99.9)": np.quantile(smooth5_iqr.reshape(110, -1), 0.999, axis=1),
        "V2 Morph Opening 3x3 + IQR (Max)": np.max(opened3_iqr.reshape(110, -1), axis=1),
    }

    results = []
    curves_for_plot = {}

    for name, img_scores in strategies.items():
        fpr, tpr, auroc = compute_roc(labels, img_scores)
        ap = compute_average_precision(labels, img_scores)

        normal_scores = np.sort(img_scores[norm_mask])
        stain_scores = img_scores[stain_mask]

        thresh_90 = float(normal_scores[-2])
        tp_90 = int(np.sum(stain_scores > thresh_90))
        fp_90 = int(np.sum(normal_scores > thresh_90))
        sens_90 = tp_90 / num_stains
        spec_90 = (num_normals - fp_90) / num_normals

        thresh_100 = float(normal_scores[-1])
        tp_100 = int(np.sum(stain_scores > thresh_100))
        fp_100 = int(np.sum(normal_scores > thresh_100))
        sens_100 = tp_100 / num_stains
        spec_100 = 1.0

        record = {
            "strategy": name,
            "auroc": round(auroc, 4),
            "average_precision": round(ap, 4),
            "threshold_spec90": round(thresh_90, 4),
            "sensitivity_spec90": round(sens_90, 4),
            "specificity_spec90": round(spec_90, 4),
            "tp_spec90": tp_90,
            "fp_spec90": fp_90,
            "threshold_spec100": round(thresh_100, 4),
            "sensitivity_spec100": round(sens_100, 4),
            "specificity_spec100": round(spec_100, 4),
            "tp_spec100": tp_100,
            "fp_spec100": fp_100,
        }
        results.append(record)

        if name in [
            "V1 Baseline (Raw p99.5)",
            "V2 Smooth 5x5 + IQR (p99.9)",
            "V2 Morph Opening 3x3 + IQR (Max)",
            "V2 Raw + IQR (p99.9)",
        ]:
            curves_for_plot[name] = (fpr, tpr, auroc)

    table_csv = OUTPUT_DIR / "comparison_table.csv"
    with table_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best_scores = strategies["V2 Smooth 5x5 + IQR (p99.9)"]
    best_thresh = results[6]["threshold_spec90"]
    per_image_rows = []
    v1_per_image = ROOT / "artifacts" / "fabric_stain_evaluation" / "per_image.csv"
    orig_rows = []
    if v1_per_image.is_file():
        with v1_per_image.open("r") as f:
            orig_rows = list(csv.DictReader(f))

    for idx in range(len(labels)):
        img_name = orig_rows[idx]["image"] if idx < len(orig_rows) else f"image_{idx}"
        kind = "stain" if labels[idx] == 1 else "good"
        score = float(best_scores[idx])
        pred = int(score > best_thresh)
        per_image_rows.append(
            {
                "image": img_name,
                "label": int(labels[idx]),
                "kind": kind,
                "v1_score_p995": float(orig_rows[idx]["image_score_p995"]) if orig_rows else None,
                "v2_smooth_iqr_score": round(score, 4),
                "v2_prediction_spec90": pred,
            }
        )

    with (OUTPUT_DIR / "per_image_v2.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    draw_roc_curves(curves_for_plot, OUTPUT_DIR / "roc_curve_comparison.png")

    report = {
        "status": "passed",
        "description": "Offline V2 Thresholding and Post-Processing Evaluation on Fabric Stain Maps",
        "baseline_v1_auroc": results[0]["auroc"],
        "best_v2_auroc": max(r["auroc"] for r in results),
        "best_v2_strategy": max(results, key=lambda x: x["auroc"])["strategy"],
        "strategies": results,
    }
    with (OUTPUT_DIR / "offline_evaluation_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print("\n=== V1 vs V2 Evaluation Results ===")
    print(f"{'Strategy':<36} | {'AUROC':<7} | {'Sens @90%':<10} | {'Sens @100%':<11}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['strategy']:<36} | {r['auroc']:<7.3f} | {r['sensitivity_spec90']*100:<9.1f}% | {r['sensitivity_spec100']*100:<10.1f}%"
        )
    print("\nSaved artifacts in:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
