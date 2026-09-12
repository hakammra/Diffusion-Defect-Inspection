"""Export all 110 fabric test images for V4 evaluation gallery."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V4_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v4"
NPZ_PATH = V4_DIR / "test_maps_v4_float16.npz"
CSV_PATH = V4_DIR / "per_image_v4.csv"
DATA_ZIP_PATH = ROOT / "output" / "datasets" / "fabric_stain_pilot.zip"
GALLERY_DIR = V4_DIR / "gallery"
IMAGES_DIR = GALLERY_DIR / "images"


def colorize_heatmap(residual_map: np.ndarray, vmin: float = 0.0, vmax: float = 35.0) -> Image.Image:
    norm = np.clip((residual_map - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    r = np.clip(norm * 2.2, 0, 1)
    g = np.clip(norm * 1.4 - 0.2, 0, 1)
    b = np.clip(norm * 3.0 - 1.2, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def hysteresis_segmentation(v4_map: np.ndarray, tau_high: float, tau_low: float, min_area: int = 40) -> np.ndarray:
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


def run():
    if not NPZ_PATH.is_file():
        raise FileNotFoundError(f"Missing {NPZ_PATH}")
    if not DATA_ZIP_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_ZIP_PATH}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(NPZ_PATH)
    v4_maps = data["v4_maps"].astype(np.float32)
    scores = data["image_scores"].astype(np.float32)
    labels = data["labels"].astype(np.int64)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    zip_archive = zipfile.ZipFile(DATA_ZIP_PATH)

    tau_strict = 17.95
    tau_high = 16.0
    tau_low = 4.0
    min_area = 40

    cards_html = []
    stats = {"total": len(rows), "good": 0, "stain": 0, "strict_tp": 0, "hysteresis_tp": 0}

    print(f"Generating V4 gallery for {len(rows)} test images...")

    for idx, r in enumerate(rows):
        img_rel = r["image"]
        kind = r["kind"]
        score = float(r["v4_score"])
        is_stain = labels[idx] == 1

        if is_stain:
            stats["stain"] += 1
        else:
            stats["good"] += 1

        zip_member = f"fabric_stain_pilot/{img_rel}"
        with zip_archive.open(zip_member) as img_stream:
            orig_im = Image.open(io.BytesIO(img_stream.read())).convert("RGB").resize((224, 224), Image.BILINEAR)

        v4_map = v4_maps[idx]
        heatmap_im = colorize_heatmap(v4_map, vmin=0.0, vmax=max(30.0, score * 0.8))

        # 1. Strict peak mask
        mask_strict = v4_map > tau_strict

        # 2. Hysteresis full-body mask
        mask_hyst = hysteresis_segmentation(v4_map, tau_high, tau_low, min_area=min_area)

        mask_strict_im = Image.fromarray((mask_strict.astype(np.uint8) * 255)).convert("RGB")
        mask_hyst_im = Image.fromarray((mask_hyst.astype(np.uint8) * 255)).convert("RGB")

        base_name = f"sample_{idx:03d}"
        orig_file = f"{base_name}_orig.jpg"
        heat_file = f"{base_name}_heat.jpg"
        mask_s_file = f"{base_name}_mask_strict.png"
        mask_h_file = f"{base_name}_mask_hyst.png"

        orig_im.save(IMAGES_DIR / orig_file, quality=90)
        heatmap_im.save(IMAGES_DIR / heat_file, quality=90)
        mask_strict_im.save(IMAGES_DIR / mask_s_file)
        mask_hyst_im.save(IMAGES_DIR / mask_h_file)

        strict_detected = bool(np.any(mask_strict))
        hyst_detected = bool(np.any(mask_hyst))
        area = int(np.sum(mask_hyst))

        if is_stain and strict_detected:
            stats["strict_tp"] += 1
        if is_stain and hyst_detected:
            stats["hysteresis_tp"] += 1

        category_tag = "normal" if not is_stain else ("strict" if strict_detected else ("recovered" if hyst_detected else "missed"))

        status_badge = (
            "<span class='badge badge-normal'>Normal (Clean)</span>"
            if not is_stain and not hyst_detected
            else (
                "<span class='badge badge-missed'>Normal (Fold Crease)</span>"
                if not is_stain and hyst_detected
                else (
                    "<span class='badge badge-detected'>Strict & Full Body</span>"
                    if strict_detected
                    else (
                        "<span class='badge badge-recovered'>Faint Recovered</span>"
                        if hyst_detected
                        else "<span class='badge badge-missed'>Subtle / Missed</span>"
                    )
                )
            )
        )

        cards_html.append(
            f"""
        <div class="card" data-category="{category_tag}">
            <div class="card-header">
                <span class="card-title">#{idx:03d}: {Path(img_rel).name}</span>
                {status_badge}
            </div>
            <div class="card-body">
                <div class="panel">
                    <img src="images/{orig_file}" alt="Original">
                    <div class="label">Original</div>
                </div>
                <div class="panel">
                    <img src="images/{heat_file}" alt="V4 Heatmap">
                    <div class="label">V4 Reflect Heatmap</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_s_file}" alt="Strict Mask">
                    <div class="label">Strict Mask (&tau;=17.9)</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_h_file}" alt="Full Body Mask">
                    <div class="label">Full Body (Area={area}px)</div>
                </div>
            </div>
            <div class="card-footer">
                <span>Score: <strong>{score:.2f}</strong></span>
                <span>Type: <strong>{kind.upper()}</strong></span>
            </div>
        </div>
        """
        )

    recovered_count = stats["hysteresis_tp"] - stats["strict_tp"]
    missed_count = stats["stain"] - stats["hysteresis_tp"]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fabric Stain V4 Complete Outputs Gallery</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0F172A; color: #F8FAFC; margin: 0; padding: 20px; }}
        h1 {{ margin-bottom: 8px; font-size: 24px; }}
        .stats-bar {{ display: flex; gap: 20px; background: #1E293B; padding: 16px; border-radius: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
        .stat-item {{ display: flex; flex-direction: column; }}
        .stat-value {{ font-size: 22px; font-weight: bold; color: #38BDF8; }}
        .stat-label {{ font-size: 12px; color: #94A3B8; }}
        .filter-bar {{ display: flex; gap: 10px; margin-bottom: 24px; flex-wrap: wrap; }}
        button {{ background: #334155; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s; }}
        button:hover, button.active {{ background: #2563EB; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 16px; }}
        .card {{ background: #1E293B; border-radius: 8px; overflow: hidden; border: 1px solid #334155; display: flex; flex-direction: column; }}
        .card-header {{ padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; background: #182234; border-bottom: 1px solid #334155; }}
        .card-title {{ font-size: 13px; font-weight: 600; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
        .badge-normal {{ background: #065F46; color: #6EE7B7; }}
        .badge-detected {{ background: #1E3A8A; color: #93C5FD; }}
        .badge-recovered {{ background: #78350F; color: #FDE68A; }}
        .badge-missed {{ background: #450A0A; color: #FCA5A5; }}
        .card-body {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; padding: 10px; }}
        .panel {{ text-align: center; }}
        .panel img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; background: #000; }}
        .panel .label {{ font-size: 10px; color: #94A3B8; margin-top: 4px; }}
        .card-footer {{ padding: 8px 14px; background: #182234; border-top: 1px solid #334155; display: flex; justify-content: space-between; font-size: 12px; color: #CBD5E1; }}
    </style>
    <script>
        function filterCards(category) {{
            document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            const cards = document.querySelectorAll('.card');
            cards.forEach(card => {{
                if (category === 'all' || card.dataset.category === category) {{
                    card.style.display = 'flex';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</head>
<body>
    <h1>Fabric Stain Inspection &mdash; Complete V4 Output Gallery (110 Test Images)</h1>
    <div class="stats-bar">
        <div class="stat-item"><span class="stat-value">{stats['total']}</span><span class="stat-label">Total Test Images</span></div>
        <div class="stat-item"><span class="stat-value">{stats['good']}</span><span class="stat-label">Normal Controls</span></div>
        <div class="stat-item"><span class="stat-value">{stats['stain']}</span><span class="stat-label">Stained Samples</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #60A5FA;">{stats['strict_tp']}/100</span><span class="stat-label">Strict Detections (&tau;=17.9)</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #FBBF24;">{stats['hysteresis_tp']}/100</span><span class="stat-label">Hysteresis Recovered</span></div>
    </div>

    <div class="filter-bar">
        <button class="active" onclick="filterCards('all')">Show All (110)</button>
        <button onclick="filterCards('normal')">Normal Images (10)</button>
        <button onclick="filterCards('strict')">Strict Detections ({stats['strict_tp']})</button>
        <button onclick="filterCards('recovered')">Faint Stains Recovered ({recovered_count})</button>
        <button onclick="filterCards('missed')">Faint / Missed Stains ({missed_count})</button>
    </div>

    <div class="grid">
        {''.join(cards_html)}
    </div>
</body>
</html>
"""

    index_html_path = GALLERY_DIR / "index.html"
    index_html_path.write_text(html_content, encoding="utf-8")
    print(f"\nV4 Gallery successfully created at:\n{index_html_path}")


if __name__ == "__main__":
    run()
