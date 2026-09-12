"""Export all 110 fabric test images for V5 evaluation gallery."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V5_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v5"
NPZ_PATH = V5_DIR / "test_maps_v5_float16.npz"
CSV_PATH = V5_DIR / "per_image_v5.csv"
DATA_ZIP_PATH = ROOT / "output" / "datasets" / "fabric_stain_pilot.zip"
GALLERY_DIR = V5_DIR / "gallery"
IMAGES_DIR = GALLERY_DIR / "images"


def colorize_heatmap(residual_map: np.ndarray, vmin: float = 0.0, vmax: float = 30.0) -> Image.Image:
    norm = np.clip((residual_map - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    r = np.clip(norm * 2.2, 0, 1)
    g = np.clip(norm * 1.4 - 0.2, 0, 1)
    b = np.clip(norm * 3.0 - 1.2, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def analyze_defect_clusters(v5_map: np.ndarray, tau_high: float = 15.0, tau_low: float = 3.3,
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


def run():
    if not NPZ_PATH.is_file():
        print(f"NPZ not found at {NPZ_PATH}; run after V5 Kaggle results are extracted.")
        return
    if not DATA_ZIP_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_ZIP_PATH}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(NPZ_PATH)
    v5_maps = data["v5_maps"].astype(np.float32)
    scores = data["image_scores"].astype(np.float32)
    labels = data["labels"].astype(np.int64)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    zip_archive = zipfile.ZipFile(DATA_ZIP_PATH)

    tau_strict = 17.95
    tau_high = 15.0
    tau_low = 3.3
    min_area = 35

    cards_html = []
    stats = {"total": len(rows), "good": 0, "stain": 0, "strict_tp": 0, "stain_tp": 0, "crease_count": 0}

    print(f"Generating V5 gallery for {len(rows)} test images...")

    for idx, r in enumerate(rows):
        img_rel = r["image"]
        kind = r["kind"]
        score = float(r["v5_score"])
        is_stain = labels[idx] == 1

        if is_stain:
            stats["stain"] += 1
        else:
            stats["good"] += 1

        zip_member = f"fabric_stain_pilot/{img_rel}"
        with zip_archive.open(zip_member) as img_stream:
            orig_im = Image.open(io.BytesIO(img_stream.read())).convert("RGB").resize((224, 224), Image.BILINEAR)

        v5_map = v5_maps[idx]
        heatmap_im = colorize_heatmap(v5_map, vmin=0.0, vmax=max(30.0, score * 0.8))

        mask_strict = v5_map > tau_strict
        has_s, has_c, m_s, m_c, m_tot = analyze_defect_clusters(
            v5_map, tau_high=tau_high, tau_low=tau_low, min_area=min_area, max_aspect_ratio=8.0
        )

        mask_strict_im = Image.fromarray((mask_strict.astype(np.uint8) * 255)).convert("RGB")
        mask_stain_im = Image.fromarray((m_s.astype(np.uint8) * 255)).convert("RGB")
        mask_total_im = Image.fromarray((m_tot.astype(np.uint8) * 255)).convert("RGB")

        base_name = f"sample_{idx:03d}"
        orig_file = f"{base_name}_orig.jpg"
        heat_file = f"{base_name}_heat.jpg"
        mask_s_file = f"{base_name}_mask_strict.png"
        mask_st_file = f"{base_name}_mask_stain.png"
        mask_tot_file = f"{base_name}_mask_total.png"

        orig_im.save(IMAGES_DIR / orig_file, quality=90)
        heatmap_im.save(IMAGES_DIR / heat_file, quality=90)
        mask_strict_im.save(IMAGES_DIR / mask_s_file)
        mask_stain_im.save(IMAGES_DIR / mask_st_file)
        mask_total_im.save(IMAGES_DIR / mask_tot_file)

        strict_detected = bool(np.any(mask_strict))
        area = int(np.sum(m_s))

        if is_stain and strict_detected:
            stats["strict_tp"] += 1
        if is_stain and has_s:
            stats["stain_tp"] += 1
        if has_c:
            stats["crease_count"] += 1

        category_tag = "normal" if not is_stain else ("strict" if strict_detected else ("recovered" if has_s else "missed"))
        if has_c and not has_s:
            category_tag = "crease"

        status_badge = (
            "<span class='badge badge-normal'>Normal (Clean)</span>"
            if not is_stain and not has_s and not has_c
            else (
                "<span class='badge badge-crease'>Physical Crease/Fold</span>"
                if has_c and not has_s
                else (
                    "<span class='badge badge-detected'>Strict & Full Body</span>"
                    if strict_detected
                    else (
                        "<span class='badge badge-recovered'>Faint Stain Recovered</span>"
                        if has_s
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
                    <img src="images/{heat_file}" alt="V5 Multi-Scale Heatmap">
                    <div class="label">V5 Heatmap</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_s_file}" alt="Strict Mask">
                    <div class="label">Strict Mask (&tau;=17.9)</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_st_file}" alt="Chemical Stain Mask">
                    <div class="label">Stain Mask (Area={area}px)</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_tot_file}" alt="Total Defect Mask">
                    <div class="label">Total Defects</div>
                </div>
            </div>
            <div class="card-footer">
                <span>Score: <strong>{score:.2f}</strong></span>
                <span>Type: <strong>{kind.upper()}</strong></span>
            </div>
        </div>
        """
        )

    recovered_count = stats["stain_tp"] - stats["strict_tp"]
    missed_count = stats["stain"] - stats["stain_tp"]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fabric Stain V5 Complete Outputs Gallery</title>
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
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(540px, 1fr)); gap: 16px; }}
        .card {{ background: #1E293B; border-radius: 8px; overflow: hidden; border: 1px solid #334155; display: flex; flex-direction: column; }}
        .card-header {{ padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; background: #182234; border-bottom: 1px solid #334155; }}
        .card-title {{ font-size: 13px; font-weight: 600; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
        .badge-normal {{ background: #065F46; color: #6EE7B7; }}
        .badge-detected {{ background: #1E3A8A; color: #93C5FD; }}
        .badge-recovered {{ background: #78350F; color: #FDE68A; }}
        .badge-crease {{ background: #4C1D95; color: #DDD6FE; }}
        .badge-missed {{ background: #450A0A; color: #FCA5A5; }}
        .card-body {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; padding: 10px; }}
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
    <h1>Fabric Stain Inspection &mdash; Complete V5 Output Gallery (110 Test Images)</h1>
    <div class="stats-bar">
        <div class="stat-item"><span class="stat-value">{stats['total']}</span><span class="stat-label">Total Test Images</span></div>
        <div class="stat-item"><span class="stat-value">{stats['good']}</span><span class="stat-label">Normal Controls</span></div>
        <div class="stat-item"><span class="stat-value">{stats['stain']}</span><span class="stat-label">Stained Samples</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #60A5FA;">{stats['strict_tp']}/100</span><span class="stat-label">Strict Detections (&tau;=17.9)</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #FBBF24;">{stats['stain_tp']}/100</span><span class="stat-label">Chemical Stains Recovered</span></div>
    </div>

    <div class="filter-bar">
        <button class="active" onclick="filterCards('all')">Show All (110)</button>
        <button onclick="filterCards('normal')">Normal Images (10)</button>
        <button onclick="filterCards('strict')">Strict Detections ({stats['strict_tp']})</button>
        <button onclick="filterCards('recovered')">Faint Stains Recovered ({recovered_count})</button>
        <button onclick="filterCards('crease')">Physical Creases ({stats['crease_count']})</button>
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
    print(f"\nV5 Gallery successfully created at:\n{index_html_path}")


if __name__ == "__main__":
    run()
