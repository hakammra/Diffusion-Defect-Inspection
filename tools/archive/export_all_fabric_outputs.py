from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP_PATH = ROOT / "output" / "datasets" / "fabric_stain_pilot.zip"
NPZ_PATH = ROOT / "artifacts" / "fabric_stain_evaluation_v3" / "test_maps_v3_float16.npz"
CSV_PATH = ROOT / "artifacts" / "fabric_stain_evaluation_v3" / "per_image_v3.csv"
GALLERY_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v3" / "gallery"
IMAGES_DIR = GALLERY_DIR / "images"


def colorize_heatmap(array: np.ndarray, vmin: float = 0.0, vmax: float = 40.0) -> Image.Image:
    norm = np.clip((array - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    # Magma-like gradient: black -> purple -> red -> orange -> yellow -> white
    r = np.clip(norm * 2.5, 0, 1)
    g = np.clip((norm - 0.3) * 2.0, 0, 1)
    b = np.clip(np.where(norm < 0.5, norm * 2.0, (1.0 - norm) * 1.5 + 0.5), 0, 1)

    rgb = np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def remove_small_components(mask: np.ndarray, min_area: int = 10) -> np.ndarray:
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    cleaned = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                comp = []
                q = [(r, c)]
                visited[r, c] = True
                while q:
                    cr, cc = q.pop()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                if len(comp) >= min_area:
                    for cr, cc in comp:
                        cleaned[cr, cc] = True
    return cleaned


def run():
    if not NPZ_PATH.is_file():
        raise FileNotFoundError(f"Missing {NPZ_PATH}")
    if not DATA_ZIP_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_ZIP_PATH}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(NPZ_PATH)
    v3_maps = data["v3_maps"].astype(np.float32)
    scores = (data["image_scores"] if "image_scores" in data else data["scores"]).astype(np.float32)
    labels = data["labels"].astype(np.int64)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    zip_archive = zipfile.ZipFile(DATA_ZIP_PATH)

    tau_strict = 25.65  # Calibrated on max validation normal
    tau_high = 14.0     # Confident seed threshold
    tau_low = 4.5       # Stain body threshold (captures full stain shape)

    cards_html = []
    stats = {"total": len(rows), "good": 0, "stain": 0, "strict_tp": 0, "hysteresis_tp": 0}

    def hysteresis_mask_with_margin(v3_map, th_high=tau_high, th_low=tau_low, min_area=15, margin=14):
        m = v3_map.copy()
        m[:margin, :] = 0
        m[-margin:, :] = 0
        m[:, :margin] = 0
        m[:, -margin:] = 0
        
        seed_mask = m > th_high
        body_mask = m > th_low
        if not np.any(seed_mask):
            return np.zeros_like(seed_mask)
        
        H, W = m.shape
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
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and body_mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    if has_seed and len(cluster) >= min_area:
                        for cr, cc in cluster:
                            final_mask[cr, cc] = True
        return final_mask

    print(f"Generating gallery for {len(rows)} test images...")

    for idx, r in enumerate(rows):
        img_rel = r["image"]  # e.g. test/good/19.jpg
        kind = r["kind"]
        score = float(r["v3_score"])
        is_stain = labels[idx] == 1

        if is_stain:
            stats["stain"] += 1
        else:
            stats["good"] += 1

        zip_member = f"fabric_stain_pilot/{img_rel}"
        with zip_archive.open(zip_member) as img_stream:
            orig_im = Image.open(io.BytesIO(img_stream.read())).convert("RGB").resize((224, 224), Image.BILINEAR)

        v3_map = v3_maps[idx]
        heatmap_im = colorize_heatmap(v3_map, vmin=0.0, vmax=max(35.0, score * 0.8))

        # 1. Strict peak mask
        bin_strict = v3_map > tau_strict
        mask_strict = remove_small_components(bin_strict, min_area=10)

        # 2. Hysteresis full-body mask
        mask_hyst = hysteresis_mask_with_margin(v3_map, tau_high, tau_low, min_area=15, margin=14)

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

        strict_detected = np.any(mask_strict)
        hyst_detected = np.any(mask_hyst)

        if is_stain and strict_detected:
            stats["strict_tp"] += 1
        if is_stain and hyst_detected:
            stats["hysteresis_tp"] += 1

        category_tag = "normal" if not is_stain else ("strict" if strict_detected else ("recovered" if hyst_detected else "missed"))

        status_badge = (
            "<span class='badge badge-normal'>Normal (Clean)</span>"
            if not is_stain and not hyst_detected
            else (
                "<span class='badge badge-missed'>Normal (False Alarm)</span>"
                if not is_stain and hyst_detected
                else (
                    "<span class='badge badge-detected'>Strict & Full Body</span>"
                    if strict_detected
                    else (
                        "<span class='badge badge-recovered'>Full Body Recovered</span>"
                        if hyst_detected
                        else "<span class='badge badge-missed'>Faint / Missed</span>"
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
                    <img src="images/{heat_file}" alt="V3 Heatmap">
                    <div class="label">V3 Heatmap</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_s_file}" alt="Strict Mask">
                    <div class="label">Strict Mask (&tau;=25.6)</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_h_file}" alt="Full Body Mask">
                    <div class="label">Full Body (Hysteresis)</div>
                </div>
            </div>
            <div class="card-footer">
                <span>Score: <strong>{score:.2f}</strong></span>
                <span>Type: <strong>{kind.upper()}</strong></span>
            </div>
        </div>
        """
        )

    recovered_count = stats['hysteresis_tp'] - stats['strict_tp']
    missed_count = stats['stain'] - stats['hysteresis_tp']

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fabric Stain V3 Full Outputs Gallery</title>
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
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 16px; }}
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
    <h1>Fabric Stain Inspection &mdash; Complete Output Gallery (110 Test Images)</h1>
    <div class="stats-bar">
        <div class="stat-item"><span class="stat-value">{stats['total']}</span><span class="stat-label">Total Test Images</span></div>
        <div class="stat-item"><span class="stat-value">{stats['good']}</span><span class="stat-label">Normal Controls</span></div>
        <div class="stat-item"><span class="stat-value">{stats['stain']}</span><span class="stat-label">Stained Samples</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #60A5FA;">{stats['strict_tp']}/100</span><span class="stat-label">Detected at Strict (&tau;=25.6)</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #FBBF24;">{stats['hysteresis_tp']}/100</span><span class="stat-label">Full Body (Hysteresis)</span></div>
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
    print(f"\nGallery successfully created at:\n{index_html_path}")


if __name__ == "__main__":
    run()
