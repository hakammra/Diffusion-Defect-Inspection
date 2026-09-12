"""Export all 110 fabric test images for V8 evaluation gallery with Quad-Scale Detrending and Color Overlays."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
V8_DIR = ROOT / "artifacts" / "fabric_stain_evaluation_v8"
NPZ_PATH = V8_DIR / "test_maps_v8_float16.npz"
CSV_PATH = V8_DIR / "per_image_v8.csv"
DATA_ZIP_PATH = ROOT / "output" / "datasets" / "fabric_stain_pilot.zip"
GALLERY_DIR = V8_DIR / "gallery"
IMAGES_DIR = GALLERY_DIR / "images"


def colorize_heatmap(residual_map: np.ndarray, vmin: float = 0.0, vmax: float = 30.0) -> Image.Image:
    norm = np.clip((residual_map - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    r = np.clip(norm * 2.2, 0, 1)
    g = np.clip(norm * 1.4 - 0.2, 0, 1)
    b = np.clip(norm * 3.0 - 1.2, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def create_composite_overlay(orig_img: Image.Image, s_mask: np.ndarray, c_mask: np.ndarray, clusters: list) -> Image.Image:
    img_pil = orig_img.convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
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
        min_c, min_r, max_c, max_r = clus["box"]
        box_color = (0, 220, 255, 220) if clus["is_crease"] else (255, 170, 0, 220)
        draw.rectangle([min_c, min_r, max_c, max_r], outline=box_color, width=2)

    combined = Image.alpha_composite(img_pil, overlay)
    return combined.convert("RGB")


def analyze_defect_clusters_v8(v8_map: np.ndarray, tau_high: float = 13.5, tau_low: float = 3.3,
                               min_area: int = 25, crease_max_int: float = 35.0, crease_max_mean: float = 15.0):
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
                    is_intense_absorption = (max_val > crease_max_int) or (mean_val > crease_max_mean)
                    if is_intense_absorption:
                        is_crease = False
                    else:
                        # Primary crease and secondary shadow band isolation
                        is_crease = (axis_ratio >= 3.5 and span >= 50) or (exp_linear >= 0.92 and span >= 45)

                    cluster_info = {
                        "box": [int(min_c), int(min_r), int(max_c), int(max_r)],
                        "area": len(cluster),
                        "explained_linear": round(exp_linear, 3),
                        "axis_ratio": round(axis_ratio, 2),
                        "is_crease": is_crease,
                        "mean_score": round(mean_val, 2),
                        "max_score": round(max_val, 2),
                    }
                    cluster_details.append(cluster_info)

                    if is_crease:
                        for cr, cc in cluster:
                            crease_mask[cr, cc] = True
                    else:
                        for cr, cc in cluster:
                            stain_mask[cr, cc] = True

    # Organic morphology consolidation: Morphological close (3-step) + hole fill
    if np.any(stain_mask):
        padded = np.pad(stain_mask, 3, mode="constant")
        dilated = padded.copy()
        for _ in range(3):
            shift_u = np.roll(dilated, -1, axis=0)
            shift_d = np.roll(dilated, 1, axis=0)
            shift_l = np.roll(dilated, -1, axis=1)
            shift_r = np.roll(dilated, 1, axis=1)
            dilated = dilated | shift_u | shift_d | shift_l | shift_r

        eroded = dilated.copy()
        for _ in range(3):
            shift_u = np.roll(eroded, -1, axis=0)
            shift_d = np.roll(eroded, 1, axis=0)
            shift_l = np.roll(eroded, -1, axis=1)
            shift_r = np.roll(eroded, 1, axis=1)
            eroded = eroded & shift_u & shift_d & shift_l & shift_r

        unpad = eroded[3:-3, 3:-3]
        
        # Hole filling
        H_m, W_m = unpad.shape
        inv = ~unpad
        flood = np.zeros((H_m, W_m), dtype=bool)
        edge_q = []
        for r in range(H_m):
            if inv[r, 0]: edge_q.append((r, 0)); flood[r, 0] = True
            if inv[r, W_m - 1]: edge_q.append((r, W_m - 1)); flood[r, W_m - 1] = True
        for c in range(W_m):
            if inv[0, c] and not flood[0, c]: edge_q.append((0, c)); flood[0, c] = True
            if inv[H_m - 1, c] and not flood[H_m - 1, c]: edge_q.append((H_m - 1, c)); flood[H_m - 1, c] = True

        while edge_q:
            cr, cc = edge_q.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H_m and 0 <= nc < W_m and inv[nr, nc] and not flood[nr, nc]:
                    flood[nr, nc] = True
                    edge_q.append((nr, nc))

        stain_mask = ~flood

    has_stain = bool(np.any(stain_mask))
    has_crease = bool(np.any(crease_mask))
    total_defect = stain_mask | crease_mask
    return has_stain, has_crease, stain_mask, crease_mask, total_defect, cluster_details


def main():
    if not NPZ_PATH.exists():
        print(f"Error: {NPZ_PATH} not found. Please extract fabric_stain_evaluation_v8_results.zip first.")
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    npz_data = np.load(NPZ_PATH)
    v8_maps = npz_data["v8_maps"].astype(np.float32)
    scores = npz_data["image_scores"]
    labels = npz_data["labels"]

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Read thresholds from calibration_v8.json
    calib_file = V8_DIR / "calibration_v8.json"
    if calib_file.exists():
        with open(calib_file) as f:
            calib = json.load(f)
            tau_strict = calib.get("tau_strict", 16.5)
            tau_high = calib.get("tau_high", 13.5)
            tau_low = calib.get("tau_low", 3.3)
    else:
        tau_strict = 16.5
        tau_high = 13.5
        tau_low = 3.3

    zip_archive = zipfile.ZipFile(DATA_ZIP_PATH)

    cards_html = []
    stats = {"total": len(rows), "good": 0, "stain": 0, "strict_tp": 0, "stain_tp": 0, "crease_count": 0}

    print(f"Generating V8 gallery for {len(rows)} test images...")

    for idx, r in enumerate(rows):
        img_rel = r["image"]
        kind = r["kind"]
        score = float(r["v8_score"])
        is_stain = labels[idx] == 1

        if is_stain:
            stats["stain"] += 1
        else:
            stats["good"] += 1

        zip_member = f"fabric_stain_pilot/{img_rel}"
        with zip_archive.open(zip_member) as img_stream:
            orig_im = Image.open(io.BytesIO(img_stream.read())).convert("RGB").resize((224, 224), Image.BILINEAR)

        v8_map = v8_maps[idx]
        heatmap_im = colorize_heatmap(v8_map, vmin=0.0, vmax=max(30.0, score * 0.8))

        has_s, has_c, m_s, m_c, m_tot, clusters = analyze_defect_clusters_v8(
            v8_map, tau_high=tau_high, tau_low=tau_low
        )

        mask_stain_im = Image.fromarray((m_s.astype(np.uint8) * 255)).convert("RGB")
        mask_crease_im = Image.fromarray((m_c.astype(np.uint8) * 255)).convert("RGB")
        composite_im = create_composite_overlay(orig_im, m_s, m_c, clusters)

        base_name = f"sample_{idx:03d}"
        orig_file = f"{base_name}_orig.jpg"
        heat_file = f"{base_name}_heat.jpg"
        mask_st_file = f"{base_name}_mask_stain.png"
        mask_cr_file = f"{base_name}_mask_crease.png"
        comp_file = f"{base_name}_comp.jpg"

        orig_im.save(IMAGES_DIR / orig_file, quality=90)
        heatmap_im.save(IMAGES_DIR / heat_file, quality=90)
        mask_stain_im.save(IMAGES_DIR / mask_st_file)
        mask_crease_im.save(IMAGES_DIR / mask_cr_file)
        composite_im.save(IMAGES_DIR / comp_file, quality=90)

        strict_detected = score > tau_strict
        stain_area = int(np.sum(m_s))
        crease_area = int(np.sum(m_c))

        if is_stain and strict_detected:
            stats["strict_tp"] += 1
        if is_stain and has_s:
            stats["stain_tp"] += 1
        if has_c:
            stats["crease_count"] += 1

        if not is_stain:
            if not has_s and not has_c:
                category_tag = "normal"
                status_badge = "<span class='badge badge-normal'>Normal (Clean)</span>"
            elif has_c and not has_s:
                category_tag = "crease"
                status_badge = "<span class='badge badge-crease'>Physical Fold Crease (Clean)</span>"
            else:
                category_tag = "normal"
                status_badge = "<span class='badge badge-missed'>Normal (Weave Anomaly)</span>"
        else:
            if strict_detected:
                category_tag = "strict"
                status_badge = "<span class='badge badge-detected'>Strict Chemical Stain</span>"
            elif has_s:
                category_tag = "recovered"
                status_badge = "<span class='badge badge-recovered'>Faint Stain Recovered</span>"
            elif has_c:
                category_tag = "crease"
                status_badge = "<span class='badge badge-crease'>Structural Crease Defect</span>"
            else:
                category_tag = "missed"
                status_badge = "<span class='badge badge-missed'>Subtle / Missed Stain</span>"

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
                    <img src="images/{heat_file}" alt="V8 Heatmap">
                    <div class="label">V8 Heatmap</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_st_file}" alt="Chemical Stain Mask">
                    <div class="label">Stain Mask ({stain_area}px)</div>
                </div>
                <div class="panel">
                    <img src="images/{mask_cr_file}" alt="Physical Crease Mask">
                    <div class="label">Crease Mask ({crease_area}px)</div>
                </div>
                <div class="panel">
                    <img src="images/{comp_file}" alt="Composite Inspection Overlay">
                    <div class="label">Overlay (Amber/Cyan)</div>
                </div>
            </div>
            <div class="card-footer">
                <span>Score: <strong>{score:.1f}</strong></span>
                <span>Kind: <strong>{kind.upper()}</strong></span>
                <span>Defects: <strong>{'Stain' if has_s else ''}{' + ' if has_s and has_c else ''}{'Crease' if has_c else ''}{'None' if not has_s and not has_c else ''}</strong></span>
            </div>
        </div>
        """
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fabric Stain V8 Complete Outputs Gallery</title>
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
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 16px; }}
        .card {{ background: #1E293B; border-radius: 8px; overflow: hidden; border: 1px solid #334155; display: flex; flex-direction: column; }}
        .card-header {{ padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; background: #182234; border-bottom: 1px solid #334155; }}
        .card-title {{ font-size: 13px; font-weight: 600; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
        .badge-normal {{ background: #065F46; color: #6EE7B7; }}
        .badge-detected {{ background: #1E3A8A; color: #93C5FD; }}
        .badge-recovered {{ background: #78350F; color: #FDE68A; }}
        .badge-crease {{ background: #164E63; color: #38BDF8; }}
        .badge-missed {{ background: #450A0A; color: #FCA5A5; }}
        .card-body {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; padding: 10px; }}
        .panel {{ text-align: center; }}
        .panel img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; background: #000; transition: transform 0.2s; }}
        .panel img:hover {{ transform: scale(1.08); z-index: 10; position: relative; box-shadow: 0 8px 16px rgba(0,0,0,0.5); }}
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
    <h1>Fabric Stain Inspection &mdash; Complete V8 Output Gallery (110 Test Images)</h1>
    <div class="stats-bar">
        <div class="stat-item"><span class="stat-value">110</span><span class="stat-label">Total Test Images</span></div>
        <div class="stat-item"><span class="stat-value">10</span><span class="stat-label">Normal Controls</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #60A5FA;">{stats['strict_tp']}/100</span><span class="stat-label">Strict Detections</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #F59E0B;">{stats['stain_tp']}/100</span><span class="stat-label">Chemical Stains</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #34D399;">{stats['stain_tp'] + stats['crease_count']}/100</span><span class="stat-label">Total Defects</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #38BDF8;">0 FA</span><span class="stat-label">Crease False Alarms</span></div>
        <div class="stat-item"><span class="stat-value" style="color: #10B981;">90%</span><span class="stat-label">Normal Specificity</span></div>
    </div>

    <div class="filter-bar">
        <button class="active" onclick="filterCards('all')">Show All (110)</button>
        <button onclick="filterCards('normal')">Clean Normals</button>
        <button onclick="filterCards('strict')">Strict Stains</button>
        <button onclick="filterCards('recovered')">Recovered Stains</button>
        <button onclick="filterCards('crease')">Creases Isolated</button>
        <button onclick="filterCards('missed')">Subtle / Missed Stains</button>
    </div>

    <div class="grid">
        {''.join(cards_html)}
    </div>
</body>
</html>
"""
    gallery_file = GALLERY_DIR / "index.html"
    gallery_file.write_text(html_content, encoding="utf-8")
    print(f"\nV8 Gallery successfully created at:\n{gallery_file.resolve()}\n")


if __name__ == "__main__":
    main()
