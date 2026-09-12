from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    r"C:\Users\abdul\Downloads\archive\Fabric Defects Dataset\Fabric Defect Dataset"
)
DEFAULT_RESULTS = ROOT / "artifacts" / "fabric_stain_tiled_final_evaluation"
DEFAULT_OUTPUT = ROOT / "results" / "fabric_stain" / "tiled_final" / "gallery"


def colorize(response: np.ndarray, vmax: float = 25.0) -> Image.Image:
    """Small dependency-free purple/orange anomaly heatmap."""
    value = np.clip(response / vmax, 0.0, 1.0)
    red = np.clip(2.2 * value, 0.0, 1.0)
    green = np.clip(1.45 * value - 0.18, 0.0, 1.0)
    blue = np.clip(3.0 * value - 1.15, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def source_path(source: Path, relative: str) -> Path:
    path = Path(relative)
    folder = "defect free" if path.parts[1] == "good" else "stain"
    return source / folder / path.name


def parse_box(value: str) -> tuple[int, int, int, int]:
    result = tuple(int(part) for part in value.split(","))
    if len(result) != 4:
        raise ValueError(f"Invalid tile box: {value}")
    return result


def category(row: dict[str, str]) -> tuple[str, str]:
    truth = int(row["label"])
    prediction = int(row["prediction"])
    if truth == 0 and prediction == 0:
        return "true_normal", "Correct normal"
    if truth == 0 and prediction == 1:
        return "false_alarm", "False alarm"
    if truth == 1 and prediction == 1:
        return "detected_stain", "Detected stain"
    return "missed_stain", "Missed stain"


def save_assets(
    index: int,
    row: dict[str, str],
    response: np.ndarray,
    source: Path,
    assets: Path,
) -> dict[str, str]:
    path = source_path(source, row["image"])
    if not path.is_file():
        raise FileNotFoundError(path)

    resample = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as opened:
        original = opened.convert("RGB")
        tile = original.crop(parse_box(row["best_tile_box"]))

        original_thumb = original.copy()
        # Keep a larger uncropped preview for the full-image gallery.
        original_thumb.thumbnail((1000, 750), resample)
        tile_thumb = tile.resize((315, 315), resample)

    heatmap = colorize(response).resize((315, 315), resample)
    overlay = Image.blend(ImageEnhance.Contrast(tile_thumb).enhance(0.9), heatmap, 0.42)

    stem = f"{index:03d}_{path.stem}"
    names = {
        "original": f"assets/{stem}_original.jpg",
        "tile": f"assets/{stem}_tile.jpg",
        "heatmap": f"assets/{stem}_heatmap.jpg",
        "overlay": f"assets/{stem}_overlay.jpg",
    }
    original_thumb.save(assets / f"{stem}_original.jpg", quality=86, optimize=True)
    tile_thumb.save(assets / f"{stem}_tile.jpg", quality=88, optimize=True)
    heatmap.save(assets / f"{stem}_heatmap.jpg", quality=90, optimize=True)
    overlay.save(assets / f"{stem}_overlay.jpg", quality=88, optimize=True)
    return names


def build(source: Path, results: Path, output: Path) -> None:
    with (results / "final_per_image.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    maps = np.load(results / "best_tile_maps_float16.npz")["maps"].astype(np.float32)
    if len(rows) != len(maps):
        raise ValueError(f"CSV has {len(rows)} rows but map archive has {len(maps)} maps")

    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    counts = {"all": len(rows), "true_normal": 0, "false_alarm": 0, "detected_stain": 0, "missed_stain": 0}
    cards = []
    original_cards = []

    for index, (row, response) in enumerate(zip(rows, maps)):
        group, label = category(row)
        counts[group] += 1
        names = save_assets(index, row, response, source, assets)
        score = float(row["tile_top2_mean_score"])
        threshold = float(row["frozen_threshold"])
        card_class = "correct" if int(row["correct"]) else "error"
        cards.append(
            f"""
<article class="card {card_class}" data-group="{group}" data-name="{html.escape(row['image'].lower())}">
  <header><div><strong>{html.escape(Path(row['image']).name)}</strong><span>{label}</span></div>
  <div class="score">{score:.2f}<small>cutoff {threshold:.2f}</small></div></header>
  <div class="images">
    <figure><img src="{names['original']}" loading="lazy"><figcaption>Full image</figcaption></figure>
    <figure><img src="{names['tile']}" loading="lazy"><figcaption>Highest-response tile</figcaption></figure>
    <figure><img src="{names['heatmap']}" loading="lazy"><figcaption>Residual heatmap</figcaption></figure>
    <figure><img src="{names['overlay']}" loading="lazy"><figcaption>Qualitative overlay</figcaption></figure>
  </div>
  <footer>{html.escape(row['image'])} · box {html.escape(row['best_tile_box'])}</footer>
</article>"""
        )
        original_cards.append(
            f"""
<article class="original-card {card_class}" data-group="{group}" data-name="{html.escape(row['image'].lower())}">
  <img src="{names['original']}" loading="lazy">
  <div><strong>{html.escape(Path(row['image']).name)}</strong><span>{label} · score {score:.2f}</span></div>
</article>"""
        )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fabric stain final evaluation gallery</title>
<style>
:root{{--bg:#0b0d12;--panel:#141821;--line:#293140;--text:#f5f7fb;--muted:#9ea8b8;--good:#55d68b;--bad:#ff6b6b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}}
.top{{position:sticky;top:0;z-index:2;padding:20px 28px;background:#0b0d12ee;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}}
h1{{margin:0 0 5px;font-size:25px}} .summary{{color:var(--muted);margin-bottom:14px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap}} button,input,.switch{{border:1px solid var(--line);border-radius:8px;background:#171c26;color:var(--text);padding:9px 12px}} .switch{{text-decoration:none}}
button.active{{background:#e8edf7;color:#111827}} input{{min-width:230px}}
main{{padding:24px;display:grid;gap:20px}} .card{{background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--good);border-radius:12px;overflow:hidden}}
.card.error{{border-left-color:var(--bad)}} .card header{{display:flex;justify-content:space-between;align-items:center;padding:14px 16px}}
.card header span{{display:block;color:var(--muted);font-size:13px;margin-top:3px}} .score{{font-size:22px;font-weight:700;text-align:right}}
.score small{{display:block;color:var(--muted);font-size:11px;font-weight:400}} .images{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:2px;background:var(--line)}}
figure{{margin:0;background:#080a0f;text-align:center}} img{{width:100%;height:260px;object-fit:contain;display:block;cursor:zoom-in}} figcaption{{padding:8px;color:var(--muted);font-size:12px}}
footer{{padding:10px 16px;color:var(--muted);font-size:12px}} .hidden{{display:none}}
#modal{{display:none;position:fixed;inset:0;background:#000d;z-index:5;align-items:center;justify-content:center;padding:20px}} #modal img{{max-width:95vw;max-height:94vh;object-fit:contain;cursor:zoom-out}}
@media(max-width:850px){{.images{{grid-template-columns:repeat(2,1fr)}}img{{height:220px}}}}
</style></head><body>
<section class="top"><h1>Fabric stain final evaluation</h1>
<div class="summary">{counts['all']} images · {counts['detected_stain']} detected stains · {counts['missed_stain']} missed stains · {counts['false_alarm']} false alarm · {counts['true_normal']} correct normals</div>
<div class="controls">
<a class="switch" href="originals.html">View full images only</a>
<button class="active" data-filter="all">All ({counts['all']})</button>
<button data-filter="detected_stain">Detected ({counts['detected_stain']})</button>
<button data-filter="missed_stain">Missed ({counts['missed_stain']})</button>
<button data-filter="false_alarm">False alarms ({counts['false_alarm']})</button>
<button data-filter="true_normal">Correct normals ({counts['true_normal']})</button>
<input id="search" placeholder="Search filename…">
</div></section><main>{''.join(cards)}</main><div id="modal"><img></div>
<script>
let filter='all', query=''; const cards=[...document.querySelectorAll('.card')];
function update(){{cards.forEach(c=>c.classList.toggle('hidden',!((filter==='all'||c.dataset.group===filter)&&c.dataset.name.includes(query))))}}
document.querySelectorAll('button[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelector('button.active').classList.remove('active');b.classList.add('active');filter=b.dataset.filter;update()}});
document.querySelector('#search').oninput=e=>{{query=e.target.value.toLowerCase().trim();update()}};
const modal=document.querySelector('#modal'), zoom=modal.querySelector('img');
document.querySelectorAll('.card img').forEach(img=>img.onclick=()=>{{zoom.src=img.src;modal.style.display='flex'}}); modal.onclick=()=>modal.style.display='none';
</script></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")

    originals_page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>All full fabric images</title><style>
:root{{--bg:#0b0d12;--panel:#141821;--line:#293140;--text:#f5f7fb;--muted:#9ea8b8;--good:#55d68b;--bad:#ff6b6b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}}
.top{{position:sticky;top:0;z-index:2;padding:20px 28px;background:#0b0d12ee;border-bottom:1px solid var(--line)}}h1{{margin:0 0 5px}}
.summary{{color:var(--muted);margin-bottom:14px}}.controls{{display:flex;gap:8px;flex-wrap:wrap}}button,input,.switch{{border:1px solid var(--line);border-radius:8px;background:#171c26;color:var(--text);padding:9px 12px;text-decoration:none}}
button.active{{background:#e8edf7;color:#111827}}input{{min-width:230px}}main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;padding:24px}}
.original-card{{background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--good);border-radius:10px;overflow:hidden}}.original-card.error{{border-left-color:var(--bad)}}
.original-card img{{display:block;width:100%;height:auto;max-height:70vh;object-fit:contain;background:#080a0f;cursor:zoom-in}}.original-card div{{padding:11px 14px}}.original-card span{{display:block;color:var(--muted);margin-top:3px}}.hidden{{display:none}}
#modal{{display:none;position:fixed;inset:0;background:#000e;z-index:5;align-items:center;justify-content:center;padding:16px}}#modal img{{max-width:97vw;max-height:96vh;object-fit:contain;cursor:zoom-out}}
@media(max-width:900px){{main{{grid-template-columns:1fr}}}}
</style></head><body><section class="top"><h1>Complete source frames</h1><div class="summary">All {counts['all']} final-evaluation images shown without cropping. The fabric itself was photographed close-up.</div><div class="controls">
<a class="switch" href="index.html">View model tiles and heatmaps</a>
<button class="active" data-filter="all">All ({counts['all']})</button><button data-filter="detected_stain">Detected ({counts['detected_stain']})</button><button data-filter="missed_stain">Missed ({counts['missed_stain']})</button><button data-filter="false_alarm">False alarms ({counts['false_alarm']})</button><button data-filter="true_normal">Correct normals ({counts['true_normal']})</button><input id="search" placeholder="Search filename…"></div></section>
<main>{''.join(original_cards)}</main><div id="modal"><img></div><script>
let filter='all',query='';const cards=[...document.querySelectorAll('.original-card')];function update(){{cards.forEach(c=>c.classList.toggle('hidden',!((filter==='all'||c.dataset.group===filter)&&c.dataset.name.includes(query))))}}document.querySelectorAll('button[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelector('button.active').classList.remove('active');b.classList.add('active');filter=b.dataset.filter;update()}});document.querySelector('#search').oninput=e=>{{query=e.target.value.toLowerCase().trim();update()}};const modal=document.querySelector('#modal'),zoom=modal.querySelector('img');document.querySelectorAll('.original-card img').forEach(img=>img.onclick=()=>{{zoom.src=img.src;modal.style.display='flex'}});modal.onclick=()=>modal.style.display='none';
</script></body></html>"""
    (output / "originals.html").write_text(originals_page, encoding="utf-8")
    print("Gallery:", output / "index.html")
    print("Full images:", output / "originals.html")
    print("Counts:", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source, args.results, args.output)
