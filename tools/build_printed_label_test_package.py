"""Assemble the real printed-label test set and independent draft masks.

The annotation rules in this file were defined from registered inputs before
any DTU-Net prediction was produced for the real test set.  Generated overlays
must be reviewed before changing the package status from draft to locked.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "datasets" / "printed_label_test_locked_v1"
REFERENCE = ROOT / "data" / "printed_labels" / "reference" / "label_master.png"
SIZE = (1063, 650)

SOURCES = {
    "normal": ROOT / "output" / "datasets" / "printed_label_test_normal_v1" / "test" / "normal",
    "missing_print": ROOT / "output" / "datasets" / "printed_label_defective_miss_v1" / "test" / "missing_print",
    "smudge": ROOT / "output" / "datasets" / "printed_label_defective_smudge_v1" / "test" / "smudge",
    "tear": ROOT / "output" / "datasets" / "printed_label_defective_tear_v1" / "test" / "tear",
}

# Full affected regions for the white overlays. They represent a
# missing/covered-print simulation rather than a physical ink-transfer failure.
MISSING_POLYGONS = {
    "M01": [(428, 386), (995, 386), (995, 628), (428, 628)],
    "M02": [(270, 164), (443, 164), (443, 392), (270, 392)],
    "M03": [(246, 282), (505, 272), (521, 408), (265, 418)],
}

# Smudge pixels are selected only where added dark ink appears over a light part
# of the clean reference. These ROIs prevent ordinary printed content from
# becoming ground truth merely because of sub-pixel registration differences.
SMUDGE_ROIS = {
    "S01": [(450, 420, 845, 565)],
    "S02": [(455, 55, 1015, 155), (750, 415, 1062, 565)],
    "S03": [(265, 215, 500, 335)],
}

# Missing paper regions in canonical coordinates. One mask is shared across the
# five views of each physical label because marker registration removes pose.
TEAR_POLYGONS = {
    "T01": [(1015, 0), (1062, 0), (1062, 649), (825, 649)],
    "T02": [(0, 185), (275, 215), (325, 305), (0, 292)],
    "T03": [(0, 455), (385, 649), (0, 649)],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_id(stem: str) -> str:
    return stem.split("_", 1)[0]


def capture_condition(stem: str) -> str:
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else ""


def polygon_mask(points: list[tuple[int, int]]) -> Image.Image:
    mask = Image.new("L", SIZE, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


def smudge_mask(image: Image.Image, label_id: str, reference: np.ndarray) -> Image.Image:
    observed = np.asarray(image.convert("L"))
    roi = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    for left, top, right, bottom in SMUDGE_ROIS[label_id]:
        roi[top:bottom, left:right] = True
    # Exclude a four-pixel neighbourhood around expected printed ink. This makes
    # the mask conservative: it retains clearly added strokes while preventing
    # small registration shifts at letters and barcode bars from becoming truth.
    expected_ink = Image.fromarray((reference < 200).astype(np.uint8) * 255)
    expected_ink = np.asarray(expected_ink.filter(ImageFilter.MaxFilter(9))) > 0
    selected = roi & (observed < 120) & ~expected_ink
    mask = Image.fromarray(selected.astype(np.uint8) * 255)
    return mask.filter(ImageFilter.MaxFilter(3))


def make_mask(image: Image.Image, category: str, label_id: str, reference: np.ndarray) -> tuple[Image.Image, str]:
    if category == "normal":
        return Image.new("L", SIZE, 0), "empty_normal_mask"
    if category == "missing_print":
        return polygon_mask(MISSING_POLYGONS[label_id]), "manual_full_overlay_polygon"
    if category == "smudge":
        return smudge_mask(image, label_id, reference), "manual_roi_plus_added_dark_ink_rule"
    if category == "tear":
        return polygon_mask(TEAR_POLYGONS[label_id]), "manual_missing_paper_polygon"
    raise ValueError(category)


def overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    alpha = mask.point(lambda value: 105 if value else 0)
    color = Image.new("RGBA", SIZE, (255, 36, 72, 0))
    color.putalpha(alpha)
    return Image.alpha_composite(base, color).convert("RGB")


def review_sheet(rows: list[dict], destination: Path) -> None:
    chosen = []
    for label_id in ["M01", "M02", "M03", "S01", "S02", "S03", "T01", "T02", "T03"]:
        chosen.append(next(row for row in rows if row["physical_id"] == label_id))
    columns = 3
    panel_width, panel_height = 354, 217
    row_height = panel_height + 55
    sheet = Image.new("RGB", (columns * panel_width, len(chosen) * row_height + 50), "#171717")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    draw.text((12, 12), "LOCKED GROUND TRUTH - created before model evaluation", fill="white", font=font)
    for row_index, row in enumerate(chosen):
        image = Image.open(OUTPUT / row["image"]).convert("RGB")
        mask = Image.open(OUTPUT / row["mask"]).convert("L")
        panels = [image, mask.convert("RGB"), overlay(image, mask)]
        y = 50 + row_index * row_height
        for column, panel in enumerate(panels):
            sheet.paste(panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS), (column * panel_width, y))
        label = f"{row['physical_id']} | input / mask / overlay | {row['annotation_method']}"
        draw.text((12, y + panel_height + 10), label, fill="white", font=font)
    sheet.save(destination, quality=94, subsampling=0)


def main() -> None:
    if not REFERENCE.is_file():
        raise SystemExit(f"Missing reference: {REFERENCE}")
    reference = np.asarray(Image.open(REFERENCE).convert("L").resize(SIZE, Image.Resampling.BILINEAR))
    rows: list[dict] = []
    for category, source_dir in SOURCES.items():
        paths = sorted(source_dir.glob("*.png"))
        if not paths:
            raise SystemExit(f"No registered images found: {source_dir}")
        image_dir = OUTPUT / "images" / category
        mask_dir = OUTPUT / "ground_truth" / category
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for source in paths:
            image = Image.open(source).convert("RGB")
            if image.size != SIZE:
                raise ValueError(f"Unexpected image size {image.size}: {source}")
            label_id = physical_id(source.stem)
            destination = image_dir / source.name
            shutil.copy2(source, destination)
            mask, method = make_mask(image, category, label_id, reference)
            mask_destination = mask_dir / source.name
            mask.save(mask_destination, optimize=True)
            rows.append(
                {
                    "image": destination.relative_to(OUTPUT).as_posix(),
                    "mask": mask_destination.relative_to(OUTPUT).as_posix(),
                    "category": category,
                    "is_anomaly": int(category != "normal"),
                    "physical_id": label_id,
                    "capture_condition": capture_condition(source.stem),
                    "annotation_method": method,
                    "image_sha256": sha256(destination),
                    "mask_sha256": sha256(mask_destination),
                    "mask_pixels": int(np.count_nonzero(np.asarray(mask))),
                }
            )

    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    exclusions = [
        {
            "source": "T01_B3_082.jpg",
            "category": "tear",
            "reason": "Only two reliable fiducials remain; no defensible model-input registration.",
        },
        {
            "source": "T01_B4_085.jpg",
            "category": "tear",
            "reason": "Only two reliable fiducials remain; no defensible model-input registration.",
        },
    ]
    with (OUTPUT / "registration_exclusions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exclusions[0]))
        writer.writeheader()
        writer.writerows(exclusions)

    review_sheet(rows, OUTPUT / "mask_review_sheet.jpg")
    counts = {category: sum(row["category"] == category for row in rows) for category in SOURCES}
    protocol = {
        "status": "locked_for_first_real_test",
        "model_predictions_viewed_before_annotation": False,
        "registered_image_count": len(rows),
        "counts": counts,
        "registered_anomaly_count": sum(row["is_anomaly"] for row in rows),
        "registration_failure_count": len(exclusions),
        "physical_test_label_count": len({row["physical_id"] for row in rows}),
        "interpretation": [
            "M01-M03 are missing/covered-print simulations made with white paper overlays.",
            "The 43 registered anomaly images represent nine physical defects with repeated captures.",
            "Model-only metrics exclude two registration failures; end-to-end metrics report them separately.",
        ],
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(OUTPUT), "zip", OUTPUT.parent, OUTPUT.name)
    print(json.dumps(protocol, indent=2))
    print("Review:", OUTPUT / "mask_review_sheet.jpg")
    print("Archive:", archive)


if __name__ == "__main__":
    main()
