"""Audit printed-label photographs and create a compact contact sheet.

This script is deliberately dependency-light so it can run before the training
environment is installed. It requires only Pillow and NumPy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


NAME_RE = re.compile(r"^(?P<label>[A-Za-z]\d{2})_(?P<view>B\d+)_(?P<shot>\d+)\.(?:jpg|jpeg)$", re.I)


def difference_hash(gray: Image.Image, size: int = 16) -> str:
    pixels = np.asarray(gray.resize((size + 1, size), Image.Resampling.LANCZOS))
    bits = pixels[:, 1:] > pixels[:, :-1]
    packed = np.packbits(bits.reshape(-1))
    return packed.tobytes().hex()


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def audit_image(path: Path) -> tuple[dict, Image.Image]:
    raw = path.read_bytes()
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    gray_image = image.convert("L")
    gray = np.asarray(gray_image.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)

    # Mean squared horizontal/vertical gradient: a simple relative sharpness signal.
    sharpness = float((np.diff(gray, axis=0) ** 2).mean() + (np.diff(gray, axis=1) ** 2).mean())
    parsed = NAME_RE.match(path.name)
    metadata = parsed.groupdict() if parsed else {"label": None, "view": None, "shot": None}
    record = {
        "file": path.name,
        **metadata,
        "width": image.width,
        "height": image.height,
        "megapixels": round(image.width * image.height / 1_000_000, 3),
        "mean_brightness": round(float(gray.mean()), 2),
        "dark_clip_percent": round(float((gray <= 5).mean() * 100), 3),
        "bright_clip_percent": round(float((gray >= 250).mean() * 100), 3),
        "sharpness_score": round(sharpness, 2),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dhash": difference_hash(gray_image),
    }
    return record, image


def make_contact_sheet(items: list[tuple[dict, Image.Image]], destination: Path) -> None:
    columns = 5
    cell_width, cell_height = 300, 260
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "#171717")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)

    for index, (record, image) in enumerate(items):
        row, column = divmod(index, columns)
        fitted = ImageOps.contain(image, (cell_width - 16, cell_height - 50), Image.Resampling.LANCZOS)
        x = column * cell_width + (cell_width - fitted.width) // 2
        y = row * cell_height + 8
        canvas.paste(fitted, (x, y))
        caption = f"{record['file']}  sharp={record['sharpness_score']:.0f}"
        draw.text((column * cell_width + 8, row * cell_height + cell_height - 32), caption, fill="white", font=font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(
        path for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not paths:
        raise SystemExit(f"No JPEG images found in {args.source}")

    items = [audit_image(path) for path in paths]
    records = [record for record, _ in items]
    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "photo_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    by_label: dict[str, list[str]] = defaultdict(list)
    by_view: Counter[str] = Counter()
    for record in records:
        by_label[str(record["label"])].append(str(record["view"]))
        by_view[str(record["view"])] += 1

    exact_duplicates = []
    by_sha: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_sha[str(record["sha256"])].append(str(record["file"]))
    exact_duplicates = [group for group in by_sha.values() if len(group) > 1]

    near_duplicates = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            distance = hamming_hex(str(left["dhash"]), str(right["dhash"]))
            if distance <= 4:
                near_duplicates.append({"left": left["file"], "right": right["file"], "distance": distance})

    sharpness = np.asarray([float(record["sharpness_score"]) for record in records])
    brightness = np.asarray([float(record["mean_brightness"]) for record in records])
    report = {
        "source": str(args.source.resolve()),
        "image_count": len(records),
        "labels": {label: sorted(views) for label, views in sorted(by_label.items())},
        "view_counts": dict(sorted(by_view.items())),
        "dimensions": dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
        "sharpness": {
            "minimum": round(float(sharpness.min()), 2),
            "median": round(float(np.median(sharpness)), 2),
            "maximum": round(float(sharpness.max()), 2),
            "lowest_five": [
                {"file": record["file"], "score": record["sharpness_score"]}
                for record in sorted(records, key=lambda item: float(item["sharpness_score"]))[:5]
            ],
        },
        "brightness": {
            "minimum_mean": round(float(brightness.min()), 2),
            "median_mean": round(float(np.median(brightness)), 2),
            "maximum_mean": round(float(brightness.max()), 2),
        },
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_pairs_dhash_le_4": near_duplicates,
        "filename_parse_failures": [r["file"] for r in records if r["label"] is None],
    }
    (args.output / "photo_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_contact_sheet(items, args.output / "contact_sheet.jpg")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
