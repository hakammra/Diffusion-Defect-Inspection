"""Register real printed-label photos and create a training-ready archive."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.label_registration import register_label_detailed  # noqa: E402


def contact_sheet(paths: list[Path], destination: Path) -> None:
    columns, cell_width, cell_height = 5, 300, 230
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#171717")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_width - 12, cell_height - 42), Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * cell_height + 6
        sheet.paste(image, (x, y))
        draw.text((column * cell_width + 8, row * cell_height + cell_height - 28), path.stem, fill="white", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-path",
        default="train/normal",
        help="Destination below the dataset root, for example train/normal or validation/normal",
    )
    args = parser.parse_args()

    raw_paths = sorted(path for path in args.source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})
    if not raw_paths:
        raise SystemExit(f"No JPEGs found in {args.source}")
    split_parts = Path(args.split_path).parts
    if not split_parts or any(part in {"", ".", ".."} for part in split_parts):
        raise SystemExit(f"Unsafe split path: {args.split_path!r}")
    image_dir = args.output.joinpath(*split_parts)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for path in raw_paths:
        try:
            registered, markers, registration = register_label_detailed(path)
            destination = image_dir / f"{path.stem}.png"
            registered.save(destination, optimize=True)
            rows.append({
                "source": path.name,
                "registered": destination.relative_to(args.output).as_posix(),
                "markers_xy": json.dumps(markers.round(2).tolist()),
                "registration_mode": registration["mode"],
                "registration_quality": round(float(registration["quality"]), 6),
            })
        except Exception as error:  # retain a complete audit instead of hiding failed files
            failures.append({"source": path.name, "error": str(error)})

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source", "registered", "markers_xy", "registration_mode", "registration_quality"],
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "source_count": len(raw_paths),
        "registered_count": len(rows),
        "failed_count": len(failures),
        "failures": failures,
        "output_size": [1063, 650],
        "split": "/".join(split_parts),
        "registration_modes": dict(
            sorted(__import__("collections").Counter(row["registration_mode"] for row in rows).items())
        ),
        "warning": "Registration success alone does not establish detection accuracy.",
    }
    (args.output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    prepared_paths = sorted(image_dir.glob("*.png"))
    if prepared_paths:
        contact_sheet(prepared_paths, args.output / "registered_contact_sheet.jpg")
    archive = shutil.make_archive(str(args.output), "zip", args.output.parent, args.output.name)
    print(json.dumps(report, indent=2))
    print("Archive:", archive)


if __name__ == "__main__":
    main()
