from pathlib import Path
import csv

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "output" / "pdf"
DATA_DIR = ROOT / "data" / "printed_labels"
PDF_PATH = PDF_DIR / "printed_label_collection_sheet.pdf"
PNG_PATH = DATA_DIR / "reference" / "label_master.png"
MANIFEST_PATH = DATA_DIR / "manifests" / "capture_manifest.csv"

LABEL_W_MM = 90
LABEL_H_MM = 55


def ensure_structure():
    folders = [
        DATA_DIR / "reference",
        DATA_DIR / "manifests",
        DATA_DIR / "raw" / "train_normal",
        DATA_DIR / "raw" / "val_normal",
        DATA_DIR / "raw" / "test" / "good",
        DATA_DIR / "raw" / "test" / "missing_print",
        DATA_DIR / "raw" / "test" / "smudge",
        DATA_DIR / "raw" / "test" / "tear",
        DATA_DIR / "masks" / "test" / "missing_print",
        DATA_DIR / "masks" / "test" / "smudge",
        DATA_DIR / "masks" / "test" / "tear",
        PDF_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        if folder.name not in {"reference", "manifests"}:
            (folder / ".gitkeep").touch()

    headers = [
        "image_id",
        "split",
        "physical_label_id",
        "session_id",
        "condition",
        "defect_type",
        "image_path",
        "mask_path",
        "camera",
        "lighting",
        "distance_cm",
        "notes",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(headers)


def register_fonts():
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("LabelRegular", str(regular)))
        pdfmetrics.registerFont(TTFont("LabelBold", str(bold)))
        return "LabelRegular", "LabelBold"
    return "Helvetica", "Helvetica-Bold"


def barcode_widths():
    # A fixed, deterministic visual pattern. It is not intended for scanning.
    digits = "104230224202601"
    widths = [1, 1, 2, 1, 3, 1]
    for digit in digits:
        value = int(digit)
        widths.extend([1 + value % 3, 1, 1 + (value // 3) % 3, 1])
    widths.extend([3, 1, 2, 1, 1])
    return widths


def draw_pdf_label(c, x, y, regular, bold):
    w, h = LABEL_W_MM * mm, LABEL_H_MM * mm
    c.setFillColorRGB(1, 1, 1)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.55 * mm)
    c.rect(x, y, w, h, fill=1, stroke=1)

    marker = 4.2 * mm
    inset = 2.0 * mm
    for mx, my in [
        (x + inset, y + h - inset - marker),
        (x + w - inset - marker, y + h - inset - marker),
        (x + inset, y + inset),
        (x + w - inset - marker, y + inset),
    ]:
        c.setFillColorRGB(0, 0, 0)
        c.rect(mx, my, marker, marker, fill=1, stroke=0)

    left = x + 9 * mm
    right = x + w - 9 * mm
    c.setFillColorRGB(0.06, 0.06, 0.06)
    c.setFont(bold, 15)
    c.drawString(left, y + h - 10.5 * mm, "QUALITY CONTROL")
    c.setLineWidth(0.35 * mm)
    c.line(left, y + h - 12.5 * mm, right, y + h - 12.5 * mm)

    c.setFont(regular, 8.5)
    c.drawString(left, y + h - 18.0 * mm, "ITEM")
    c.drawString(left, y + h - 23.5 * mm, "BATCH")
    c.drawString(left, y + h - 29.0 * mm, "LOT")
    c.setFont(bold, 9.3)
    c.drawString(left + 16 * mm, y + h - 18.0 * mm, "A-104")
    c.drawString(left + 16 * mm, y + h - 23.5 * mm, "2026-01")
    c.drawString(left + 16 * mm, y + h - 29.0 * mm, "L-230224")

    bar_x = left
    bar_y = y + 8.0 * mm
    bar_h = 10.0 * mm
    widths = barcode_widths()
    total = sum(widths)
    unit = (right - left) / total
    black = True
    for width in widths:
        if black:
            c.rect(bar_x, bar_y, width * unit, bar_h, fill=1, stroke=0)
        bar_x += width * unit
        black = not black

    c.setFillColorRGB(0.12, 0.12, 0.12)
    c.setFont(regular, 6.7)
    c.drawCentredString(x + w / 2, y + 5.2 * mm, "A104  230224  202601")


def draw_crop_marks(c, x, y):
    w, h = LABEL_W_MM * mm, LABEL_H_MM * mm
    length = 3.0 * mm
    gap = 1.2 * mm
    c.setStrokeColorRGB(0.45, 0.45, 0.45)
    c.setLineWidth(0.15 * mm)
    for px in (x, x + w):
        c.line(px, y - gap, px, y - gap - length)
        c.line(px, y + h + gap, px, y + h + gap + length)
    for py in (y, y + h):
        c.line(x - gap, py, x - gap - length, py)
        c.line(x + w + gap, py, x + w + gap + length, py)


def create_pdf():
    regular, bold = register_fonts()
    page_w, page_h = A4
    c = canvas.Canvas(str(PDF_PATH), pagesize=A4)
    c.setTitle("Printed Label Anomaly Inspection - Collection Sheet")
    gap_x = 8 * mm
    gap_y = 7 * mm
    grid_w = 2 * LABEL_W_MM * mm + gap_x
    grid_h = 4 * LABEL_H_MM * mm + 3 * gap_y
    start_x = (page_w - grid_w) / 2
    start_y = (page_h - grid_h) / 2

    for row in range(4):
        for col in range(2):
            x = start_x + col * (LABEL_W_MM * mm + gap_x)
            y = start_y + (3 - row) * (LABEL_H_MM * mm + gap_y)
            draw_pdf_label(c, x, y, regular, bold)
            draw_crop_marks(c, x, y)

    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont(regular, 6.5)
    c.drawCentredString(page_w / 2, 5.5 * mm, "PRINT AT ACTUAL SIZE / 100%  -  DO NOT FIT TO PAGE")
    c.save()


def pil_font(name, size):
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def create_reference_png():
    # 300 dpi: 90 x 55 mm becomes approximately 1063 x 650 px.
    width, height = 1063, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, width - 4, height - 4), outline="black", width=7)
    marker = 50
    inset = 24
    for mx, my in [
        (inset, inset),
        (width - inset - marker, inset),
        (inset, height - inset - marker),
        (width - inset - marker, height - inset - marker),
    ]:
        draw.rectangle((mx, my, mx + marker, my + marker), fill="black")

    bold = pil_font("arialbd.ttf", 58)
    field = pil_font("arial.ttf", 31)
    value = pil_font("arialbd.ttf", 34)
    small = pil_font("arial.ttf", 24)
    left, right = 108, width - 108
    draw.text((left, 80), "QUALITY CONTROL", font=bold, fill="black")
    draw.line((left, 158, right, 158), fill="black", width=4)
    rows = [("ITEM", "A-104"), ("BATCH", "2026-01"), ("LOT", "L-230224")]
    for idx, (key, val) in enumerate(rows):
        y = 185 + idx * 63
        draw.text((left, y), key, font=field, fill="black")
        draw.text((315, y - 2), val, font=value, fill="black")

    bar_left, bar_right = left, right
    bar_y, bar_h = 395, 118
    widths = barcode_widths()
    total = sum(widths)
    unit = (bar_right - bar_left) / total
    x = float(bar_left)
    black = True
    for segment in widths:
        next_x = x + segment * unit
        if black:
            draw.rectangle((round(x), bar_y, round(next_x), bar_y + bar_h), fill="black")
        x = next_x
        black = not black
    footer = "A104  230224  202601"
    box = draw.textbbox((0, 0), footer, font=small)
    draw.text(((width - (box[2] - box[0])) / 2, 535), footer, font=small, fill="black")
    image.save(PNG_PATH, dpi=(300, 300))


if __name__ == "__main__":
    ensure_structure()
    create_reference_png()
    create_pdf()
    print(PDF_PATH)
    print(PNG_PATH)
    print(MANIFEST_PATH)
