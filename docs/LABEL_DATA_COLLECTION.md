# Printed-label data collection protocol

## Objective

Build a small, leakage-safe dataset for visual printed-label defects. The first version targets **missing print, smudges, and tears** on one fixed label design. The DTU-Net diffusion model will train only on normal labels and will use reconstruction differences to locate unfamiliar defects.

The current scope is visual appearance. A wrong but cleanly printed batch number requires OCR or semantic validation and is outside this first version.

## 1. Print and prepare the labels

1. Print `output/pdf/printed_label_collection_sheet.pdf` on A4 paper at **Actual Size / 100%**. Disable “Fit to page.”
2. Cut along the crop marks.
3. Print at least three sheets so the pilot can use 21 separate physical samples.
4. Write a unique ID on the **back** of every physical label: `N01`, `N02`, … and `D01`, `D02`, … . Do not write on the front.

## 2. Keep the splits physically separate

Never place photographs of the same physical label in more than one split. This prevents the model from being tested on paper texture, cuts, or marks it already saw during training.

Pilot target:

| Split | Separate physical labels | Captures per label | Target images |
|---|---:|---:|---:|
| Train normal | 8 | 5 | 40 |
| Validation normal | 2 | 5 | 10 |
| Test normal | 2 | 5 | 10 |
| Test missing print | 3 | 5 | 15 |
| Test smudge | 3 | 5 | 15 |
| Test tear | 3 | 5 | 15 |
| **Total** | **21** |  | **105** |

If fewer printed labels are available, start smaller but preserve separate physical labels for each split.

## 3. Capture normal images

- Begin with a controlled setup: phone parallel to the label, about 30–40 cm away, diffuse light, plain background, no filters.
- Keep the four black corner markers visible. They will support perspective correction and alignment.
- Take small natural variations in position, rotation, distance, and lighting.
- Use at least two capture sessions if possible. Record the session, camera, lighting, and distance in `data/printed_labels/manifests/capture_manifest.csv`.
- Save full-resolution JPEG or PNG files. Avoid screenshots and messaging-app compression.

Use filenames such as:

`N01_S01_001.jpg`, `N01_S01_002.jpg`, and `D01_S02_001.jpg`.

## 4. Create test defects

Create defects only on labels reserved for the test set.

- **Missing print:** cover or remove a small part of a word, number, border, or barcode using white correction material. Vary size and location.
- **Smudge:** add a controlled dark ink or graphite smear over or next to printed content. Include light and heavy examples.
- **Tear:** make a notch or tear that affects the label edge or printed region. Keep the sample safe to handle.

Capture the defective label from five small pose and lighting variations. Keep each physical label in one defect category.

## 5. Store the images

Place files in these folders:

```text
data/printed_labels/raw/
  train_normal/
  val_normal/
  test/
    good/
    missing_print/
    smudge/
    tear/
```

Masks will be drawn later only for the defective test images:

```text
data/printed_labels/masks/test/
  missing_print/
  smudge/
  tear/
```

The mask filename must match its image stem. For example, `D01_S02_001.jpg` uses `D01_S02_001.png` as its binary mask.

## Completion check

Before training, verify that:

- every training and validation image is normal;
- physical label IDs do not cross splits;
- all four corner markers are visible;
- each image has a manifest row;
- test defects vary in size and location;
- original full-resolution files are preserved.

After collection, the next implementation step is automatic corner-marker alignment, resizing, and a dataset audit before any training begins.
