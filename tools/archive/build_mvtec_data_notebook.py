from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r'C:\Users\abdul\Documents\LabelInspect')
NB = ROOT / 'notebooks' / '03_mvtec_leather_data_check.ipynb'


def md(text):
    return {'cell_type':'markdown','metadata':{},'source':[line+'\n' for line in text.strip().splitlines()]}


def code(text):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':[line+'\n' for line in text.strip().splitlines()]}

cells = [
md('''# Stage 3A — MVTec AD leather data check

This notebook prepares the **reduced reproduction protocol** for the WACV 2025 DTU-Net/Tsimplex paper. It does not train a model.

It verifies one official MVTec AD category (`leather`), checks every image and defect-mask pair, creates a deterministic split using only `train/good`, and writes a manifest with file hashes. Test annotations are recorded for later evaluation and are never used to fit or tune the model.

Obtain MVTec AD from the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and follow its CC BY-NC-SA 4.0 terms. Upload the extracted dataset to Colab or Google Drive. Expected structure:

```text
mvtec_anomaly_detection/
  leather/
    train/good/*.png
    test/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/*_mask.png
```
'''),
code('''from pathlib import Path
import hashlib, json, os, random, shutil
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

SEED = 230224
VAL_FRACTION = 0.20

# EDIT THIS if your extracted dataset is elsewhere.
# Google Drive example:
# from google.colab import drive; drive.mount('/content/drive')
# DATA_ROOT = Path('/content/drive/MyDrive/mvtec_anomaly_detection')
DATA_ROOT = Path(os.environ.get('MVTEC_ROOT', '/content/mvtec_anomaly_detection'))

WORK = Path('/kaggle/working/labelinspect') if Path('/kaggle/working').exists() else Path('/content/labelinspect')
OUTPUT = WORK / 'mvtec_leather_check'
OUTPUT.mkdir(parents=True, exist_ok=True)
print('Dataset root:', DATA_ROOT)
print('Output:', OUTPUT)
'''),
code('''def locate_category(root: Path, category='leather') -> Path:
    candidates = [root/category, root/'mvtec_anomaly_detection'/category]
    if root.name == category:
        candidates.insert(0, root)
    for candidate in candidates:
        if (candidate/'train'/'good').is_dir() and (candidate/'test').is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find {category}/train/good under {root}. "
        "Edit DATA_ROOT in the previous cell so it points to the extracted MVTec AD root."
    )

category_root = locate_category(DATA_ROOT)
train_good = sorted((category_root/'train'/'good').glob('*.png'))
test_images = sorted((category_root/'test').glob('*/*.png'))
assert train_good, 'No normal training images found.'
assert test_images, 'No test images found.'

# Validate that every file is readable and record dimensions.
def verify_images(paths):
    dimensions = Counter()
    for path in paths:
        with Image.open(path) as image:
            dimensions[image.size] += 1
            image.verify()
    return dimensions

dimensions = verify_images(train_good + test_images)
print('Category:', category_root)
print('Normal training images:', len(train_good))
print('Test images:', len(test_images))
print('Image dimensions:', dict(dimensions))
'''),
code('''rng = random.Random(SEED)
shuffled = train_good.copy()
rng.shuffle(shuffled)
val_count = max(1, round(len(shuffled) * VAL_FRACTION))
val_good = sorted(shuffled[:val_count])
fit_good = sorted(shuffled[val_count:])
assert set(fit_good).isdisjoint(val_good)

records = []
missing_masks = []

def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def rel(path: Path):
    return path.relative_to(category_root).as_posix()

for split, paths in [('train_normal', fit_good), ('val_normal', val_good)]:
    for path in paths:
        records.append({'protocol_split':split, 'kind':'good', 'image':rel(path),
                        'mask':None, 'image_sha256':sha256(path)})

kind_counts = Counter()
for path in test_images:
    kind = path.parent.name
    kind_counts[kind] += 1
    mask = None
    if kind != 'good':
        mask = category_root/'ground_truth'/kind/f'{path.stem}_mask.png'
        if not mask.is_file():
            missing_masks.append(str(mask))
    records.append({'protocol_split':'test', 'kind':kind, 'image':rel(path),
                    'mask':rel(mask) if mask else None,
                    'image_sha256':sha256(path),
                    'mask_sha256':sha256(mask) if mask and mask.is_file() else None})

assert not missing_masks, 'Missing ground-truth masks:\\n' + '\\n'.join(missing_masks[:10])
verify_images([category_root/r['mask'] for r in records if r.get('mask')])
manifest = {
    'dataset':'MVTec AD', 'category':'leather', 'source':'official provider',
    'license':'CC BY-NC-SA 4.0', 'seed':SEED, 'validation_fraction':VAL_FRACTION,
    'split_policy':'Deterministic 80/20 partition of original train/good; official test retained untouched.',
    'no_test_tuning':True, 'train_normal_count':len(fit_good),
    'val_normal_count':len(val_good), 'test_counts':dict(sorted(kind_counts.items())),
    'records':records,
}
(OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k != 'records'}, indent=2))
print('All test anomaly images have matching masks.')
'''),
code('''# Visual sanity check: normal images plus one example from each defect type.
normal_examples = fit_good[:2]
defect_examples = []
for kind in sorted(k for k in kind_counts if k != 'good'):
    defect_examples.append(next(p for p in test_images if p.parent.name == kind))
selected = normal_examples + defect_examples

fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3*len(selected)))
if len(selected) == 1:
    axes = np.array([axes])
for row, path in enumerate(selected):
    image = np.asarray(Image.open(path).convert('RGB'))
    kind = 'train/good' if path in normal_examples else path.parent.name
    axes[row,0].imshow(image); axes[row,0].set_title(kind); axes[row,0].axis('off')
    if kind == 'train/good':
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        mask_path = category_root/'ground_truth'/kind/f'{path.stem}_mask.png'
        mask = np.asarray(Image.open(mask_path).convert('L'))
    axes[row,1].imshow(mask, cmap='gray', vmin=0, vmax=255)
    axes[row,1].set_title('Ground truth mask' if kind != 'train/good' else 'No anomaly')
    axes[row,1].axis('off')
plt.tight_layout()
preview_path = OUTPUT/'dataset_preview.png'
fig.savefig(preview_path, dpi=140, bbox_inches='tight')
plt.show()
print('Saved:', preview_path)
'''),
code('''summary = {
    'status':'passed',
    'category':'leather',
    'category_root':str(category_root),
    'train_normal_count':len(fit_good),
    'val_normal_count':len(val_good),
    'test_counts':dict(sorted(kind_counts.items())),
    'all_images_readable':True,
    'all_anomaly_masks_matched':True,
    'manifest_sha256':sha256(OUTPUT/'manifest.json'),
    'next_step':'Train reduced DTU-Net/Tsimplex on train_normal; use val_normal for reconstruction threshold; evaluate once on official test.',
}
(OUTPUT/'dataset_check.json').write_text(json.dumps(summary, indent=2))
archive = shutil.make_archive(str(WORK/'mvtec_leather_data_check'), 'zip', OUTPUT)
print(json.dumps(summary, indent=2))
print('\\nDATA CHECK PASSED')
print('Download:', archive)
''')]

notebook = {
  'cells': cells,
  'metadata': {
    'accelerator':'GPU',
    'colab':{'name':'03_mvtec_leather_data_check.ipynb','provenance':[]},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.x'}
  },
  'nbformat':4,'nbformat_minor':5
}
NB.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(NB)
