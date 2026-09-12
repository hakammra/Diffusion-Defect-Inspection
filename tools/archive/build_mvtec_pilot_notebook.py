from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r'C:\Users\abdul\Documents\LabelInspect')
source_nb = json.loads((ROOT/'notebooks'/'02_author_model_check.ipynb').read_text(encoding='utf-8'))
author_writer_cell = ''.join(source_nb['cells'][2]['source'])
author_writer_cell = author_writer_cell.replace("script = work / 'author_smoke.py'", "script = WORK / 'author_smoke.py'")


def lines(text):
    return text.splitlines(keepends=True) or ['']


def md(text):
    return {'cell_type':'markdown','metadata':{},'source':lines(text.strip()+'\n')}


def code(text):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':lines(text.strip()+'\n')}

cells = [
md('''# Stage 3B — MVTec leather DTU-Net/Tsimplex training pilot

This is a **reduced reproduction pilot**, based on Kumar et al., WACV 2025. It trains the authors’ DTU-Net/Tsimplex path on normal MVTec AD leather images for 100 optimizer steps and saves a resumable checkpoint.

The purpose of this run is to measure stable loss, speed, and memory on the available Tesla T4 before selecting a defensible longer schedule. It is not a reproduced paper result and does not evaluate the official test set.

Dataset policy:
- Training uses only the deterministic `train/good` training partition.
- The 49-image normal validation partition and all official test images remain unused in this pilot.
- Dataset: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), CC BY-NC-SA 4.0.
- Method/code: [Kumar et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Kumar_Self-Supervised_Anomaly_Segmentation_via_Diffusion_Models_with_Dynamic_Transformer_UNet_WACV_2025_paper.html).
'''),
code(r'''from pathlib import Path
import importlib.util, json, os, random, shutil, subprocess, sys, time

import numpy as np
import torch
from PIL import Image

assert torch.cuda.is_available(), 'Select a T4 GPU runtime first.'

# Mount Drive so checkpoints survive a Colab disconnection.
from google.colab import drive
drive.mount('/content/drive')

DATA_ROOT = Path('/content/drive/MyDrive/mvtec_anomaly_detection')
CHECKPOINT_DIR = Path('/content/drive/MyDrive/LabelInspect/checkpoints/mvtec_leather')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

WORK = Path('/content/labelinspect')
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT = WORK/'mvtec_leather_pilot'
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 230224
TARGET_STEPS = 100       # Increase later; this first run is a timing pilot.
BATCH_SIZE = 2
SAVE_EVERY = 25
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
print('GPU:', torch.cuda.get_device_name(0))
print('Dataset:', DATA_ROOT)
print('Persistent checkpoints:', CHECKPOINT_DIR)
'''),
code(r'''# Install only missing runtime packages. Do not install the upstream requirements file.
missing = []
for package, module in [('timm','timm'), ('einops','einops'), ('numba','numba')]:
    if importlib.util.find_spec(module) is None:
        missing.append(package)
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
print('Dependencies ready.')
'''),
code(author_writer_cell),
code(r'''# Load the same pinned and hash-verified author components that passed Stage 2.
spec = importlib.util.spec_from_file_location('labelinspect_author_smoke', WORK/'author_smoke.py')
author = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = author
spec.loader.exec_module(author)

import numba
numba.set_num_threads(min(2, numba.get_num_threads()))
source_cache = WORK/'upstream'/author.COMMIT
hashes = author.fetch_sources(source_cache)
model_module, diffusion_ns, Adapter, compatibility = author.load_author_components(source_cache, OUTPUT)
print('Pinned author commit:', author.COMMIT)
'''),
code(r'''def locate_category(root: Path, category='leather') -> Path:
    candidates = [root/category, root/'mvtec_anomaly_detection'/category]
    if root.name == category:
        candidates.insert(0, root)
    for candidate in candidates:
        if (candidate/'train'/'good').is_dir():
            return candidate
    raise FileNotFoundError(f'Could not find leather/train/good below {root}. Edit DATA_ROOT.')

category_root = locate_category(DATA_ROOT)
all_normal = sorted((category_root/'train'/'good').glob('*.png'))
assert len(all_normal) == 245, f'Expected 245 leather train/good images, found {len(all_normal)}.'

# Reproduce the already-audited 196/49 split exactly.
split_rng = random.Random(SEED)
shuffled = all_normal.copy()
split_rng.shuffle(shuffled)
val_count = max(1, round(len(shuffled)*0.20))
val_paths = sorted(shuffled[:val_count])
train_paths = sorted(shuffled[val_count:])
assert len(train_paths) == 196 and len(val_paths) == 49
assert set(train_paths).isdisjoint(val_paths)
print('Training normals:', len(train_paths), '| held-out validation normals:', len(val_paths))

RESAMPLE = getattr(Image, 'Resampling', Image).BILINEAR

def load_image(path: Path):
    with Image.open(path) as image:
        image = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE)
        array = np.asarray(image, dtype=np.float32).copy()/127.5 - 1.0
    return torch.from_numpy(array).permute(2,0,1)

def deterministic_batch(step: int):
    # Step-based sampling makes resumed runs reproduce the same image sequence.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, step]))
    indices = rng.choice(len(train_paths), size=BATCH_SIZE, replace=False)
    return torch.stack([load_image(train_paths[int(i)]) for i in indices])

def deterministic_times(step: int):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, step, 1]))
    values = rng.integers(1, 1000, size=BATCH_SIZE, dtype=np.int64)
    if len(set(values.tolist())) != len(values):
        values[1] = (values[0] % 999) + 1
    return torch.from_numpy(values)
'''),
code(r'''random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda:0')
model_config = {
    'img_size':224, 'patch_size':16, 'in_chans':3, 'embed_dim':384,
    'depth':12, 'num_heads':6, 'mlp_ratio':4., 'num_classes':None,
    'mlp_time_embed':True, 'use_dec':['DAFF','DAFF','DAFF'],
    'PE_type':'SPE', 'refinement':True, 'qkv_bias':False,
}
backbone = model_module.UDHVT(**model_config).to(device)
model = Adapter(backbone)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.)
diffusion = diffusion_ns['GaussianDiffusionModel'](
    [224,224], diffusion_ns['get_beta_schedule'](1000,'cosine'),
    img_channels=3, loss_type='l2', noise='4dsimplex',
    octave=6, frequency=64, persistence=.9, train=False,
)

latest = CHECKPOINT_DIR/'latest.pt'
start_step = 0
history = []
if latest.is_file():
    checkpoint = torch.load(latest, map_location=device, weights_only=False)
    if checkpoint.get('author_commit') != author.COMMIT:
        raise ValueError('Checkpoint author commit does not match this notebook.')
    if checkpoint.get('model_config') != model_config:
        raise ValueError('Checkpoint model configuration does not match this notebook.')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = int(checkpoint['step'])
    history = list(checkpoint.get('history', []))
    print('Resuming from step', start_step)
else:
    print('Starting a new pilot.')
if start_step > TARGET_STEPS:
    raise ValueError(f'Checkpoint is already at step {start_step}; set TARGET_STEPS to at least that value.')

# Compile Tsimplex before measuring the training loop.
probe = torch.zeros(1,1,4,4, device=device)
diffusion.noise_fn(probe, torch.tensor([5], device=device))
torch.cuda.reset_peak_memory_stats()
'''),
code(r'''def save_checkpoint(step):
    payload = {
        'step':step,
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'history':history,
        'author_commit':author.COMMIT,
        'source_sha256':hashes,
        'model_config':model_config,
        'noise_parameters':{'octave':6,'frequency':64,'persistence':0.9},
        'seed':SEED,
        'dataset_category':'leather',
        'protocol':'196 deterministic train normals; 49 held-out validation normals',
    }
    temporary = CHECKPOINT_DIR/'latest.tmp.pt'
    torch.save(payload, temporary)
    temporary.replace(latest)

model.train()
run_started = time.perf_counter()
step_times = []
for step in range(start_step, TARGET_STEPS):
    step_started = time.perf_counter()
    x = deterministic_batch(step).to(device, non_blocking=True)
    t = deterministic_times(step).to(device)
    optimizer.zero_grad(set_to_none=True)
    losses, noisy, predicted = diffusion.calc_loss(model, x, None, t)
    loss = losses['loss'].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite loss at step {step+1}: {loss}')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f'Non-finite gradient norm at step {step+1}')
    optimizer.step()
    elapsed = time.perf_counter()-step_started
    step_times.append(elapsed)
    history.append({'step':step+1, 'loss':float(loss.detach()),
                    'grad_norm':float(grad_norm), 'seconds':elapsed})
    if (step+1) % 10 == 0 or step == start_step:
        recent = np.mean([item['loss'] for item in history[-10:]])
        print(f"Step {step+1:4d}/{TARGET_STEPS} | loss {float(loss):.6f} | recent mean {recent:.6f} | {elapsed:.2f}s")
    if (step+1) % SAVE_EVERY == 0 or step+1 == TARGET_STEPS:
        save_checkpoint(step+1)

run_seconds = time.perf_counter()-run_started
print('Checkpoint:', latest)
print('Steps completed this run:', TARGET_STEPS-start_step)
'''),
code(r'''import csv
import matplotlib.pyplot as plt

with (OUTPUT/'training_history.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['step','loss','grad_norm','seconds'])
    writer.writeheader(); writer.writerows(history)

loss_values = [item['loss'] for item in history]
window = min(20, len(loss_values))
smoothed = np.convolve(loss_values, np.ones(window)/window, mode='valid') if window else []
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(range(1,len(loss_values)+1), loss_values, alpha=.35, label='step loss')
if len(smoothed):
    ax.plot(range(window,len(loss_values)+1), smoothed, linewidth=2, label=f'{window}-step mean')
ax.set(xlabel='Optimizer step', ylabel='L2 noise-prediction loss', title='Reduced MVTec leather pilot')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(OUTPUT/'loss_curve.png', dpi=160)
plt.show()

completed_this_run = TARGET_STEPS-start_step
report = {
    'status':'passed',
    'scope':'reduced_training_pilot_only',
    'paper_result_reproduced':False,
    'dataset':'MVTec AD', 'category':'leather',
    'train_normal_count':len(train_paths), 'held_out_val_normal_count':len(val_paths),
    'test_images_used':0,
    'author_commit':author.COMMIT,
    'model_parameters':sum(p.numel() for p in model.parameters()),
    'target_steps':TARGET_STEPS, 'start_step':start_step,
    'steps_completed_this_run':completed_this_run,
    'initial_loss':history[0]['loss'] if history else None,
    'final_loss':history[-1]['loss'] if history else None,
    'last_20_mean_loss':float(np.mean(loss_values[-20:])) if loss_values else None,
    'median_step_seconds_this_run':float(np.median(step_times)) if step_times else None,
    'run_seconds':run_seconds,
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'gpu':torch.cuda.get_device_name(0),
    'checkpoint':str(latest),
    'next_decision':'Choose longer target step count from measured throughput, then validate reconstruction on held-out normal images.',
}
(OUTPUT/'pilot_report.json').write_text(json.dumps(report, indent=2))
archive = shutil.make_archive(str(WORK/'mvtec_leather_pilot_results'), 'zip', OUTPUT)
print(json.dumps(report, indent=2))
print('\nTRAINING PILOT PASSED')
print('Download this small result bundle:', archive)
print('The persistent model checkpoint remains in Google Drive:', latest)
''')]

notebook = {
    'cells':cells,
    'metadata':{
        'accelerator':'GPU',
        'colab':{'name':'04_mvtec_leather_pilot_train.ipynb','provenance':[]},
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.x'}
    },
    'nbformat':4,'nbformat_minor':5
}
out = ROOT/'notebooks'/'04_mvtec_leather_pilot_train.ipynb'
out.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(out)
