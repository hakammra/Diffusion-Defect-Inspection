"""Build the Kaggle notebook for the real printed-label training run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "08_fabric_stain_train_kaggle.ipynb"
DESTINATION = ROOT / "notebooks" / "10_printed_label_train_kaggle.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    install_cell = template["cells"][2]
    embedded_author_cell = template["cells"][3]
    load_author_cell = template["cells"][4]

    setup = code("""from pathlib import Path
import importlib.util, json, os, random, shutil, subprocess, sys, time, zipfile

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working/labelinspect')
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT = WORK/'printed_label_training'
OUTPUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = WORK/'checkpoints'/'printed_label'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def find_dataset(search_root):
    candidates=[]
    for path in search_root.rglob('printed_label_train_v1'):
        if path.is_dir() and (path/'train'/'normal').is_dir():
            candidates.append(path)
    return sorted(set(candidates))

dataset_candidates=find_dataset(KAGGLE_INPUT)
if not dataset_candidates:
    matching=[]
    for archive_path in KAGGLE_INPUT.rglob('*.zip'):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names=['/'+item.filename.replace('\\\\','/').lstrip('/') for item in archive.infolist()]
                if any('/printed_label_train_v1/train/normal/' in name for name in names):
                    matching.append(archive_path)
        except zipfile.BadZipFile:
            pass
    if len(matching)==1:
        extraction_root=WORK/'uploaded_data'
        extraction_root.mkdir(parents=True,exist_ok=True)
        resolved=extraction_root.resolve()
        with zipfile.ZipFile(matching[0]) as archive:
            for item in archive.infolist():
                target=(extraction_root/item.filename).resolve()
                if target != resolved and resolved not in target.parents:
                    raise ValueError(f'Unsafe ZIP member: {item.filename}')
            archive.extractall(extraction_root)
        dataset_candidates=find_dataset(extraction_root)
        print('Extracted:',matching[0])
if len(dataset_candidates)!=1:
    raise FileNotFoundError('Expected one printed_label_train_v1 dataset, found: '+repr([str(p) for p in dataset_candidates]))
DATA_ROOT=dataset_candidates[0]

SEED = 230224
TARGET_STEPS = 2000
BATCH_SIZE = 2
SAVE_EVERY = 250
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
print('GPU:', torch.cuda.get_device_name(0))
print('Dataset:', DATA_ROOT)
print('Working output:', WORK)
""")

    data = code("""IMAGE_EXTENSIONS={'.jpg','.jpeg','.png'}
train_paths=sorted(
    path for path in (DATA_ROOT/'train'/'normal').iterdir()
    if path.suffix.lower() in IMAGE_EXTENSIONS
)
assert len(train_paths)==40, f'Expected 40 registered normal images, found {len(train_paths)}'
physical_ids={path.stem.split('_')[0] for path in train_paths}
view_ids={path.stem.split('_')[1] for path in train_paths}
assert physical_ids=={f'N{index:02d}' for index in range(1,9)}, physical_ids
assert view_ids=={f'B{index}' for index in range(1,6)}, view_ids
print('Training normals:',len(train_paths))
print('Physical labels:',sorted(physical_ids))
print('Capture groups:',sorted(view_ids))

RESAMPLE=getattr(Image,'Resampling',Image).BILINEAR

def pad_to_square(image,size=IMAGE_SIZE):
    image=ImageOps.contain(image,(size,size),RESAMPLE)
    canvas=Image.new('RGB',(size,size),(238,238,238))
    canvas.paste(image,((size-image.width)//2,(size-image.height)//2))
    return canvas

def load_training_image(path,rng):
    with Image.open(path) as source:
        image=source.convert('RGB')
        # Mild deterministic jitter represents residual registration and lighting variation.
        image=ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.90,1.10)))
        image=ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.92,1.08)))
        angle=float(rng.uniform(-1.0,1.0))
        image=image.rotate(angle,resample=Image.Resampling.BICUBIC,expand=False,fillcolor=(238,238,238))
        image=pad_to_square(image)
        array=np.asarray(image,dtype=np.float32).copy()/127.5-1.0
    return torch.from_numpy(array).permute(2,0,1)

def deterministic_batch(step):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,step]))
    indices=rng.choice(len(train_paths),size=BATCH_SIZE,replace=False)
    return torch.stack([load_training_image(train_paths[int(i)],rng) for i in indices])

def deterministic_times(step):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,step,1]))
    values=rng.integers(1,1000,size=BATCH_SIZE,dtype=np.int64)
    if len(set(values.tolist()))!=len(values):
        values[1]=(values[0]%999)+1
    return torch.from_numpy(values)

sample=deterministic_batch(0)
assert sample.shape==(BATCH_SIZE,3,IMAGE_SIZE,IMAGE_SIZE)
assert torch.isfinite(sample).all() and sample.min()>=-1 and sample.max()<=1
print('Training batch check:',tuple(sample.shape),float(sample.min()),float(sample.max()))
""")

    model = code("""random.seed(SEED)
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

latest = CHECKPOINT_DIR/'printed_label_latest.pt'
attached=list(KAGGLE_INPUT.rglob('printed_label_latest.pt'))
if len(attached)>1:
    raise ValueError('Attach at most one printed_label_latest.pt checkpoint.')
if len(attached)==1:
    shutil.copy2(attached[0],latest)

start_step=0
history=[]
if latest.is_file():
    checkpoint=torch.load(latest,map_location=device,weights_only=False)
    if checkpoint.get('author_commit')!=author.COMMIT:
        raise ValueError('Checkpoint author commit does not match this notebook.')
    if checkpoint.get('dataset_category')!='printed_label_train_v1':
        raise ValueError('This is not a printed-label checkpoint.')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step=int(checkpoint['step'])
    history=list(checkpoint.get('history',[]))
    print('Resuming from step',start_step)
else:
    print('Starting a new printed-label model from step 0.')
if start_step>TARGET_STEPS:
    raise ValueError(f'Checkpoint is at step {start_step}; set TARGET_STEPS to at least that value.')

probe=torch.zeros(1,1,4,4,device=device)
diffusion.noise_fn(probe,torch.tensor([5],device=device))
torch.cuda.reset_peak_memory_stats()
""")

    training = code("""def save_checkpoint(step):
    payload={
        'step':step,
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'history':history,
        'author_commit':author.COMMIT,
        'source_sha256':hashes,
        'model_config':model_config,
        'noise_parameters':{'octave':6,'frequency':64,'persistence':0.9},
        'seed':SEED,
        'dataset_category':'printed_label_train_v1',
        'protocol':'40 registered normal photos from eight physical labels; no test images used',
        'preprocessing':'four-fiducial perspective registration; aspect-preserving resize and padding',
    }
    temporary=CHECKPOINT_DIR/'printed_label_latest.tmp.pt'
    torch.save(payload,temporary)
    temporary.replace(latest)

model.train()
run_started=time.perf_counter()
step_times=[]
for step in range(start_step,TARGET_STEPS):
    step_started=time.perf_counter()
    x=deterministic_batch(step).to(device,non_blocking=True)
    t=deterministic_times(step).to(device)
    optimizer.zero_grad(set_to_none=True)
    losses,noisy,predicted=diffusion.calc_loss(model,x,None,t)
    loss=losses['loss'].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite loss at step {step+1}: {loss}')
    loss.backward()
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f'Non-finite gradient norm at step {step+1}')
    optimizer.step()
    elapsed=time.perf_counter()-step_started
    step_times.append(elapsed)
    history.append({'step':step+1,'loss':float(loss.detach()),'grad_norm':float(grad_norm),'seconds':elapsed})
    if (step+1)%10==0 or step==start_step:
        recent=np.mean([item['loss'] for item in history[-10:]])
        print(f'Step {step+1:4d}/{TARGET_STEPS} | loss {float(loss):.6f} | recent mean {recent:.6f} | {elapsed:.2f}s')
    if (step+1)%SAVE_EVERY==0 or step+1==TARGET_STEPS:
        save_checkpoint(step+1)

run_seconds=time.perf_counter()-run_started
print('Checkpoint:',latest)
print('Steps completed this run:',TARGET_STEPS-start_step)
""")

    results = code("""import csv
import matplotlib.pyplot as plt

with (OUTPUT/'training_history.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['step','loss','grad_norm','seconds'])
    writer.writeheader(); writer.writerows(history)

loss_values=[item['loss'] for item in history]
window=min(20,len(loss_values))
smoothed=np.convolve(loss_values,np.ones(window)/window,mode='valid') if window else []
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(range(1,len(loss_values)+1),loss_values,alpha=.35,label='step loss')
if len(smoothed):
    ax.plot(range(window,len(loss_values)+1),smoothed,linewidth=2,label=f'{window}-step mean')
ax.set(xlabel='Optimizer step',ylabel='L2 noise-prediction loss',title='Printed-label normal-only training')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(OUTPUT/'loss_curve.png',dpi=160)
plt.show()

report={
    'status':'passed',
    'scope':'printed_label_training_only',
    'paper_result_reproduced':False,
    'dataset':'LabelInspect real printed labels v1',
    'train_normal_count':len(train_paths),
    'train_physical_label_count':len(physical_ids),
    'validation_images_used':0,
    'test_images_used':0,
    'author_commit':author.COMMIT,
    'model_parameters':sum(p.numel() for p in model.parameters()),
    'target_steps':TARGET_STEPS,
    'start_step':start_step,
    'steps_completed_this_run':TARGET_STEPS-start_step,
    'initial_loss':history[0]['loss'] if history else None,
    'final_loss':history[-1]['loss'] if history else None,
    'last_20_mean_loss':float(np.mean(loss_values[-20:])) if loss_values else None,
    'median_step_seconds_this_run':float(np.median(step_times)) if step_times else None,
    'run_seconds':run_seconds,
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'gpu':torch.cuda.get_device_name(0),
    'checkpoint':str(latest),
    'next_decision':'Collect N09-N12 and defective labels, register them, calibrate using validation normals, then run one locked test.',
}
(OUTPUT/'training_report.json').write_text(json.dumps(report,indent=2))
shutil.copy2(latest,Path('/kaggle/working/printed_label_latest.pt'))
archive=shutil.make_archive('/kaggle/working/printed_label_training_results','zip',OUTPUT)
print(json.dumps(report,indent=2))
print('\\nPRINTED-LABEL TRAINING PASSED')
print('Download checkpoint: /kaggle/working/printed_label_latest.pt')
print('Download result bundle:',archive)
""")

    template["cells"] = [
        markdown("""# Real printed-label DTU-Net/Tsimplex training

This notebook trains the paper authors' DTU-Net/Tsimplex path for **2,000 optimizer
steps using 40 registered normal photographs from eight physical labels**. It is the
training stage of the proposed printed-label application. It does not evaluate defects
and does not claim that the paper's numerical results were reproduced.

Attach `printed_label_train_v1.zip`, select a Kaggle GPU, enable Internet for the
pinned author source download, and run all cells. Download both output files at the end.
"""),
        setup,
        install_cell,
        embedded_author_cell,
        load_author_cell,
        data,
        model,
        training,
        results,
    ]
    DESTINATION.write_text(json.dumps(template, indent=1), encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
