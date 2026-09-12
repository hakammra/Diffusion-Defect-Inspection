"""Build the normal-only calibration notebook for the printed-label application."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "09_fabric_stain_evaluate_kaggle.ipynb"
DESTINATION = ROOT / "notebooks" / "11_printed_label_validate_kaggle.ipynb"


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
import csv, importlib.util, json, random, shutil, subprocess, sys, time, zipfile

import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT=Path('/kaggle/input')
WORK=Path('/kaggle/working/labelinspect')
WORK.mkdir(parents=True,exist_ok=True)
OUTPUT=WORK/'printed_label_validation'
OUTPUT.mkdir(parents=True,exist_ok=True)

def find_dataset(search_root):
    found=[]
    for path in search_root.rglob('printed_label_validation_v1'):
        if path.is_dir() and (path/'validation'/'normal').is_dir():
            found.append(path)
    return sorted(set(found))

dataset_candidates=find_dataset(KAGGLE_INPUT)
if not dataset_candidates:
    matching=[]
    for archive_path in KAGGLE_INPUT.rglob('*.zip'):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names=['/'+item.filename.replace('\\\\','/').lstrip('/') for item in archive.infolist()]
                if any('/printed_label_validation_v1/validation/normal/' in name for name in names):
                    matching.append(archive_path)
        except zipfile.BadZipFile:
            pass
    if len(matching)==1:
        extraction_root=WORK/'uploaded_validation'
        extraction_root.mkdir(parents=True,exist_ok=True)
        resolved=extraction_root.resolve()
        with zipfile.ZipFile(matching[0]) as archive:
            for item in archive.infolist():
                target=(extraction_root/item.filename).resolve()
                if target!=resolved and resolved not in target.parents:
                    raise ValueError(f'Unsafe ZIP member: {item.filename}')
            archive.extractall(extraction_root)
        dataset_candidates=find_dataset(extraction_root)
if len(dataset_candidates)!=1:
    raise FileNotFoundError('Expected one printed_label_validation_v1 dataset: '+repr([str(p) for p in dataset_candidates]))
DATA_ROOT=dataset_candidates[0]

all_pt_files=sorted(KAGGLE_INPUT.rglob('printed_label_latest.pt'))
if not all_pt_files:
    # Kaggle can unpack a torch.save checkpoint because it is internally a ZIP archive.
    extracted_roots=[]
    for data_pickle in KAGGLE_INPUT.rglob('data.pkl'):
        candidate=data_pickle.parent
        if (candidate/'data').is_dir() and (candidate/'version').is_file():
            extracted_roots.append(candidate)
    extracted_roots=sorted(set(extracted_roots))
    if len(extracted_roots)==1:
        archive_root=extracted_roots[0]
        rebuilt=WORK/'rebuilt_printed_label_checkpoint.pt'
        with zipfile.ZipFile(rebuilt,'w',compression=zipfile.ZIP_STORED) as archive:
            for source_file in sorted(archive_root.rglob('*')):
                if source_file.is_file():
                    archive.write(source_file,f'{archive_root.name}/{source_file.relative_to(archive_root).as_posix()}')
        all_pt_files=[rebuilt]
        print('Rebuilt Kaggle-extracted checkpoint:',rebuilt)
if len(all_pt_files)!=1:
    raise FileNotFoundError('Expected one printed-label checkpoint; found: '+repr([str(p) for p in all_pt_files]))
CHECKPOINT=all_pt_files[0]

SEED=230224
IMAGE_SIZE=224
BATCH_SIZE=4
T_DISTANCE=250
PIXEL_NORMAL_FPR=0.005
IMAGE_SCORE_QUANTILE=0.995
print('GPU:',torch.cuda.get_device_name(0))
print('Validation dataset:',DATA_ROOT)
print('Checkpoint:',CHECKPOINT)
""")

    data = code("""IMAGE_EXTENSIONS={'.jpg','.jpeg','.png'}
val_paths=sorted(
    path for path in (DATA_ROOT/'validation'/'normal').iterdir()
    if path.suffix.lower() in IMAGE_EXTENSIONS
)
assert len(val_paths)==10,f'Expected 10 validation normals, found {len(val_paths)}'
assert {path.stem.split('_')[0] for path in val_paths}=={'N09','N10'}
RESAMPLE=getattr(Image,'Resampling',Image).BILINEAR

def load_image(path):
    with Image.open(path) as source:
        image=ImageOps.contain(source.convert('RGB'),(IMAGE_SIZE,IMAGE_SIZE),RESAMPLE)
        canvas=Image.new('RGB',(IMAGE_SIZE,IMAGE_SIZE),(238,238,238))
        canvas.paste(image,((IMAGE_SIZE-image.width)//2,(IMAGE_SIZE-image.height)//2))
        array=np.asarray(canvas,dtype=np.float32).copy()/127.5-1.
    return torch.from_numpy(array).permute(2,0,1)

sample=load_image(val_paths[0])
assert sample.shape==(3,224,224) and torch.isfinite(sample).all()
# Registered 1063x650 labels occupy these rows after aspect-preserving resize.
label_roi=np.zeros((IMAGE_SIZE,IMAGE_SIZE),dtype=bool)
resized_height=round(650*IMAGE_SIZE/1063)
roi_top=(IMAGE_SIZE-resized_height)//2
label_roi[roi_top:roi_top+resized_height,:]=True
print('Validation normals:',len(val_paths),'ROI rows:',roi_top,'to',roi_top+resized_height-1)
""")

    model = code("""device=torch.device('cuda:0')
checkpoint=torch.load(CHECKPOINT,map_location=device,weights_only=False)
assert checkpoint['step']==2000,f"Expected step 2000, found {checkpoint['step']}"
assert checkpoint['author_commit']==author.COMMIT
assert checkpoint.get('dataset_category')=='printed_label_train_v1'
model_config=checkpoint['model_config']
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
backbone=model_module.UDHVT(**model_config).to(device);model=Adapter(backbone)
model.load_state_dict(checkpoint['model']);model.eval()
diffusion=diffusion_ns['GaussianDiffusionModel'](
    [224,224],diffusion_ns['get_beta_schedule'](1000,'cosine'),img_channels=3,
    loss_type='l2',noise='4dsimplex',octave=6,frequency=64,persistence=.9,train=False)
diffusion.noise_fn(torch.zeros(1,1,4,4,device=device),torch.tensor([5],device=device))
torch.cuda.reset_peak_memory_stats()
print('Loaded step',checkpoint['step'],'model with',sum(p.numel() for p in model.parameters()),'parameters.')
""")

    reconstruct = code("""def reconstruct_paths(paths,batch_size=BATCH_SIZE):
    reconstructions=[];inputs=[];seconds=[]
    for start in range(0,len(paths),batch_size):
        batch_paths=paths[start:start+batch_size]
        x=torch.stack([load_image(path) for path in batch_paths]).to(device)
        tick=time.perf_counter()
        with torch.inference_mode():
            recon=diffusion.forward_backward(model,x,None,see_whole_sequence=None,
                                              t_distance=T_DISTANCE,denoise_fn='noise_fn')
        torch.cuda.synchronize()
        seconds.append(time.perf_counter()-tick)
        inputs.append(x.cpu());reconstructions.append(recon.cpu())
        print(f'Reconstructed {min(start+len(batch_paths),len(paths))}/{len(paths)}',flush=True)
    return torch.cat(inputs),torch.cat(reconstructions),seconds

def residual_maps(inputs,reconstructions):
    return (inputs-reconstructions).square().mean(dim=1).numpy().astype(np.float32)

inputs,reconstructions,batch_seconds=reconstruct_paths(val_paths)
maps=residual_maps(inputs,reconstructions)
""")

    calibrate = code("""roi_values=maps[:,label_roi]
pixel_threshold=float(np.quantile(roi_values,1-PIXEL_NORMAL_FPR,method='higher'))
image_scores=np.quantile(roi_values,IMAGE_SCORE_QUANTILE,axis=1)
# Ten images are a small calibration set; max gives zero observed validation false positives with strict >.
image_threshold=float(np.max(image_scores))
predicted_masks=(maps>pixel_threshold)&label_roi[None]
rows=[]
for index,path in enumerate(val_paths):
    rows.append({
        'file':path.name,
        'image_score':float(image_scores[index]),
        'predicted_pixel_fraction':float(predicted_masks[index,label_roi].mean()),
    })
with (OUTPUT/'validation_per_image.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

calibration={
    'status':'passed',
    'source':'10 held-out normal photographs from physical labels N09 and N10',
    'checkpoint_step':int(checkpoint['step']),
    'author_commit':author.COMMIT,
    't_distance':T_DISTANCE,
    'pixel_threshold_rule':f'higher {(1-PIXEL_NORMAL_FPR):.3f} quantile of validation-normal ROI residual pixels',
    'pixel_threshold':pixel_threshold,
    'target_validation_normal_pixel_fpr':PIXEL_NORMAL_FPR,
    'observed_validation_normal_pixel_fpr':float(predicted_masks[:,label_roi].mean()),
    'image_score_rule':f'{IMAGE_SCORE_QUANTILE:.3f} quantile of ROI residual pixels',
    'image_threshold_rule':'maximum validation-normal image score',
    'image_threshold':image_threshold,
    'observed_validation_image_fpr':float(np.mean(image_scores>image_threshold)),
    'validation_count':len(val_paths),
    'mean_inference_seconds_per_image':float(sum(batch_seconds)/len(val_paths)),
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'test_images_used':0,
    'frozen_for_test':['checkpoint','registration','input transform','t_distance','pixel threshold','image score','image threshold'],
}
(OUTPUT/'calibration.json').write_text(json.dumps(calibration,indent=2))
print(json.dumps(calibration,indent=2))
""")

    preview = code("""def display_rgb(tensor):
    return ((tensor.permute(1,2,0).numpy()+1)/2).clip(0,1)

chosen=[0,4,9]
fig,axes=plt.subplots(len(chosen),4,figsize=(13,8))
for row,index in enumerate(chosen):
    axes[row,0].imshow(display_rgb(inputs[index]));axes[row,0].set_title(val_paths[index].name)
    axes[row,1].imshow(display_rgb(reconstructions[index]));axes[row,1].set_title('Reconstruction')
    axes[row,2].imshow(maps[index],cmap='magma');axes[row,2].set_title('Residual map')
    axes[row,3].imshow(predicted_masks[index],cmap='gray',vmin=0,vmax=1);axes[row,3].set_title('Normal FP mask')
    for axis in axes[row]: axis.axis('off')
fig.suptitle('Printed-label validation normals — calibration only')
fig.tight_layout();fig.savefig(OUTPUT/'validation_preview.png',dpi=160,bbox_inches='tight');plt.show()

archive=shutil.make_archive('/kaggle/working/printed_label_validation_results','zip',OUTPUT)
print('\\nPRINTED-LABEL VALIDATION PASSED')
print('Download:',archive)
print('Keep the checkpoint unchanged for the locked test.')
""")

    template["cells"] = [
        markdown("""# Printed-label normal validation and threshold calibration

This notebook loads the frozen 2,000-step printed-label checkpoint and uses only the
ten registered normal photographs from physical labels `N09` and `N10`. It fixes the
partial-diffusion distance and calibrates pixel- and image-level anomaly thresholds.
It does not load test images and does not report detection accuracy.

Attach `printed_label_validation_v1.zip` and `printed_label_latest.pt`, select a Kaggle
GPU, enable Internet, and run all cells. Download the validation-results ZIP at the end.
"""),
        setup,
        install_cell,
        embedded_author_cell,
        load_author_cell,
        data,
        model,
        reconstruct,
        calibrate,
        preview,
    ]
    DESTINATION.write_text(json.dumps(template, indent=1), encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
