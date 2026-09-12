"""Build the pre-test diffusion-distance selection notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "11_printed_label_validate_kaggle.ipynb"
DESTINATION = ROOT / "notebooks" / "12_printed_label_model_selection_kaggle.ipynb"


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
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT=Path('/kaggle/input')
WORK=Path('/kaggle/working/labelinspect')
WORK.mkdir(parents=True,exist_ok=True)
OUTPUT=WORK/'printed_label_model_selection'
OUTPUT.mkdir(parents=True,exist_ok=True)

def find_dataset(search_root):
    found=[]
    for path in search_root.rglob('printed_label_validation_v1'):
        if path.is_dir() and (path/'validation'/'normal').is_dir(): found.append(path)
    return sorted(set(found))

dataset_candidates=find_dataset(KAGGLE_INPUT)
if not dataset_candidates:
    matching=[]
    for archive_path in KAGGLE_INPUT.rglob('*.zip'):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names=['/'+item.filename.replace('\\\\','/').lstrip('/') for item in archive.infolist()]
                if any('/printed_label_validation_v1/validation/normal/' in name for name in names): matching.append(archive_path)
        except zipfile.BadZipFile: pass
    if len(matching)==1:
        extraction_root=WORK/'uploaded_validation';extraction_root.mkdir(parents=True,exist_ok=True)
        resolved=extraction_root.resolve()
        with zipfile.ZipFile(matching[0]) as archive:
            for item in archive.infolist():
                target=(extraction_root/item.filename).resolve()
                if target!=resolved and resolved not in target.parents: raise ValueError(f'Unsafe ZIP member: {item.filename}')
            archive.extractall(extraction_root)
        dataset_candidates=find_dataset(extraction_root)
if len(dataset_candidates)!=1:
    raise FileNotFoundError('Expected one printed_label_validation_v1 dataset: '+repr([str(p) for p in dataset_candidates]))
DATA_ROOT=dataset_candidates[0]

all_pt_files=sorted(KAGGLE_INPUT.rglob('printed_label_latest.pt'))
if not all_pt_files:
    extracted_roots=[]
    for data_pickle in KAGGLE_INPUT.rglob('data.pkl'):
        candidate=data_pickle.parent
        if (candidate/'data').is_dir() and (candidate/'version').is_file(): extracted_roots.append(candidate)
    extracted_roots=sorted(set(extracted_roots))
    if len(extracted_roots)==1:
        archive_root=extracted_roots[0]
        rebuilt=WORK/'rebuilt_printed_label_checkpoint.pt'
        with zipfile.ZipFile(rebuilt,'w',compression=zipfile.ZIP_STORED) as archive:
            for source_file in sorted(archive_root.rglob('*')):
                if source_file.is_file(): archive.write(source_file,f'{archive_root.name}/{source_file.relative_to(archive_root).as_posix()}')
        all_pt_files=[rebuilt]
        print('Rebuilt Kaggle-extracted checkpoint:',rebuilt)
if len(all_pt_files)!=1:
    raise FileNotFoundError('Expected one printed-label checkpoint; found: '+repr([str(p) for p in all_pt_files]))
CHECKPOINT=all_pt_files[0]

SEED=230224
IMAGE_SIZE=224
BATCH_SIZE=4
DISTANCES=[50,100,250]
PIXEL_NORMAL_FPR=0.005
IMAGE_SCORE_QUANTILE=0.995
print('GPU:',torch.cuda.get_device_name(0))
print('Validation dataset:',DATA_ROOT)
print('Checkpoint:',CHECKPOINT)
print('Distances:',DISTANCES)
""")

    data = code("""IMAGE_EXTENSIONS={'.jpg','.jpeg','.png'}
val_paths=sorted(path for path in (DATA_ROOT/'validation'/'normal').iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
assert len(val_paths)==10
assert {path.stem.split('_')[0] for path in val_paths}=={'N09','N10'}
RESAMPLE=getattr(Image,'Resampling',Image).BILINEAR
NEAREST=getattr(Image,'Resampling',Image).NEAREST

def pad_image(image,resample=RESAMPLE,fill=(238,238,238)):
    fitted=ImageOps.contain(image,(IMAGE_SIZE,IMAGE_SIZE),resample)
    canvas=Image.new(image.mode,(IMAGE_SIZE,IMAGE_SIZE),fill)
    canvas.paste(fitted,((IMAGE_SIZE-fitted.width)//2,(IMAGE_SIZE-fitted.height)//2))
    return canvas

def tensor_from_image(image):
    array=np.asarray(pad_image(image.convert('RGB')),dtype=np.float32).copy()/127.5-1.
    return torch.from_numpy(array).permute(2,0,1)

def synthetic_defect(source,kind,index):
    image=source.convert('RGB')
    changed=image.copy()
    width,height=changed.size
    if kind=='missing_print':
        # Remove a different narrow barcode interval for each source image.
        x0=int(width*(0.25+0.035*(index%8))); x1=x0+int(width*0.055)
        y0=int(height*0.60); y1=int(height*0.82)
        ImageDraw.Draw(changed).rectangle((x0,y0,x1,y1),fill=(239,239,239))
    elif kind=='smudge':
        overlay=Image.new('RGBA',changed.size,(0,0,0,0))
        draw=ImageDraw.Draw(overlay)
        x0=int(width*(0.52+0.02*(index%4))); y0=int(height*(0.28+0.025*(index%3)))
        draw.ellipse((x0,y0,x0+int(width*.12),y0+int(height*.12)),fill=(35,35,35,165))
        changed=Image.alpha_composite(changed.convert('RGBA'),overlay).convert('RGB')
    elif kind=='tear':
        y=int(height*(0.34+0.035*(index%6)))
        ImageDraw.Draw(changed).polygon([
            (width-1,y-int(height*.10)),(width-1,y+int(height*.13)),
            (int(width*.86),y+int(height*.035)),(int(width*.92),y-int(height*.03)),
        ],fill=(205,205,205))
    else: raise ValueError(kind)
    before=np.asarray(image,dtype=np.int16)
    after=np.asarray(changed,dtype=np.int16)
    mask=Image.fromarray((np.max(np.abs(after-before),axis=2)>8).astype(np.uint8)*255)
    mask_array=np.asarray(pad_image(mask,resample=NEAREST,fill=0))>0
    return tensor_from_image(changed),mask_array

normal_tensors=[]
synthetic_tensors=[]
synthetic_masks=[]
synthetic_kinds=[]
synthetic_names=[]
for index,path in enumerate(val_paths):
    with Image.open(path) as opened: source=opened.convert('RGB')
    normal_tensors.append(tensor_from_image(source))
    for kind in ['missing_print','smudge','tear']:
        tensor,mask=synthetic_defect(source,kind,index)
        assert mask.any(),(path,kind)
        synthetic_tensors.append(tensor);synthetic_masks.append(mask)
        synthetic_kinds.append(kind);synthetic_names.append(path.stem+'__'+kind)
normal_tensors=torch.stack(normal_tensors)
synthetic_tensors=torch.stack(synthetic_tensors)
synthetic_masks=np.stack(synthetic_masks)

label_roi=np.zeros((IMAGE_SIZE,IMAGE_SIZE),dtype=bool)
resized_height=round(650*IMAGE_SIZE/1063);roi_top=(IMAGE_SIZE-resized_height)//2
label_roi[roi_top:roi_top+resized_height,:]=True
assert normal_tensors.shape==(10,3,224,224) and synthetic_tensors.shape==(30,3,224,224)
print('Prepared',len(normal_tensors),'real normals and',len(synthetic_tensors),'synthetic validation defects.')
""")

    model = code("""device=torch.device('cuda:0')
checkpoint=torch.load(CHECKPOINT,map_location=device,weights_only=False)
assert checkpoint['step']==2000
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
print('Loaded frozen step',checkpoint['step'],'checkpoint.')
""")

    evaluate = code("""def reconstruct(tensors,distance):
    outputs=[];elapsed=0.
    for start in range(0,len(tensors),BATCH_SIZE):
        x=tensors[start:start+BATCH_SIZE].to(device)
        tick=time.perf_counter()
        with torch.inference_mode():
            result=diffusion.forward_backward(model,x,None,see_whole_sequence=None,
                                               t_distance=distance,denoise_fn='noise_fn')
        torch.cuda.synchronize();elapsed+=time.perf_counter()-tick
        outputs.append(result.cpu())
    return torch.cat(outputs),elapsed

def residual(inputs,recons):
    return (inputs-recons).square().mean(dim=1).numpy().astype(np.float32)

def mask_metrics(pred,truth):
    pred=np.asarray(pred,dtype=bool);truth=np.asarray(truth,dtype=bool)
    tp=int(np.sum(pred&truth));fp=int(np.sum(pred&~truth));fn=int(np.sum(~pred&truth))
    return {
        'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
    }

all_results=[];detail_rows=[];cached={}
for distance in DISTANCES:
    # Reset seeds before each candidate so this comparison is reproducible.
    random.seed(SEED+distance);np.random.seed(SEED+distance);torch.manual_seed(SEED+distance);torch.cuda.manual_seed_all(SEED+distance)
    normal_recon,normal_seconds=reconstruct(normal_tensors,distance)
    defect_recon,defect_seconds=reconstruct(synthetic_tensors,distance)
    normal_maps=residual(normal_tensors,normal_recon)
    defect_maps=residual(synthetic_tensors,defect_recon)
    pixel_threshold=float(np.quantile(normal_maps[:,label_roi],1-PIXEL_NORMAL_FPR,method='higher'))
    normal_image_scores=np.quantile(normal_maps[:,label_roi],IMAGE_SCORE_QUANTILE,axis=1)
    image_threshold=float(np.max(normal_image_scores))
    defect_image_scores=np.quantile(defect_maps[:,label_roi],IMAGE_SCORE_QUANTILE,axis=1)
    predicted=(defect_maps>pixel_threshold)&label_roi[None]
    per_defect=[]
    for index,(name,kind) in enumerate(zip(synthetic_names,synthetic_kinds)):
        metrics=mask_metrics(predicted[index],synthetic_masks[index])
        detected=bool(defect_image_scores[index]>image_threshold)
        row={'distance':distance,'file':name,'kind':kind,'image_score':float(defect_image_scores[index]),
             'detected':detected,**metrics}
        detail_rows.append(row);per_defect.append(row)
    result={
        'distance':distance,
        'pixel_threshold':pixel_threshold,
        'image_threshold':image_threshold,
        'normal_pixel_fpr':float(((normal_maps>pixel_threshold)&label_roi[None])[:,label_roi].mean()),
        'normal_image_fpr':float(np.mean(normal_image_scores>image_threshold)),
        'synthetic_mean_dice':float(np.mean([row['dice'] for row in per_defect])),
        'synthetic_mean_iou':float(np.mean([row['iou'] for row in per_defect])),
        'synthetic_image_sensitivity':float(np.mean([row['detected'] for row in per_defect])),
        'mean_seconds_per_image':float((normal_seconds+defect_seconds)/(len(normal_tensors)+len(synthetic_tensors))),
        'by_kind':{},
    }
    for kind in ['missing_print','smudge','tear']:
        subset=[row for row in per_defect if row['kind']==kind]
        result['by_kind'][kind]={
            'mean_dice':float(np.mean([row['dice'] for row in subset])),
            'image_sensitivity':float(np.mean([row['detected'] for row in subset])),
        }
    all_results.append(result)
    cached[distance]=(normal_recon,defect_recon,normal_maps,defect_maps,predicted)
    print(json.dumps(result,indent=2))

selected=max(all_results,key=lambda item:(item['synthetic_mean_dice'],item['synthetic_image_sensitivity'],-item['distance']))
selection={
    'status':'passed',
    'scope':'validation_only_model_selection',
    'real_test_images_used':0,
    'checkpoint_step':int(checkpoint['step']),
    'author_commit':author.COMMIT,
    'candidate_distances':DISTANCES,
    'selection_rule':'highest mean synthetic-defect Dice; then sensitivity; then shorter distance',
    'selected':selected,
    'all_candidates':all_results,
    'synthetic_validation_warning':'Digitally generated validation defects support parameter selection but do not estimate real-defect accuracy.',
    'frozen_for_real_test':['checkpoint','registration','input transform','selected distance','pixel threshold','image score','image threshold'],
}
(OUTPUT/'model_selection.json').write_text(json.dumps(selection,indent=2))
with (OUTPUT/'synthetic_validation_per_image.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(detail_rows[0]));writer.writeheader();writer.writerows(detail_rows)
print('SELECTED CONFIGURATION')
print(json.dumps(selected,indent=2))
""")

    preview = code("""distance=selected['distance']
normal_recon,defect_recon,normal_maps,defect_maps,predicted=cached[distance]
def show_tensor(tensor): return ((tensor.permute(1,2,0).numpy()+1)/2).clip(0,1)

indices=[0,1,2]
fig,axes=plt.subplots(3,5,figsize=(15,9))
for row,index in enumerate(indices):
    axes[row,0].imshow(show_tensor(synthetic_tensors[index]));axes[row,0].set_title(synthetic_kinds[index])
    axes[row,1].imshow(show_tensor(defect_recon[index]));axes[row,1].set_title('Reconstruction')
    axes[row,2].imshow(defect_maps[index],cmap='magma');axes[row,2].set_title('Residual')
    axes[row,3].imshow(predicted[index],cmap='gray',vmin=0,vmax=1);axes[row,3].set_title('Prediction')
    axes[row,4].imshow(synthetic_masks[index],cmap='gray',vmin=0,vmax=1);axes[row,4].set_title('Known synthetic mask')
    for axis in axes[row]: axis.axis('off')
fig.suptitle(f'Validation-only distance selection — selected t={distance}')
fig.tight_layout();fig.savefig(OUTPUT/'model_selection_preview.png',dpi=160,bbox_inches='tight');plt.show()

archive=shutil.make_archive('/kaggle/working/printed_label_model_selection_results','zip',OUTPUT)
print('\\nMODEL SELECTION PASSED')
print('Download:',archive)
print('Do not attach the real test dataset until this output is saved.')
""")

    template["cells"] = [
        markdown("""# Printed-label pre-test diffusion-distance selection

This notebook compares partial-diffusion distances 50, 100 and 250 using only the ten
registered **validation normals** (`N09` and `N10`) plus controlled digital defects with
known masks. It does not load `N11`, `N12`, or any real defect image.

Each candidate threshold is calibrated from the untouched validation normals. The
distance with the highest mean synthetic-defect Dice is selected and frozen for the
real test. Synthetic validation supports parameter selection; it is not reported as
real detection accuracy.

Attach only `printed_label_validation_v1.zip` and `printed_label_latest.pt`. Select a
Kaggle GPU, enable Internet, run all cells, and download the result ZIP.
"""),
        setup,
        install_cell,
        embedded_author_cell,
        load_author_cell,
        data,
        model,
        evaluate,
        preview,
    ]
    DESTINATION.write_text(json.dumps(template, indent=1), encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
