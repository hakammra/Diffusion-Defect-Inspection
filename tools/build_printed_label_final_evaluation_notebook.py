"""Build the locked real printed-label evaluation notebook for Kaggle."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "12_printed_label_model_selection_kaggle.ipynb"
SELECTION_PATH = ROOT / "results" / "printed_labels" / "model_selection.json"
DESTINATION = ROOT / "notebooks" / "13_printed_label_final_evaluate_kaggle.ipynb"


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
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selected = selection["selected"]
    if selected["distance"] != 50:
        raise ValueError("Expected the previously selected distance t=50")
    selected_json = json.dumps(selected, separators=(",", ":"))

    install_cell = template["cells"][2]
    embedded_author_cell = template["cells"][3]
    load_author_cell = template["cells"][4]

    setup = code(f"""from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, importlib.util, json, math, random, shutil, subprocess, sys, time, zipfile

import numpy as np
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle.'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'

KAGGLE_INPUT=Path('/kaggle/input')
WORK=Path('/kaggle/working/labelinspect')
WORK.mkdir(parents=True,exist_ok=True)
OUTPUT=WORK/'printed_label_final_evaluation'
OUTPUT.mkdir(parents=True,exist_ok=True)

def find_test_datasets(search_root):
    found=[]
    for manifest in search_root.rglob('manifest.csv'):
        candidate=manifest.parent
        if (candidate/'images'/'normal').is_dir() and (candidate/'ground_truth'/'tear').is_dir():
            found.append(candidate)
    return sorted(set(found))

dataset_candidates=find_test_datasets(KAGGLE_INPUT)
if not dataset_candidates:
    matching=[]
    for archive_path in KAGGLE_INPUT.rglob('*.zip'):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names=['/'+item.filename.replace('\\\\','/').lstrip('/') for item in archive.infolist()]
                if any('/printed_label_test_locked_v1/manifest.csv' in name for name in names):
                    matching.append(archive_path)
        except zipfile.BadZipFile:
            pass
    if len(matching)==1:
        extraction_root=WORK/'uploaded_real_test';extraction_root.mkdir(parents=True,exist_ok=True)
        resolved=extraction_root.resolve()
        with zipfile.ZipFile(matching[0]) as archive:
            for item in archive.infolist():
                target=(extraction_root/item.filename).resolve()
                if target!=resolved and resolved not in target.parents:
                    raise ValueError(f'Unsafe ZIP member: {{item.filename}}')
            archive.extractall(extraction_root)
        dataset_candidates=find_test_datasets(extraction_root)
if len(dataset_candidates)!=1:
    raise FileNotFoundError('Expected one locked printed-label test dataset; found: '+repr([str(p) for p in dataset_candidates]))
DATA_ROOT=dataset_candidates[0]

checkpoint_candidates=sorted(set(KAGGLE_INPUT.rglob('printed_label_latest.pt')) | set(KAGGLE_INPUT.rglob('latest.pt')))
if not checkpoint_candidates:
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
                    archive.write(source_file,f'{{archive_root.name}}/{{source_file.relative_to(archive_root).as_posix()}}')
        checkpoint_candidates=[rebuilt]
        print('Rebuilt Kaggle-extracted checkpoint:',rebuilt)
if len(checkpoint_candidates)!=1:
    raise FileNotFoundError('Expected one printed-label checkpoint; found: '+repr([str(p) for p in checkpoint_candidates]))
CHECKPOINT=checkpoint_candidates[0]

FROZEN_SELECTION=json.loads(r'''{selected_json}''')
SEED=230274
IMAGE_SIZE=224
BATCH_SIZE=4
T_DISTANCE=50
PIXEL_THRESHOLD=0.023506546393036842
IMAGE_SCORE_QUANTILE=0.995
IMAGE_THRESHOLD=0.030804647132754326
assert FROZEN_SELECTION['distance']==T_DISTANCE
assert abs(FROZEN_SELECTION['pixel_threshold']-PIXEL_THRESHOLD)<1e-12
assert abs(FROZEN_SELECTION['image_threshold']-IMAGE_THRESHOLD)<1e-12
print('GPU:',torch.cuda.get_device_name(0))
print('Locked test dataset:',DATA_ROOT)
print('Checkpoint:',CHECKPOINT)
print('Frozen distance and thresholds:',T_DISTANCE,PIXEL_THRESHOLD,IMAGE_THRESHOLD)
""")

    data = code(r'''def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

protocol=json.loads((DATA_ROOT/'protocol.json').read_text())
assert protocol['status']=='locked_for_first_real_test',protocol
with (DATA_ROOT/'manifest.csv').open(newline='',encoding='utf8') as stream:
    records=list(csv.DictReader(stream))
expected_counts={'normal':10,'missing_print':15,'smudge':15,'tear':13}
assert Counter(row['category'] for row in records)==Counter(expected_counts),Counter(row['category'] for row in records)
assert len(records)==53 and sum(int(row['is_anomaly']) for row in records)==43
assert len({row['image'] for row in records})==len(records)
for row in records:
    image_path=DATA_ROOT/row['image'];mask_path=DATA_ROOT/row['mask']
    assert image_path.is_file() and mask_path.is_file(),row
    assert sha256(image_path)==row['image_sha256'],image_path
    assert sha256(mask_path)==row['mask_sha256'],mask_path

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

tensors=[];truth_masks=[]
for row in records:
    with Image.open(DATA_ROOT/row['image']) as opened:tensors.append(tensor_from_image(opened))
    with Image.open(DATA_ROOT/row['mask']) as opened:
        truth_masks.append(np.asarray(pad_image(opened.convert('L'),resample=NEAREST,fill=0))>0)
tensors=torch.stack(tensors)
truth_masks=np.stack(truth_masks)
image_labels=np.asarray([int(row['is_anomaly']) for row in records],dtype=bool)
categories=np.asarray([row['category'] for row in records])
names=np.asarray([Path(row['image']).stem for row in records])
physical_ids=np.asarray([row['physical_id'] for row in records])

label_roi=np.zeros((IMAGE_SIZE,IMAGE_SIZE),dtype=bool)
resized_height=round(650*IMAGE_SIZE/1063);roi_top=(IMAGE_SIZE-resized_height)//2
label_roi[roi_top:roi_top+resized_height,:]=True
assert tensors.shape==(53,3,224,224) and truth_masks.shape==(53,224,224)
assert not truth_masks[~image_labels].any()
assert all(mask.any() for mask in truth_masks[image_labels])
print('Hashes and masks verified for',len(records),'locked test images.')
print('Counts:',dict(Counter(categories)))
''')

    model = code(r'''device=torch.device('cuda:0')
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
print('Compiling Tsimplex noise function on first use...')
diffusion.noise_fn(torch.zeros(1,1,4,4,device=device),torch.tensor([5],device=device))
torch.cuda.reset_peak_memory_stats()
print('Loaded frozen step',checkpoint['step'],'checkpoint from author commit',author.COMMIT)
''')

    inference = code(r'''def reconstruct(inputs):
    outputs=[];batch_times=[]
    for start in range(0,len(inputs),BATCH_SIZE):
        x=inputs[start:start+BATCH_SIZE].to(device)
        tick=time.perf_counter()
        with torch.inference_mode():
            result=diffusion.forward_backward(model,x,None,see_whole_sequence=None,
                                               t_distance=T_DISTANCE,denoise_fn='noise_fn')
        torch.cuda.synchronize();batch_times.append(time.perf_counter()-tick)
        outputs.append(result.cpu())
        print(f'Reconstructed {min(start+BATCH_SIZE,len(inputs))}/{len(inputs)}')
    return torch.cat(outputs),batch_times

# This is the first model access to the locked real test images.
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
reconstructions,batch_times=reconstruct(tensors)
score_maps=(tensors-reconstructions).square().mean(dim=1).numpy().astype(np.float32)
predicted_masks=(score_maps>PIXEL_THRESHOLD)&label_roi[None]
image_scores=np.quantile(score_maps[:,label_roi],IMAGE_SCORE_QUANTILE,axis=1)
image_predictions=image_scores>IMAGE_THRESHOLD
mean_seconds_per_image=float(sum(batch_times)/len(records))
print('Inference complete; mean seconds/image:',mean_seconds_per_image)
''')

    metrics = code(r'''from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

def safe_div(numerator,denominator):
    return float(numerator/denominator) if denominator else None

def segmentation_metrics(pred,truth):
    pred=np.asarray(pred,dtype=bool);truth=np.asarray(truth,dtype=bool)
    tp=int(np.sum(pred&truth));fp=int(np.sum(pred&~truth));fn=int(np.sum(~pred&truth));tn=int(np.sum(~pred&~truth))
    return {
        'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'dice':safe_div(2*tp,2*tp+fp+fn),
        'iou':safe_div(tp,tp+fp+fn),
        'precision':safe_div(tp,tp+fp),
        'recall':safe_div(tp,tp+fn),
    }

per_image=[]
for index,row in enumerate(records):
    segmentation=segmentation_metrics(predicted_masks[index]&label_roi,truth_masks[index]&label_roi)
    per_image.append({
        'file':names[index],
        'category':categories[index],
        'physical_id':physical_ids[index],
        'capture_condition':row['capture_condition'],
        'is_anomaly':int(image_labels[index]),
        'image_score':float(image_scores[index]),
        'image_detected':int(image_predictions[index]),
        **segmentation,
    })

tp_img=int(np.sum(image_predictions&image_labels));tn_img=int(np.sum(~image_predictions&~image_labels))
fp_img=int(np.sum(image_predictions&~image_labels));fn_img=int(np.sum(~image_predictions&image_labels))
image_summary={
    'tp':tp_img,'tn':tn_img,'fp':fp_img,'fn':fn_img,
    'auroc':float(roc_auc_score(image_labels,image_scores)),
    'average_precision':float(average_precision_score(image_labels,image_scores)),
    'sensitivity':safe_div(tp_img,tp_img+fn_img),
    'specificity':safe_div(tn_img,tn_img+fp_img),
    'precision':safe_div(tp_img,tp_img+fp_img),
    'f1':safe_div(2*tp_img,2*tp_img+fp_img+fn_img),
    'balanced_accuracy':float((safe_div(tp_img,tp_img+fn_img)+safe_div(tn_img,tn_img+fp_img))/2),
}

roi_truth=truth_masks[:,label_roi].reshape(-1)
roi_scores=score_maps[:,label_roi].reshape(-1)
pixel_summary={
    'auroc':float(roc_auc_score(roi_truth,roi_scores)),
    'average_precision':float(average_precision_score(roi_truth,roi_scores)),
    'normal_pixel_fpr':float(predicted_masks[~image_labels][:,label_roi].mean()),
}
defect_rows=[row for row in per_image if row['is_anomaly']]
pixel_summary['mean_dice_defect_images']=float(np.mean([row['dice'] for row in defect_rows]))
pixel_summary['mean_iou_defect_images']=float(np.mean([row['iou'] for row in defect_rows]))
pixel_summary['mean_pixel_recall_defect_images']=float(np.mean([row['recall'] for row in defect_rows]))

by_category={}
normal_indices=np.flatnonzero(categories=='normal')
for category in ['missing_print','smudge','tear']:
    indices=np.flatnonzero(categories==category)
    classification_indices=np.concatenate([normal_indices,indices])
    subset=[per_image[index] for index in indices]
    subset_truth=truth_masks[indices][:,label_roi].reshape(-1)
    subset_scores=score_maps[indices][:,label_roi].reshape(-1)
    by_category[category]={
        'image_count':len(indices),
        'physical_label_count':len(set(physical_ids[indices])),
        'image_sensitivity':float(np.mean(image_predictions[indices])),
        'image_auroc_vs_normals':float(roc_auc_score(image_labels[classification_indices],image_scores[classification_indices])),
        'image_average_precision_vs_normals':float(average_precision_score(image_labels[classification_indices],image_scores[classification_indices])),
        'mean_dice':float(np.mean([row['dice'] for row in subset])),
        'mean_iou':float(np.mean([row['iou'] for row in subset])),
        'pixel_auroc_within_defect_images':float(roc_auc_score(subset_truth,subset_scores)),
        'pixel_average_precision_within_defect_images':float(average_precision_score(subset_truth,subset_scores)),
    }

physical_rows=[]
for label_id in sorted(set(physical_ids)):
    indices=np.flatnonzero(physical_ids==label_id)
    detections=int(np.sum(image_predictions[indices]));total=len(indices)
    physical_rows.append({
        'physical_id':label_id,
        'category':categories[indices[0]],
        'capture_count':total,
        'detected_capture_count':detections,
        'capture_detection_rate':float(detections/total),
        'any_capture_detected':int(detections>0),
        'majority_captures_detected':int(detections>=math.ceil(total/2)),
        'median_image_score':float(np.median(image_scores[indices])),
        'max_image_score':float(np.max(image_scores[indices])),
    })

# These two inputs were rejected before the model because no defensible
# registration was possible. Keep hybrid accounting distinct from model metrics.
registration_rejected_anomalies=2
hybrid_tp=tp_img+registration_rejected_anomalies
hybrid_fn=fn_img
hybrid_summary={
    'registered_model_tp':tp_img,
    'registration_stage_tp':registration_rejected_anomalies,
    'total_anomaly_photographs':45,
    'sensitivity':safe_div(hybrid_tp,hybrid_tp+hybrid_fn),
    'specificity':safe_div(tn_img,tn_img+fp_img),
    'warning':'Hybrid pipeline result; registration rejects are not DTU-Net detections.',
}

report={
    'status':'completed_first_locked_real_test',
    'scope':'single_fixed_design_printed_label_pilot',
    'paper_result_reproduced':False,
    'checkpoint_step':int(checkpoint['step']),
    'author_commit':author.COMMIT,
    'registered_test_images':len(records),
    'registered_normal_images':int(np.sum(~image_labels)),
    'registered_anomaly_images':int(np.sum(image_labels)),
    'physical_test_labels':len(set(physical_ids)),
    'frozen_configuration':{
        'seed':SEED,'t_distance':T_DISTANCE,'pixel_threshold':PIXEL_THRESHOLD,
        'image_score_quantile':IMAGE_SCORE_QUANTILE,'image_threshold':IMAGE_THRESHOLD,
    },
    'image_level_model_only':image_summary,
    'pixel_level_model_only':pixel_summary,
    'by_category':by_category,
    'end_to_end_hybrid':hybrid_summary,
    'mean_inference_seconds_per_image':mean_seconds_per_image,
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'gpu':torch.cuda.get_device_name(0),
    'limitations':[
        'The 43 registered anomaly images are repeated captures of nine physical defects.',
        'Missing-print samples are white-paper covered-print simulations.',
        'Smudge masks conservatively include clearly added visible strokes away from expected print.',
        'Two severe tear photographs were rejected before model inference because registration was underdetermined.',
        'This pilot does not reproduce the paper training scale or reported benchmark result.',
    ],
}

(OUTPUT/'evaluation_report.json').write_text(json.dumps(report,indent=2))
with (OUTPUT/'per_image.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(per_image[0]));writer.writeheader();writer.writerows(per_image)
with (OUTPUT/'per_physical_label.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(physical_rows[0]));writer.writeheader();writer.writerows(physical_rows)
np.savez_compressed(OUTPUT/'test_maps_float16.npz',names=names,scores=score_maps.astype(np.float16),
                    predictions=predicted_masks.astype(np.uint8),ground_truth=truth_masks.astype(np.uint8))
print(json.dumps(report,indent=2))
''')

    plots = code(r'''def show_tensor(tensor):
    return ((tensor.permute(1,2,0).numpy()+1)/2).clip(0,1)

selected_indices=[
    int(np.flatnonzero(categories=='normal')[0]),
    int(np.flatnonzero(physical_ids=='M01')[0]),
    int(np.flatnonzero(physical_ids=='S01')[0]),
    int(np.flatnonzero(physical_ids=='T01')[0]),
]
fig,axes=plt.subplots(4,5,figsize=(15,10))
for row,index in enumerate(selected_indices):
    axes[row,0].imshow(show_tensor(tensors[index]));axes[row,0].set_title(f'{names[index]} | input')
    axes[row,1].imshow(show_tensor(reconstructions[index]));axes[row,1].set_title('Reconstruction')
    axes[row,2].imshow(score_maps[index],cmap='magma');axes[row,2].set_title(f'Residual | {image_scores[index]:.4f}')
    axes[row,3].imshow(truth_masks[index],cmap='gray',vmin=0,vmax=1);axes[row,3].set_title('Locked ground truth')
    axes[row,4].imshow(predicted_masks[index],cmap='gray',vmin=0,vmax=1);axes[row,4].set_title(f'Prediction | detected={bool(image_predictions[index])}')
    for axis in axes[row]:axis.axis('off')
fig.suptitle('First locked real printed-label evaluation | frozen t=50')
fig.tight_layout();fig.savefig(OUTPUT/'evaluation_preview.png',dpi=160,bbox_inches='tight');plt.show()

fpr,tpr,_=roc_curve(image_labels,image_scores)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
axes[0].plot(fpr,tpr,label=f"AUROC = {image_summary['auroc']:.3f}")
axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Image-level ROC')
axes[0].legend();axes[0].grid(alpha=.25)
score_groups=[image_scores[categories==category] for category in ['normal','missing_print','smudge','tear']]
axes[1].boxplot(score_groups,tick_labels=['normal','missing','smudge','tear'],showfliers=True)
axes[1].axhline(IMAGE_THRESHOLD,color='red',linestyle='--',label='frozen threshold')
axes[1].set(ylabel='99.5th-percentile residual',title='Frozen image scores by category')
axes[1].legend();axes[1].grid(alpha=.25,axis='y')
fig.tight_layout();fig.savefig(OUTPUT/'image_score_diagnostics.png',dpi=160,bbox_inches='tight');plt.show()

archive=shutil.make_archive('/kaggle/working/printed_label_final_evaluation_results','zip',OUTPUT)
print('\nLOCKED REAL TEST EVALUATION COMPLETED')
print('Download from Kaggle working files:',archive)
''')

    template["cells"] = [
        markdown("""# LabelInspect — first locked real-test evaluation

This notebook evaluates the trained DTU-Net/Tsimplex checkpoint on the held-out
real printed-label test set. The model, partial-diffusion distance, random seed,
pixel threshold and image threshold were fixed before this dataset was opened by
the model. Ground-truth masks were also created before any real-test prediction.

Attach exactly two Kaggle inputs:

1. `printed_label_test_locked_v1.zip`
2. `printed_label_latest.pt`

Select a GPU accelerator, enable Internet, and run all cells. Expected runtime on
a Tesla T4 is roughly 4–8 minutes after dependencies and the first Tsimplex
compilation complete. Download `printed_label_final_evaluation_results.zip` from
the Kaggle working files when the final cell finishes.

The notebook reports model-only results on 53 registered images. Two additional
severe-tear photographs that could not be registered are reported separately as
registration-stage rejects and are never described as DTU-Net detections.
"""),
        setup,
        install_cell,
        embedded_author_cell,
        load_author_cell,
        data,
        model,
        inference,
        metrics,
        plots,
    ]
    template.setdefault("metadata", {}).setdefault("colab", {})["name"] = DESTINATION.name
    DESTINATION.write_text(json.dumps(template, indent=1), encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
