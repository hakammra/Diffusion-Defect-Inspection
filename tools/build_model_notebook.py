import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
source=(root/'src/author_smoke.py').read_text(encoding='utf8')
compile(source,'author_smoke.py','exec')
def md(s):return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)}
cells=[
 md('# LabelInspect - author model integration check\n\nThis runs the actual author DTU-Net (UDHVT in the repository), the custom Tsimplex generator, the author L2 noise loss, and a short author reconstruction path. It performs only **two optimizer updates on synthetic normal images**. It does not train a useful anomaly detector or reproduce reported metrics.\n\n**Requirements:** the verified T4 runtime, Internet access for five pinned GitHub source files, and the already-installed PyTorch/timm/einops/Numba stack. This notebook does not reinstall PyTorch. Run on one GPU. The first Numba compilation can take several minutes.\n\nSource: Kumar et al., WACV 2025; https://github.com/MAXNORM8650/Annotsim, commit `dc4a9bd2a2a5b1c31223daab4bdfea3f6a5b2990`. Import fixes, the tuple-output adapter and selected configuration are recorded in the report.'),
 code("from pathlib import Path\nimport json, sys, subprocess, importlib.metadata\nimport torch\nassert torch.cuda.is_available(), 'Select a GPU accelerator first.'\nfor name in ['timm','einops','numba','numpy','Pillow']:\n    print(name, importlib.metadata.version(name))\nprint('GPU:', torch.cuda.get_device_name(0))\nwork = Path('/kaggle/working/labelinspect') if Path('/kaggle/working').exists() else Path.cwd() / 'labelinspect'\nwork.mkdir(parents=True, exist_ok=True)\n"),
 code("script = work / 'author_smoke.py'\nscript.write_text("+repr(source)+", encoding='utf8')\nprint('Wrote:', script)\n"),
 code("output = work / 'artifacts' / 'author_integration'\nsubprocess.run([sys.executable, '-u', str(script), '--output', str(output)], check=True)\n"),
 code("report = json.loads((output / 'integration_report.json').read_text())\nprint(json.dumps(report, indent=2))\nfrom IPython.display import display, Image\ndisplay(Image(filename=str(output / 'integration_preview.png')))\nimport shutil\narchive = shutil.make_archive(str(work / 'author_integration_results'), 'zip', output)\nprint('Download this result bundle:', archive)\n"),
 md('## What passing means\n\nThe selected architecture, noise, gradient update, and short reconstruction execute together on this runtime. It does not establish segmentation accuracy, speed at the full paper batch size, or reproducibility of every author variant. The preview is explicitly labelled as an integration check.\n\nSave `integration_report.json`. Next: a resumable training pipeline, dataset split, validation protocol, and a limited benchmark experiment before the printed-label extension. No threshold from this check should be used on real photographs.'),
]
for i,c in enumerate(cells):
 c['id']=f'model-{i:02d}'
 if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
out=root/'notebooks/02_author_model_check.ipynb';out.write_text(json.dumps(nb,indent=2),encoding='utf8')
print(out)
combined=json.loads((root/'notebooks/00_gpu_check.ipynb').read_text())
combined['cells'].extend(cells)
(root/'notebooks/Kaggle_Working_Checks.ipynb').write_text(json.dumps(combined,indent=2),encoding='utf8')
