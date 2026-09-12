"""Build portable notebooks and a source audit without importing upstream code."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path('C:/Users/abdul/Documents/Annotsim_Presentation/source/Annotsim-main')
(ROOT/'notebooks').mkdir(exist_ok=True)
(ROOT/'reports').mkdir(exist_ok=True)

def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text):return {'cell_type':'code','metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[]}
def notebook(name,cells):
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'cells':cells}
    for i,c in enumerate(cells):
        c['id']=f'cell-{i:02d}'
        if c['cell_type']=='code':compile(''.join(c['source']),name,'exec')
    (ROOT/'notebooks'/name).write_text(json.dumps(nb,indent=2),encoding='utf8')

notebook('00_gpu_check.ipynb',[
 md('# LabelInspect - GPU access check\n\nThis checks the runtime, not the paper model. No private data or tokens are required.\n\n**Kaggle:** import this notebook, select an available GPU in Settings > Accelerator, then run the cells. Complete account verification directly with Kaggle if required. **Colab fallback:** upload this notebook and choose a GPU runtime. Availability is controlled by the provider.\n\nOnly keep the GPU enabled while doing GPU work. Save `gpu_environment.json` after running.'),
 code('''import json, platform, sys
from pathlib import Path
import torch
report = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
          'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
if torch.cuda.is_available():
    report['gpu'] = torch.cuda.get_device_name(0)
    report['gpu_vram_gib'] = torch.cuda.get_device_properties(0).total_memory / 2**30
print(json.dumps(report, indent=2))
assert torch.cuda.is_available(), 'No GPU is active. Select a GPU accelerator or check account access/quota.'
'''),
 code('''# Small hardware check only: this is not DTU-Net and does not train on images.
torch.manual_seed(230224)
device = torch.device('cuda:0')
model = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(2, 3, 64, 64, device=device)
target = torch.randn_like(x)
losses = []
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
torch.cuda.synchronize()
report['small_optimization_check'] = 'passed'
report['check_losses'] = losses
out = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
path = out / 'gpu_environment.json'
path.write_text(json.dumps(report, indent=2))
print('GPU check passed. Saved:', path)
print('GPU:', report['gpu'])
'''),
 md('## Next step\nKeep the printed GPU name and the JSON report. The paper-model notebook will need a separately tested dependency set, data, and checkpoint/resume support. Passing this check is not proof that the paper configuration fits in GPU memory. Do not install the original requirements list blindly.'),
])

baseline=(ROOT/'src/baseline.py').read_text(encoding='utf8')
notebook('01_cpu_baseline.ipynb',[
 md('# LabelInspect - CPU baseline\n\nRun this with the GPU **off**. It generates synthetic fixed-layout labels and checks the template-residual evaluation pipeline. It does not reproduce the diffusion paper. Real photographed labels and the diffusion implementation are later milestones.\n\nDependencies: NumPy and Pillow. The baseline is embedded below so no project upload or API key is needed.'),
 code("from pathlib import Path\nimport sys, subprocess\nimport numpy\nimport PIL\nwork = Path('/kaggle/working/labelinspect') if Path('/kaggle/working').exists() else Path.cwd() / 'labelinspect'\nwork.mkdir(parents=True, exist_ok=True)\nscript = work / 'baseline.py'\nscript.write_text("+repr(baseline)+", encoding='utf8')\nprint('Workspace:', work)\nprint('NumPy:', numpy.__version__, 'Pillow:', PIL.__version__)\n"),
 code("output = work / 'artifacts' / 'synthetic_check'\nsubprocess.run([sys.executable, str(script), 'demo', '--output', str(output)], check=True)\n"),
 code("from IPython.display import display, Image\ndisplay(Image(filename=str(output / 'preview.png')))\n"),
 md('## Interpretation\nThe masks and metrics are for synthetic, aligned labels only. A strong score here establishes that this simple case works; it says nothing about real photographed-label generalization or DTU-Net performance. The template uses training images and the threshold uses normal validation images. The held-out reference masks are used only to evaluate the final predictions.\n\nFor real data, capture and split physical labels independently, then test registration, lighting variation, and small-defect sensitivity. Save output files before ending a hosted session.'),
])

syntax_errors=[]
sources=sorted(UPSTREAM.rglob('*.py'))
for path in sources:
    try:ast.parse(path.read_text(encoding='utf-8-sig'))
    except (SyntaxError,UnicodeError) as exc:syntax_errors.append({'file':path.relative_to(UPSTREAM).as_posix(),'error':str(exc)})
critical=['GaussianDiffusion.py','scripts/diffusion_training.py','src/models/UModels/UDHVT.py','utils/Simplex/noise.py','utils/dataset.py','requirements.txt','README.md']
audit={'source_url':'https://github.com/MAXNORM8650/Annotsim','source_kind':'previously downloaded local snapshot; no remote commit claimed','source_files':len(sources),'syntax_errors':syntax_errors,'critical_sha256':{n:hashlib.sha256((UPSTREAM/n).read_bytes()).hexdigest() for n in critical},'readme_entrypoints':{n:(UPSTREAM/n).exists() for n in ['src/scripts/diffusion_training_UVW.py','src/scripts/detection.py']},'checkpoint_files':[p.relative_to(UPSTREAM).as_posix() for p in UPSTREAM.rglob('*') if p.suffix in ('.pt','.pth','.ckpt')],'checks_not_performed':['upstream dependency installation','author model forward/backward','diffusion training','paper result reproduction']}
(ROOT/'reports/upstream_audit.json').write_text(json.dumps(audit,indent=2))
packages={}
for name in ['numpy','Pillow','torch','torchvision','timm','einops','numba']:
    try:packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:packages[name]=None
environment={'python':sys.version,'platform':platform.platform(),'packages':packages,'hardware_observed':{'cpu':'Intel Core i7-8650U','gpu':'Intel UHD Graphics 620','ram_bytes':17092894720},'note':'Python package report describes the runtime used here, not every Python environment on the computer.'}
(ROOT/'reports/local_environment.json').write_text(json.dumps(environment,indent=2))
included=[]
with zipfile.ZipFile(ROOT/'LabelInspect_Starter.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.suffix=='.zip' or any(s in path.parts for s in ('__pycache__','artifacts','data','.venv')):continue
        relative=path.relative_to(ROOT)
        z.write(path,Path('LabelInspect')/relative);included.append(relative.as_posix())
print(json.dumps({'notebooks':2,'syntax_errors':syntax_errors,'zip_files':included},indent=2))
