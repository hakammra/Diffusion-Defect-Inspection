from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "07_fabric_stain_pilot_train_kaggle.ipynb"
OUTPUT = ROOT / "notebooks" / "08_fabric_stain_train_kaggle.ipynb"

nb = json.loads(SOURCE.read_text(encoding="utf-8"))

replacements = {
    "# Fabric stain DTU-Net/Tsimplex training pilot": "# Fabric stain DTU-Net/Tsimplex training continuation",
    "for 100 optimizer\nsteps": "to 2,000 total optimizer\nsteps",
    "It is a pipeline and timing pilot, not a\ncompleted detector": "It resumes the verified 100-step pilot and remains a reduced training\nrun rather than a completed detector",
    "Attach `fabric_stain_pilot.zip` as a Kaggle notebook input": "Attach `fabric_stain_pilot.zip` and `fabric_stain_latest.pt` as Kaggle notebook inputs",
    "OUTPUT = WORK/'fabric_stain_pilot_training'": "OUTPUT = WORK/'fabric_stain_training'",
    "TARGET_STEPS = 100": "TARGET_STEPS = 2000",
    "SAVE_EVERY = 25": "SAVE_EVERY = 250",
    "    print('Starting a new 100-step fabric pilot.')": "    raise FileNotFoundError('Attach the 100-step checkpoint dataset before continuing. Visible .pt files: '+repr([str(path) for path in all_pt_files]))",
    "'scope':'fabric_stain_training_pilot'": "'scope':'fabric_stain_training_continuation'",
    "title='Fabric stain training pilot'": "title='Fabric stain training continuation'",
    "'/kaggle/working/fabric_stain_pilot_results'": "'/kaggle/working/fabric_stain_training_results'",
    "FABRIC TRAINING PILOT PASSED": "FABRIC TRAINING CONTINUATION PASSED",
    "Inspect pilot stability, then continue training before locked image-level evaluation.": "Calibrate on held-out normal images, then perform the locked image-level stain evaluation.",
}

for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    for old, new in replacements.items():
        source = source.replace(old, new)
    cell["source"] = source.splitlines(keepends=True)

nb["metadata"]["colab"] = {"name": OUTPUT.name, "provenance": []}
OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUTPUT)
