# Author-model integration

## What this check does

`notebooks/02_author_model_check.ipynb` tests the actual author model and selected author diffusion definitions on the verified Kaggle T4 runtime. It performs two L2 noise-prediction optimizer updates on synthetic normal labels and eight reverse-diffusion steps. It is a numerical/integration check, not useful training or a reproduced result.

Upstream commit: `dc4a9bd2a2a5b1c31223daab4bdfea3f6a5b2990`.

Five source files are downloaded from that fixed revision, and SHA-256 hashes are checked against the locally inspected copies before loading them. The custom OpenSimplex implementation is loaded in its own namespace.

## Compatibility changes

1. Replace old timm helper/layer import paths with the paths in the installed timm version. Remove unused torchvision, timm constants, and registry imports.
2. Add a NumPy import for the author file's positional-embedding helper.
3. Wrap `UDHVT.forward` to select the noise tensor from its `(prediction, 0, 0)` return value. The backbone computations are preserved.
4. Load the author diffusion class and only its required top-level definitions through Python AST. This avoids importing unrelated experiments and unused losses. The selected path supports Gaussian/Tsimplex L2 loss and sampling; this is not a general adapter for every upstream loss/variant.
5. Set the author's `train` dispatch flag to `False` so sampling calls `model(x,t,y=lab)`. Actual PyTorch training/evaluation mode is set separately with `model.train()` and `model.eval()`.
6. Choose `denoise_fn='noise_fn'` for reverse steps. In the alternative author `4dsimplex` branch, the generator is called without forwarding the configured octave/frequency/persistence values. Our selected route keeps the same configured values for training and reverse sampling.

No Gaussian posterior formulas are replaced. The structured-noise use of those formulas is an author implementation choice, not a claim that all Gaussian probability identities still hold under simplex noise.

## Explicit configuration

- Input: 224 x 224, three channels, batch size two.
- Patch size 16; embedding width 384; six attention heads.
- Code depth 12: six encoder blocks, one middle block, six decoder blocks.
- SPE patch embedding; DMHA; DAFF/HFF in encoder/middle/decoder; output refinement enabled.
- Unconditioned model, with a time token and no image-class labels.
- L2 noise-prediction loss; AdamW learning rate 0.0001.
- Tsimplex: six octaves, frequency parameter 64, persistence 0.9; author raw amplitudes retained.
- Cosine schedule with 1000 total steps; eight-step partial reconstruction solely for the integration check.

This tests the architecture illustrated by the full figure. The main paper, configurations and training script do not use one uniform depth/embedding/feed-forward setting across all experiments. These settings must not be described as an exact match to every reported table.

## What remains after a pass

The model is effectively untrained. Its reconstruction preview cannot substantiate anomaly removal. Next tasks are a reproducible dataset split, resumable training loop, measured batch-size/memory budget, validation-only hyperparameter selection, a benchmark experiment, and the real-label application comparison.

The author generator uses diffusion time as a noise coordinate. Duplicate times with the same generator seed can produce identical batch noise. Preserve this observation in the reproduction notes and test any later change as a separate variant.
