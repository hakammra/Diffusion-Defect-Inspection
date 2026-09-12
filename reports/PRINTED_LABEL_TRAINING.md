# Real printed-label training

## Outcome

The real printed-label DTU-Net/Tsimplex training run completed successfully on a
Tesla T4. The run used 40 registered normal photographs from eight physical labels
and did not use validation or test images.

| Item | Value |
|---|---:|
| Optimizer steps | 2,000 |
| Batch size | 2 |
| Model parameters | 28,863,812 |
| Initial step loss | 0.4207 |
| Final step loss | 0.0614 |
| Mean of final 20 losses | 0.0685 |
| Minimum observed loss | 0.0474 at step 1,915 |
| Runtime | 839.7 seconds |
| Median step time | 0.408 seconds |
| Peak allocated GPU memory | 0.675 GiB |

All 2,000 recorded losses were finite. The 20-step moving mean declined rapidly in
the first 500 steps and then more gradually, with expected spikes caused by different
images, diffusion timesteps and Tsimplex samples.

## Interpretation

This result shows that optimization completed and that the model learned to predict
training noise for this normal-label dataset. A falling training loss is not evidence
that defects are detected. No reconstruction quality, anomaly separation, threshold,
Dice, IoU or AUROC can be reported until independent validation and test photographs
are processed.

The checkpoint must remain separate from the small results archive. The next experiment
will freeze this checkpoint, calibrate its anomaly threshold using `N09`-`N10`, and then
evaluate once on `N11`-`N12`, missing-print, smudge and tear images.
