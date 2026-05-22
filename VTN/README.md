# VTN

This folder is the cleaned working project for **VTN: Volterra-enhanced Transformer Network**.

The original workspace contains multiple experiments, checkpoints, generated outputs, and older scripts. This folder keeps only the files needed for the revised paper and the next round of controlled experiments.

## Structure

- `models/`: core VTN model implementation.
  - `restormer_volterra.py`
  - `volterra_layer.py`
- `scripts/`: top-level training, evaluation, efficiency, ablation, and visualization utilities.
- `tasks/`: task-specific training and testing scripts.
- `multiple_distortion/`: composite-degradation generation and evaluation utilities.
- `paper/`: revised paper draft and experiment plan.
- `checkpoints/`: selected pretrained checkpoints needed for evaluation and resumed experiments.
- `data/`: dataset junctions/links for stable experiment paths.
- `datasets/`: dataset loader implementations copied from the original workspace.

## External Paths

Generated result images are intentionally not part of the clean project layout. Large datasets and checkpoints should be linked or hardlinked when possible rather than duplicated. The original workspace-level folders remain:

- `E:/restormer+volterra/data`
- `E:/restormer+volterra/dataset`
- `E:/restormer+volterra/checkpoints`
- `E:/restormer+volterra/results`

Some copied scripts still contain hard-coded paths from the original workspace. Before rerunning experiments, update them to use `VTN/data` and `VTN/checkpoints`, or move the paths into a shared config.

## Next Cleanup Steps

- Normalize script imports so every script can run from the `VTN` root.
- Rename paper-facing model strings from older names such as `RestormerVolterra`, `ReVolT`, or `VETNet` to `VTN` where appropriate.
- Add a single experiment config file for dataset/checkpoint/result paths.
