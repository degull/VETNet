# Revised Experiment Plan for VTN-IR

This checklist mirrors the reviewer requests and the tables in `main.tex`.

## 1. Result Consistency Audit

- Verify every number used in the previous Table 3(b), Table 4, and Table 6.
- Recompute RESIDE-6K single-task, multi-task, and ablation results from logs/checkpoints.
- Record the exact training setting for every result:
  - single-task
  - all-in-one
  - composite-trained
  - zero-shot
- Do not merge results from different settings into one table.

## 2. Fair Baseline Protocol

- Retrain at least Restormer under the same all-in-one protocol as VTN-IR.
- Add PromptIR if code/checkpoints are available.
- Add MambaIR or MambaIRv2 if code/checkpoints are available.
- For DA-CLIP, AdaIR, MoCE-IR, FoundIR, and DiffUIR, mark clearly whether results are retrained, reported, or zero-shot.

## 3. Efficiency

Measure on the same machine and report:

- Parameters
- FLOPs at fixed input sizes
- Latency in milliseconds
- Peak GPU memory
- Precision mode
- GPU model
- Batch size

Suggested input sizes:

- 256 x 256
- 512 x 512
- 1024 x 1024, if memory allows

## 4. Ablations

Required ablations:

- Baseline Transformer
- Volterra in attention only
- Volterra in FFN only
- Volterra in both paths
- Rank R = 1, 2, 4, 8, 16
- First-order only
- Second-order low-rank
- Third-order low-rank, only if feasible
- Low-rank vs full quadratic, only if feasible

## 5. Visualization

Create:

- Qualitative comparison figure for rain, blur, haze, snow, and composite degradation.
- Volterra quadratic response map.
- Feature map or interaction map comparing baseline and VTN-IR.
- Use consistent crop boxes, crop sizes, and labels.

## 6. Writing Updates

- Clearly define Volterra before using it as a model name.
- Explain why second-order interactions are useful for restoration.
- Add a notation table if space allows.
- Add a concise block diagram or pseudocode.
- Explicitly distinguish VTN-IR from Restormer, MR-VNet, HorNet, and generic gating.
- Avoid claiming "unified" based only on task-specific training.
