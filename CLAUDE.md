# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Kaggle competition solution for **UW-Madison GI Tract Image Segmentation**: multi-class
(3-organ) medical image segmentation of `large_bowel`, `small_bowel`, `stomach` from abdominal
MRI slices. There is no Python package, no test suite, and no build/lint tooling — the entire
project is three Jupyter notebooks meant to be run cell-by-cell on **Kaggle GPU runtimes**, plus
committed artifacts (trained checkpoint, figures, history spreadsheet) and `README.md` /
`docs/*.docx` write-ups. Submitted results: EfficientNet-B0 baseline public 0.80577 /
private 0.77297; **EfficientNet-B5 + aux head (latest, best) public 0.81244 / private 0.77691**.

## Current experiment: EfficientNet-B5 + aux head (submitted — beats B0)

An upscale experiment over the B0 baseline. Relative to B0 it changes:

- **Encoder `efficientnet-b5`** (was `b0`) at **`img_size = 456`** (B5's native resolution, was 384).
- **Gradient accumulation** to fit B5 in 16GB: `batch_size = 2` × `accum_steps = 4` (effective batch 8).
- **Auxiliary classification head** (`CFG.use_aux_cls = True`): `smp.Unet(aux_params=...)` adds a
  per-organ presence logit; the model now returns `(seg_logits, cls_logits)` (see `split_outputs`).
  The loss gains a `cls_weight = 0.3` BCE term on presence labels derived from the masks.
- Output filenames carry the `effnetb5` tag (`best_effnetb5_unet_c_order_fold0.pth`, etc.).

**Status (2026-06-25):** training is **complete — 20/20 epochs**, and the cosine schedule (`T_max=20`)
has fully annealed (final lr = `eta_min` 1e-6). Best validation Dice **0.9112 @ epoch 20** (B0 baseline
was 0.8832 @ epoch 16) — large_bowel 0.902 / small_bowel 0.883 / stomach 0.949. Gains over the last
~7 epochs are noise-level (≤0.0003 from epoch 13's 0.9110), so the model has effectively converged;
more epochs on the same schedule are not worth it. The full history is
`results/training_history_effnetb5_c_order_fold0.csv`.

The **threshold-tuning and submission notebooks were realigned to the B5 setup**
(`efficientnet-b5`, `img_size=456`, `use_aux_cls=True`, plus a `forward_seg_logits` helper that
builds `smp.Unet(aux_params=...)` and pulls `seg` out of the `(seg, cls)` tuple before sigmoid), and
B5 was **submitted with its own re-tuned thresholds**: leaderboard **public 0.81244 / private 0.77691**
(B0 was 0.80577 / 0.77297, so +0.0067 public / +0.0039 private). B5 is now the best submission.

To resume training across Kaggle sessions (not needed now that it's done), the prior session's
`last_effnetb5_..._fold0.pth`, `best_effnetb5_..._fold0.pth`, and
`training_history_effnetb5_c_order_fold0.csv` must be copied back into `/kaggle/working` (the notebook
has a restore cell for this just before "Run Training").

## How the pieces fit (run order)

The three notebooks in `notebooks/` form a sequential pipeline; each is standalone but depends on
artifacts from the previous stage:

1. `uw-madison-gi-tract-image-segmentation.ipynb` — **training**. Parses image paths + `train.csv`,
   pivots annotations long→wide, builds 2.5D paths, sets up `GroupKFold` by case, trains the model,
   and writes `best_<model_tag>_unet_c_order_fold0.pth` + `training_history_<model_tag>_c_order_fold0.csv`
   (the `model_tag` is derived from the encoder, e.g. `effnetb0` / `effnetb5`). It also
   writes a per-epoch `last_..._fold0.pth` resume checkpoint and supports resuming an interrupted run
   (`CFG.resume`, default True) plus a per-session wall-clock budget (`CFG.max_train_hours`, default
   11h) that stops training cleanly before Kaggle's hard 12h session limit — so a >12h run can finish
   across multiple sessions by re-running.
2. `gi-tract-image-segmentation-threshold-tuning.ipynb` — **threshold tuning**. Rebuilds the *same*
   validation split, loads the checkpoint, sweeps per-class probability thresholds (0.01→0.69 step
   0.02), writes `best_thresholds_training_style_c_order_fold0.json`.
3. `gi-tract-image-segmentation-submission-notebook.ipynb` — **inference**. Loads the checkpoint +
   thresholds, runs test inference, encodes RLE, writes `submission.csv`. Begins with an offline
   `pip install --no-index` cell that installs `segmentation-models-pytorch` / `timm` from `.whl`
   files under `/kaggle/input` (Kaggle submission runtimes have no internet).

Each notebook has its own `CFG` class holding all hyperparameters and paths. Paths are hardcoded to
Kaggle mounts (`/kaggle/input/...`, `/kaggle/working`) and will not resolve on this local Windows
machine — editing/reviewing here is fine, but the notebooks only *run* on Kaggle.

## Critical invariants (must stay identical across all three notebooks)

The most common way to break this project is letting preprocessing drift between training and
inference. The following must match in every notebook:

- **`classes = ["large_bowel", "small_bowel", "stomach"]`** — channel order is load-bearing;
  thresholds, logits, and RLE columns are all indexed by it.
- **C-order RLE** — both `rle_decode` (reshape `order="C"`) and `rle_encode` (flatten `order="C"`).
  The README and a code comment explicitly note a prior F-order bug caused spatially misaligned
  masks. Do not change this.
- **2.5D input** — each sample stacks slices `[n-2, n, n+2]` (`slice_stride = 2`) into a 3-channel
  image; missing neighbors fall back to the nearest available slice within the same `(case, day)`.
- **Percentile normalization** — `read_image` clips to the 1st/99th intensity percentiles and
  scales to `[0,1]`. Identical in all three notebooks.
- **`img_size`**, **encoder**, and the **aux-head flag** must match across notebooks. The original
  baseline was **`img_size = 384`** + **EfficientNet-B0** + no aux head (public 0.80577); **all three
  notebooks are now on the B5 setup** — **`img_size = 456`** + **EfficientNet-B5** +
  `use_aux_cls = True` (submitted, public 0.81244, the current best). Either way the seg U-Net is `smp.Unet(..., classes=3, activation=None)` (raw
  logits; sigmoid applied only for metrics/inference); with the aux head it additionally takes
  `aux_params=dict(pooling="avg", dropout=0.2, classes=3, activation=None)` and returns a
  `(seg_logits, cls_logits)` tuple, so the inference notebooks pull `seg` via `forward_seg_logits`.
  **GroupKFold(n_splits=5)** grouped by `case` (prevents patient leakage), validation = fold 0. To go
  back to the B0 baseline, set encoder `b0`, `img_size = 384`, `use_aux_cls = False` in all three.

## Key conventions

- Loss = `BCEWithLogitsLoss + 3 * DiceLoss(mode="multilabel", from_logits=True)`. When the aux
  classification head is enabled (B5 experiment), a `0.3 * BCEWithLogitsLoss` term on per-organ
  presence labels (derived from the masks via `masks_to_presence`) is added.
- Dice metric defines empty-pred AND empty-GT as Dice = 1.0 (so correctly-empty slices aren't
  penalized). Many slices have no visible organ.
- Checkpoints may be saved either as a raw `state_dict` or a dict with `model_state_dict`; the
  loaders handle both and strip any `module.` prefix. The `last_..._fold0.pth` resume checkpoint
  additionally carries `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch`,
  and `best_dice` so training state can be fully restored.
- Per-class inference thresholds (not 0.5): large_bowel 0.39, small_bowel 0.59, stomach 0.33.
  **These were tuned on the B0 baseline**; the B5 model needs its own threshold-tuning pass (the
  submission notebook loads thresholds from `thresholds_json_path`, falling back to these B0 values
  with a warning if the JSON is absent).

## Working in this repo

- Edit notebook code via the notebook tooling; do not hand-craft `.ipynb` JSON.
- There are no commands to build/lint/test. To "run" anything, the target is Kaggle, not local.
- `models/*.pth`, `figures/*.png`, `results/*.xlsx` are committed outputs; the README notes large
  checkpoints/datasets are normally kept out of git.
