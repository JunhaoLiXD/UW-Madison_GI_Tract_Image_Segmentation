# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Kaggle competition solution for **UW-Madison GI Tract Image Segmentation**: multi-class
(3-organ) medical image segmentation of `large_bowel`, `small_bowel`, `stomach` from abdominal
MRI slices. There is no Python package, no test suite, and no build/lint tooling — the entire
project is three Jupyter notebooks meant to be run cell-by-cell on **Kaggle GPU runtimes**, plus
committed artifacts (trained checkpoint, figures, history spreadsheet) and `README.md` /
`docs/*.docx` write-ups. Final result: public 0.80577 / private 0.77297.

## How the pieces fit (run order)

The three notebooks in `notebooks/` form a sequential pipeline; each is standalone but depends on
artifacts from the previous stage:

1. `uw-madison-gi-tract-image-segmentation.ipynb` — **training**. Parses image paths + `train.csv`,
   pivots annotations long→wide, builds 2.5D paths, sets up `GroupKFold` by case, trains the model,
   and writes `best_effnetb0_unet_c_order_fold0.pth` + `training_history_c_order_fold0.csv`. It also
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
- **`img_size = 384`**, **EfficientNet-B0 U-Net** via `smp.Unet(..., classes=3, activation=None)`
  (raw logits; sigmoid applied only for metrics/inference), **GroupKFold(n_splits=5)** grouped by
  `case` (prevents patient leakage), validation = fold 0.

## Key conventions

- Loss = `BCEWithLogitsLoss + 3 * DiceLoss(mode="multilabel", from_logits=True)`.
- Dice metric defines empty-pred AND empty-GT as Dice = 1.0 (so correctly-empty slices aren't
  penalized). Many slices have no visible organ.
- Checkpoints may be saved either as a raw `state_dict` or a dict with `model_state_dict`; the
  loaders handle both and strip any `module.` prefix. The `last_..._fold0.pth` resume checkpoint
  additionally carries `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch`,
  and `best_dice` so training state can be fully restored.
- Per-class inference thresholds (not 0.5): large_bowel 0.39, small_bowel 0.59, stomach 0.33.

## Working in this repo

- Edit notebook code via the notebook tooling; do not hand-craft `.ipynb` JSON.
- There are no commands to build/lint/test. To "run" anything, the target is Kaggle, not local.
- `models/*.pth`, `figures/*.png`, `results/*.xlsx` are committed outputs; the README notes large
  checkpoints/datasets are normally kept out of git.
