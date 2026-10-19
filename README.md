# UW-Madison GI Tract Image Segmentation

## Project Overview

This project builds a deep learning pipeline for multi-class medical image segmentation on the **UW-Madison GI Tract Image Segmentation** dataset. The goal is to segment three gastrointestinal organs from abdominal MRI slices:

- `large_bowel`
- `small_bowel`
- `stomach`

The final model is a 2.5D **EfficientNet-B5 U-Net with an auxiliary organ-presence classification head**. It predicts pixel-level probability maps for each organ, converts them into binary masks using class-specific thresholds, and encodes the masks into run-length encoding (RLE) format for Kaggle submission. An EfficientNet-B0 U-Net (no auxiliary head) served as the baseline.

The project follows a complete segmentation workflow, including data parsing, RLE mask decoding, 2.5D input construction, model training, validation, threshold tuning, and final submission generation.

---

## Final Submission Result

| Submission | Validation Dice (fold 0) | Public Score | Private Score |
|---|---:|---:|---:|
| EfficientNet-B0 baseline | 0.8832 | 0.80577 | 0.77297 |
| **EfficientNet-B5 + aux head (final)** | **0.9112** | **0.81244** | **0.77691** |

B5 improves the leaderboard by **+0.0067 public / +0.0039 private** over the B0 baseline.

---

## Dataset and Task

Each training image is a grayscale MRI slice from a patient case and scan day. The label file provides one row per image and organ class, where the segmentation mask is stored as an RLE string.

The original annotation table is converted from long format to wide format so that each image has one row containing the three organ masks:

```text
id | large_bowel | small_bowel | stomach
```

Empty masks are treated as background-only samples. This is important because many slices do not contain any visible target organ.

---

## Data Preprocessing

### 1. Metadata Extraction

Image metadata is extracted from the file path and image filename. This includes:

- case ID
- day ID
- slice number
- image height and width
- pixel spacing information
- full image path

This metadata is used to match images with their labels and to construct neighboring-slice inputs.

### 2. Correct RLE Decoding

The segmentation masks are decoded using **C-order** array layout. This is an important implementation detail because using the wrong RLE order causes the decoded masks to be spatially misaligned with the MRI image.

The corrected workflow uses C-order decoding for all ground-truth masks and final RLE handling.

### 3. 2.5D Input Construction

Instead of using only the current slice, the model uses a 2.5D input representation. For each center slice `n`, the input contains:

```text
[n - 2, n, n + 2]
```

These three grayscale slices are stacked as a 3-channel image. This allows the model to use limited through-plane anatomical context while still using a standard 2D segmentation network.

If a neighboring slice does not exist, the nearest available slice is used as a fallback.

### 4. Image Normalization

Each MRI slice is normalized independently using percentile clipping:

1. Compute the 1st and 99th intensity percentiles.
2. Clip intensities to this percentile range.
3. Scale the clipped image to the `[0, 1]` range.

This reduces the effect of extreme intensity values while preserving useful anatomical contrast.

### 5. Train/Validation Split

The dataset is split using `GroupKFold` with case ID as the grouping variable. This prevents images from the same patient case from appearing in both training and validation sets.

Only fold 0 is trained and validated:

```text
Number of folds: 5
Training fold: all folds except fold 0
Validation fold: fold 0
```

---

## Model Architecture

The final model is a U-Net with an EfficientNet-B5 encoder and an auxiliary classification head that shares the encoder.

```text
Input:
    2.5D MRI image
    shape = [3, 456, 456]

Encoder:
    EfficientNet-B5
    pretrained on ImageNet

Decoder:
    U-Net decoder -> seg_logits, 3 channels
    channel 0 = large_bowel
    channel 1 = small_bowel
    channel 2 = stomach

Auxiliary head (on encoder features):
    avg pooling + dropout 0.2 -> cls_logits, 3 per-organ presence logits

Output:
    (seg_logits, cls_logits)
```

The model is implemented with `segmentation_models_pytorch`; the auxiliary head is added through `aux_params`:

```python
model = smp.Unet(
    encoder_name="efficientnet-b5",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,
    activation=None,
    aux_params=dict(
        pooling="avg",
        dropout=0.2,
        classes=3,
        activation=None,
    ),
)

seg_logits, cls_logits = model(x)
```

`activation=None` means that both outputs are raw logits. Sigmoid activation is applied only during metric calculation and inference. The B0 baseline used the same U-Net with `encoder_name="efficientnet-b0"`, 384 × 384 input, and no `aux_params` (a single segmentation output).

---

## Training Configuration

| Parameter | EfficientNet-B5 (final) | EfficientNet-B0 (baseline) |
|---|---|---|
| Encoder | EfficientNet-B5 | EfficientNet-B0 |
| Encoder weights | ImageNet | ImageNet |
| Image size | 456 × 456 (B5 native) | 384 × 384 |
| Input channels / output classes | 3 / 3 | 3 / 3 |
| Batch size | 2 × 4 gradient accumulation (effective 8) | 8 |
| Aux classification head | Enabled (per-organ presence) | None |
| Loss | BCE + 3·Dice + 0.3·BCE (presence) | BCE + 3·Dice |
| Epochs | 20 | 20 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-4 | 1e-4 |
| Weight decay | 1e-2 | 1e-2 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Minimum learning rate | 1e-6 | 1e-6 |
| AMP mixed precision | Enabled when CUDA is available | Enabled when CUDA is available |
| Fold used | Fold 0 | Fold 0 |

Gradient accumulation keeps the per-step memory peak small (it is set by the per-step batch of 2), so B5 at 456 × 456 fits on a 16 GB GPU while preserving an effective batch size of 8.

---

## Checkpointing and Resumable Training

Kaggle GPU sessions are capped at a hard 12-hour limit. In "Save & Run All" (Commit) mode, a run that hits that wall **fails and discards all `/kaggle/working` outputs**, including any checkpoint written mid-run. B5 takes ~42 min per epoch, so its 20-epoch run was finished across multiple Kaggle sessions using the following mechanism:

- **Per-epoch `last` checkpoint.** In addition to the best checkpoint `best_<model_tag>_unet_c_order_fold0.pth`, every epoch writes `last_<model_tag>_unet_c_order_fold0.pth` (`<model_tag>` is derived from the encoder, e.g. `last_effnetb5_unet_c_order_fold0.pth` for B5). It stores the model, optimizer, scheduler, and AMP scaler state plus the current epoch and best Dice. The best checkpoint is still saved only when validation Dice improves.
- **Resume.** With `CFG.resume = True`, re-running the notebook detects an existing `last` checkpoint, restores all training state, and continues from the next epoch. The training-history CSV is also rebuilt from disk so it continues without duplicate rows. The notebook includes a restore cell that copies the previous session's `last` / `best` checkpoints and history CSV from `/kaggle/input` into `/kaggle/working`.
- **Predictive wall-clock guard.** `CFG.max_train_hours` (default `11.0`) sets a per-session time budget. After each epoch the notebook estimates the average epoch time and, if one more epoch would exceed the budget, stops cleanly with both checkpoints already saved — so the run is never killed mid-epoch by the 12-hour limit. Each Kaggle session gets a fresh budget, so a training run that needs more than 12 hours can finish across multiple sessions by simply re-running.

| Parameter | Value |
|---|---:|
| `CFG.max_train_hours` | 11.0 |
| `CFG.resume` | True |

---

## Data Augmentation

Training augmentation is applied using Albumentations:

```text
Resize to 456 × 456 (384 × 384 for the B0 baseline)
Horizontal flip with p = 0.5
Affine transform with small translation, scale, and rotation
Random brightness and contrast adjustment
Tensor conversion
```

Validation uses only resizing and tensor conversion. No random augmentation is applied to validation data.

---

## Loss Function

The segmentation loss combines binary cross entropy and Dice loss. For the final B5 model, a classification BCE term on per-organ presence labels is added:

```text
Loss = BCEWithLogitsLoss(seg) + 3 × DiceLoss(seg) + 0.3 × BCEWithLogitsLoss(cls)
```

BCEWithLogitsLoss helps the model learn pixel-level foreground/background classification, while Dice loss directly encourages overlap between predicted masks and ground-truth masks. The presence targets are derived from the masks (a class is "present" if its mask has any positive pixel). The B0 baseline used only the first two terms.

The Dice loss is configured for multilabel segmentation and uses logits directly:

```python
dice_loss = smp.losses.DiceLoss(
    mode="multilabel",
    from_logits=True,
    smooth=1e-6,
)
```

---

## Validation Metric

During validation, Dice score is computed from sigmoid probabilities after thresholding at `0.5` during training.

The metric reports:

- mean Dice across the three classes
- per-class Dice for each organ

If both the predicted mask and ground-truth mask are empty for a class, the Dice score is defined as `1.0` for that class. This avoids penalizing correctly predicted empty masks.

---

## Training History

The B5 model trained the full 20 of 20 epochs. The best validation Dice was reached at the final epoch; the cosine schedule had fully annealed (final lr = `eta_min` 1e-6), and validation Dice over the last ~7 epochs moved only within noise (≤0.0003), so the run has converged.

| Metric | EfficientNet-B5 (final) | EfficientNet-B0 (baseline) |
|---|---:|---:|
| Best epoch | 20 (of 20) | 16 (of 20) |
| Train loss* | 0.2249 | 0.3411 |
| Train Dice | 0.9277 | 0.8814 |
| Validation loss* | 0.2533 | 0.2874 |
| **Validation Dice** | **0.9112** | 0.8832 |
| Large bowel validation Dice | 0.9015 | 0.8723 |
| Small bowel validation Dice | 0.8828 | 0.8447 |
| Stomach validation Dice | 0.9494 | 0.9326 |
| Time per epoch | ~42 min | ~12 min |

\* Loss values are not directly comparable: the B5 loss includes the auxiliary classification term.

B5 improves validation Dice over B0 on all three organs, with the largest gain on the hardest class (small bowel). The validation gain carried over to the leaderboard (+0.0067 public / +0.0039 private).

The full per-epoch loss and Dice history is saved as CSV under `results/`
(`training_history_effnetb5_c_order_fold0.csv` for B5 and
`training_history_effnetb0_c_order_fold0.csv` for B0).

---

## Threshold Tuning

After training, a separate validation notebook is used to tune probability thresholds. The model outputs probability maps after sigmoid activation, and each organ can use a different threshold to convert probabilities into binary masks.

The threshold tuning notebook:

1. Rebuilds the same validation split used during training.
2. Loads the trained fold-0 checkpoint (for B5, the segmentation logits are taken from the `(seg_logits, cls_logits)` tuple).
3. Runs inference on the validation set.
4. Searches candidate thresholds from `0.01` to `0.69` with step size `0.02`.
5. Selects class-specific thresholds based on validation Dice and saves them to `best_thresholds_training_style_c_order_fold0.json`.

The final B5 submission uses **per-organ thresholds re-tuned on the B5 checkpoint**, which the submission notebook loads from `thresholds_json_path`. The B0 baseline was submitted with the following thresholds; the submission notebook keeps them only as a fallback (with a warning) when the threshold JSON is absent:

| Class | B0 threshold |
|---|---:|
| Large bowel | 0.39 |
| Small bowel | 0.59 |
| Stomach | 0.33 |

---

## Inference and Submission

The inference pipeline follows these steps:

1. Load the trained EfficientNet-B5 U-Net checkpoint (with the auxiliary head, `use_aux_cls = True`) and the B5 thresholds JSON.
2. Read test images and construct 2.5D inputs.
3. Normalize each input slice using percentile normalization.
4. Resize inputs to `456 × 456`.
5. Run model inference and take `seg_logits` from the `(seg_logits, cls_logits)` output (`forward_seg_logits` helper).
6. Apply sigmoid activation to convert logits into probabilities.
7. Apply class-specific thresholds.
8. Resize predicted masks back to the original image size.
9. Encode binary masks into RLE format.
10. Save the final `submission.csv` file.

The submission file contains one row for each image and organ class. The notebook reads the checkpoint and thresholds from Kaggle dataset paths (`checkpoint_path` / `thresholds_json_path`); point these at wherever the artifacts are uploaded.

---

## Repository Structure

```text
.
├── README.md
├── CLAUDE.md
├── LICENSE
├── .gitignore
├── docs/
│   └── UW_Madison_GI_Tract_Segmentation_Technical_Documentation_EN.docx
├── notebooks/
│   ├── uw-madison-gi-tract-image-segmentation.ipynb            # training
│   ├── gi-tract-image-segmentation-threshold-tuning.ipynb      # threshold tuning
│   └── gi-tract-image-segmentation-submission-notebook.ipynb   # inference / submission
├── models/
│   └── README.md   # checkpoints (*.pth) are git-ignored; see this file to obtain them
└── results/
    ├── training_history_effnetb0_c_order_fold0.csv
    └── training_history_effnetb5_c_order_fold0.csv
```

Model checkpoints (`models/*.pth`) are **not committed** — they are too large for GitHub (the B5
checkpoint is ~370 MB) and are git-ignored. See `models/README.md` for how to obtain them. Kaggle
datasets are likewise kept out of the repository.

---

## Main Dependencies

The project uses the following major Python libraries:

```text
pytorch
segmentation-models-pytorch
timm
albumentations
opencv-python
numpy
pandas
scikit-learn
matplotlib
tqdm
```

On Kaggle, internet access may be disabled during submission. In that case, external packages such as `segmentation-models-pytorch` and `timm` should be provided through an offline Kaggle Dataset containing the required wheel files.

---

## Key Implementation Notes

- C-order RLE decoding is required for correct mask alignment.
- 2.5D input improves context by using neighboring MRI slices.
- GroupKFold prevents patient-case leakage between training and validation.
- The model uses logits during training and applies sigmoid only for metrics and inference.
- Encoder, `img_size`, and the aux-head flag must match across the training, threshold-tuning, and submission notebooks.
- Threshold tuning is performed after training because the optimal probability threshold may differ across organs, and it must be re-run whenever the model changes.
- The current implementation trains only fold 0.

---

## Possible Future Improvements

Potential improvements include:

- training all 5 folds of the EfficientNet-B5 model and ensembling them (only fold 0 is submitted so far)
- using longer 2.5D context windows
- adding stronger post-processing for removing small false-positive masks
- tuning thresholds using the final competition-style metric
- experimenting with test-time augmentation
- training with additional image sizes or multi-scale inference

---

## Summary

This project implements a complete medical image segmentation pipeline for gastrointestinal organ segmentation from abdominal MRI slices. The final solution is a 2.5D EfficientNet-B5 U-Net with an auxiliary organ-presence classification head, trained with C-order RLE mask handling, case-level grouped validation, BCE plus Dice loss (plus a presence BCE term), and class-specific threshold tuning. It reaches fold-0 validation Dice `0.9112` and public `0.81244` / private `0.77691`, improving on the EfficientNet-B0 baseline (`0.8832`; public `0.80577` / private `0.77297`).
