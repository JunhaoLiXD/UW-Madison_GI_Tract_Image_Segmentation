# Model Checkpoints

The trained model checkpoints are **not stored in this repository**. They exceed GitHub's
per-file size limit (the EfficientNet-B5 checkpoint is ~370 MB, over the 100 MB hard cap), so
`models/*.pth` is git-ignored.

## Available checkpoints

| File | Encoder | img_size | Aux head | Best val Dice | Public / Private LB |
|---|---|---:|:---:|---:|---|
| `best_effnetb0_unet_c_order_fold0.pth` | EfficientNet-B0 | 384 | no | 0.8832 | 0.80577 / 0.77297 |
| `best_effnetb5_unet_c_order_fold0.pth` | EfficientNet-B5 | 456 | yes | 0.9112 | 0.81244 / 0.77691 |

Both are fold-0 `smp.Unet` checkpoints saved as a dict with `model_state_dict` (the B5 file also
carries the auxiliary classification head). See `CLAUDE.md` and the notebooks for the exact
architecture and how to load them.

## How to obtain the weights

If you need the checkpoint files, please **open an issue** on the GitHub repository, or reach out via
the maintainer's GitHub profile [@JunhaoLiXD](https://github.com/JunhaoLiXD), and they can be shared.
