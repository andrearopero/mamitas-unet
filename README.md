# mamitas

Binary foot segmentation (class 0 = background, 1 = foot) with a U-Net whose encoder is a
pretrained MobileNetV3-Large (torchvision). The training recipe follows
Iturriago-Salas et al., *Artificial Intelligence-Driven Mobile Platform for Thermographic
Imaging to Support Maternal Health Care*, Computers 2025, 14, 466.

## Run it on Kaggle

Open `notebooks/kaggle_train.ipynb` in Kaggle, or create a notebook and run:

1. In the notebook settings: **Add Input** → `lucasiturriago/feet-mamitas` (**Version 3**),
   **Accelerator** → GPU, **Internet** → On.
2. Then:

```bash
!git clone https://github.com/andrearopero/mamitas-unet
%cd mamitas-unet
!pip install .
!python scripts/train.py --config configs/default.yaml
!python scripts/predict.py --config configs/default.yaml
```

Use `%cd`, not `!cd`: a `!` command runs in its own shell and the directory change is lost.

Kaggle mounts a dataset at different paths depending on the platform version. If
`data.dataset_dir` from the YAML does not contain `Train/images`, mamitas searches
`/kaggle/input` (down to 3 levels) and prints which folder it used. To force a folder, add
`--dataset-dir <path>` (run `!ls /kaggle/input` to see what is mounted). Outputs
(checkpoints, `history.json`, predicted masks, `metrics.json`) go to `runs/default/`.

To compare losses in one notebook, run train + predict once per loss with its own output folder:

```bash
!python scripts/train.py --config configs/default.yaml --loss dice --output-dir runs/dice
!python scripts/predict.py --config configs/default.yaml --loss dice --output-dir runs/dice
```
(`--loss` is `ce`, `dice` or `focal`; the same flag works for `train.py` and `predict.py`.)

Then compare the runs (Test metrics table, validation curves and example masks):

```bash
!python scripts/plot_results.py --dataset-dir "{p}"
```

It prints the table and saves `runs/curves.png` and `runs/examples.png`. Use `--losses ce dice`
for a subset and `--sample sample_5` to show another Test image.

Tested with torch 2.6.0 / torchvision 0.21.0 (the versions of the paper). The pretrained
MobileNetV3 weights are downloaded on first use, so Internet must be On.

## Data layout

```
<dataset_dir>/
├── Train/images/sample#.png|jpg     Train/masks/class_0/sample#.png   (or Train/masks/sample#.png)
├── Valid/...
└── Test/...
```

An image and its mask share the same file name. Split folder names are matched
case-insensitively. Masks may store `{0, 1}` (dataset v3) or `{0, 255}`; a 2-channel mask is
reduced to a label map. Dataset v3 has 248 / 71 / 35 images (train / valid / test).

## Repository layout

```
src/mamitas/
├── model/unet.py        MobileNetV3 encoder, decoder blocks, 1x1 head
├── loss/                ce.py, dice.py, focal.py
├── metrics/             iou.py, dice.py, recall.py, precision.py
├── data/                dataset.py, augmentations.py, dataloader.py
├── engines/             train.py, predict.py
└── schemas.py           Pydantic models that validate the YAML
configs/                 default.yaml (paper hyperparameters), test.yaml (tiny, CPU)
scripts/                 train.py, predict.py  (--config, --dataset-dir, --output-dir)
tests/                   pytest unit tests + a synthetic mini dataset
```

## Configuration (`configs/default.yaml`)

| Section | Keys |
|---|---|
| `data` | `dataset_dir`, split names, `image_size` (multiple of 32), `augment`, `num_workers` |
| `model` | `num_classes`, `in_channels` (3 = RGB, 1 = gray), `decoder_blocks` (4 values), `pretrained` |
| `loss` | `name` (`ce` / `dice` / `focal`), `focal_gamma`, `focal_alpha` |
| `train` | `batch_size`, `epochs`, `lr` (decoder + head), `encoder_lr`, `lr_decay`, `weight_decay`, `freeze_encoder_epochs`, `seed`, `device`, `output_dir` |
| `predict` | `split`, `checkpoint`, `output_dir` |

Unknown keys or invalid values (for example `image_size: 250`) are rejected at load time.

Defaults follow the paper: Adam, batch 36, 60 epochs, lr 1e-4 for decoder/head and 1e-5 for
the encoder, exponential decay 0.94 per epoch, input 256 x 256, focal gamma 2 / alpha 0.75, flips,
rotation within 15 degrees, color jitter and Gaussian noise. `freeze_encoder_epochs: N`
(default 0 = off) enables progressive unfreezing, opening one encoder stage every N epochs from
the deepest to the shallowest.

Batch 36 needs roughly a 16 GB GPU (the paper used a P100). On a smaller GPU lower
`train.batch_size` (a 4 GB GTX 1650 runs batch 8).

## Model

Encoder: `mobilenet_v3_large().features` split in 5 stages (16, 24, 40, 112, 960 channels at
1/2 ... 1/32 resolution). Decoder: 4 blocks of bilinear upsampling + skip concatenation + two
3x3 conv-BN-ReLU (channels from `decoder_blocks`, default 256-128-64-64). Head: 1x1 conv to
`num_classes` logits. With `in_channels: 1` the pretrained first-layer filters are summed over RGB.

## Metrics and losses

Losses: cross-entropy, soft Dice (smoothing 1) and focal. Metrics (foot class, accumulated over
the whole split): IoU, Dice, recall and precision; they are cross-checked against scikit-learn in
the tests.

## Local development

```bash
pip install -e ".[dev]"
pytest
python scripts/train.py --config configs/test.yaml     # tiny CPU smoke run (needs no internet)
```
