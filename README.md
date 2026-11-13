# mamitas

Binary foot segmentation (class 0 = background, 1 = foot) with a U-Net whose
encoder is a pretrained MobileNetV3 (torchvision).

## Colab

```bash
!git clone https://github.com/andrearopero/mamitas-unet
%cd mamitas-unet
!pip install .
!python scripts/train.py --config configs/default.yaml
```

Work in progress.
