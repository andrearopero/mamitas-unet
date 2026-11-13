"""python scripts/train.py --config configs/default.yaml"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so `_common` resolves when run from anywhere

from _common import load_from_cli  # noqa: E402

from mamitas.engines.train import train  # noqa: E402

if __name__ == "__main__":
    train(load_from_cli("Train the MobileNetV3 U-Net for foot segmentation."))
