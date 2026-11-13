"""python scripts/predict.py --config configs/default.yaml"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _common import load_from_cli  # noqa: E402

from mamitas.engines.predict import predict  # noqa: E402

if __name__ == "__main__":
    predict(load_from_cli("Predict masks and report metrics with the best checkpoint."))
