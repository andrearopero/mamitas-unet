import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from make_fixture import make_dataset  # noqa: E402


@pytest.fixture()
def dataset_dir(tmp_path):
    return make_dataset(tmp_path / "ds")
