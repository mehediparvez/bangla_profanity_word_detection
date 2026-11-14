import os
import json
import tempfile

from src.utils import load_config, save_metrics, ensure_dirs


def test_save_and_load_metrics(tmp_path):
    metrics = {"accuracy": 0.95, "f1": 0.93}
    path = str(tmp_path / "metrics.json")
    save_metrics(metrics, path)

    with open(path) as f:
        loaded = json.load(f)
    assert loaded["accuracy"] == 0.95
    assert loaded["f1"] == 0.93


def test_ensure_dirs(tmp_path):
    d1 = str(tmp_path / "a" / "b")
    d2 = str(tmp_path / "c")
    ensure_dirs(d1, d2)
    assert os.path.isdir(d1)
    assert os.path.isdir(d2)
