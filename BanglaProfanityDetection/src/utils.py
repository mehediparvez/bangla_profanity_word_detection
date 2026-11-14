import os
import yaml
import json
import pandas as pd


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def save_results_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
