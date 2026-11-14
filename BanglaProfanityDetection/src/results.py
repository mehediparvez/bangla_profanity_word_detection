"""
Shared on-disk results format so the CLI, notebooks and plots agree.

Layout (relative to the results directory):
    <key>/evaluation.json      metrics, confusion_matrix, errors  (every model)
    <key>/summary.json         mean/std over multiple runs        (fine-tuned only)
    <key>/learning_curve.json  list of per-fraction metrics       (fine-tuned only)
"""

import json
import os

import numpy as np

from .evaluate import compute_metrics, get_confusion_matrix
from .utils import save_metrics


MODEL_DISPLAY_NAMES = {
    "better_profanity": "better_profanity",
    "banglabert": "BanglaBERT",
    "xlmr": "XLM-RoBERTa",
    "llm": "LLM (zero-shot)",
}

METRICS = ["accuracy", "f1", "precision", "recall"]


def build_evaluation(y_true, y_pred, inference_time=None, extra=None):
    metrics = compute_metrics(y_true, y_pred)
    if inference_time:
        metrics["inference_time_seconds"] = inference_time
        metrics["samples_per_second"] = len(y_pred) / inference_time

    cm = get_confusion_matrix(y_true, y_pred, labels=[0, 1])
    evaluation = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": cm.tolist(),
        "errors": {
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
        },
    }
    if extra:
        evaluation.update(extra)
    return evaluation


def save_evaluation(evaluation, results_dir, key):
    path = os.path.join(results_dir, key, "evaluation.json")
    save_metrics(evaluation, path)
    return path


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect_results(results_dir="results"):
    """Load every saved model result into the inputs of generate_all_plots."""
    comparison, curves, cm_dict, error_counts = {}, {}, {}, {}

    for key, name in MODEL_DISPLAY_NAMES.items():
        evaluation = _load_json(os.path.join(results_dir, key, "evaluation.json"))
        summary = _load_json(os.path.join(results_dir, key, "summary.json"))
        curve = _load_json(os.path.join(results_dir, key, "learning_curve.json"))

        row = {}
        if summary:
            # Fine-tuned models: mean and std over the training runs
            for m in METRICS:
                stats = summary.get(f"eval_{m}", {})
                row[m] = stats.get("mean", 0)
                row[f"{m}_std"] = stats.get("std", 0)
        elif evaluation:
            for m in METRICS:
                row[m] = evaluation["metrics"].get(m, 0)
                row[f"{m}_std"] = 0.0

        if evaluation:
            row["samples_per_second"] = evaluation["metrics"].get("samples_per_second", 0)
            cm_dict[name] = np.array(evaluation["confusion_matrix"])
            error_counts[name] = evaluation["errors"]

        if row:
            comparison[name] = row
        if curve:
            curves[name] = curve

    return comparison, curves, cm_dict, error_counts
