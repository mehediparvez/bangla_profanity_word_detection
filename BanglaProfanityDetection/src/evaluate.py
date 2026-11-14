import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def get_confusion_matrix(y_true, y_pred, labels=(0, 1)):
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def get_classification_report(y_true, y_pred, labels=None):
    target_names = labels or ["non-toxic", "toxic"]
    return classification_report(
        y_true, y_pred, labels=[0, 1], target_names=target_names, zero_division=0
    )


def evaluate_model(model, tokenizer, texts, labels, max_length=256, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []

    start_time = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    inference_time = time.time() - start_time

    metrics = compute_metrics(labels, all_preds)
    metrics["inference_time_seconds"] = inference_time
    metrics["samples_per_second"] = len(texts) / inference_time
    return metrics, all_preds


def analyze_errors(texts, y_true, y_pred):
    errors = []
    for text, true, pred in zip(texts, y_true, y_pred):
        if true != pred:
            error_type = "false_positive" if pred == 1 else "false_negative"
            errors.append(
                {"text": text, "true_label": true, "predicted": pred, "error_type": error_type}
            )
    return pd.DataFrame(errors)


def aggregate_runs(all_metrics):
    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
    summary = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        summary[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }
    return summary
