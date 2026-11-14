import numpy as np

from src.evaluate import compute_metrics, get_confusion_matrix, aggregate_runs, analyze_errors


def test_compute_metrics_perfect():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 1, 0]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_partial():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert 0 < metrics["accuracy"] < 1.0
    assert 0 < metrics["f1"] < 1.0


def test_confusion_matrix():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    cm = get_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm[0][0] == 2  # true negatives
    assert cm[1][0] == 1  # false negatives
    assert cm[1][1] == 1  # true positives


def test_confusion_matrix_single_class():
    cm = get_confusion_matrix([0, 0], [0, 0])
    assert cm.shape == (2, 2)
    assert cm[0][0] == 2


def test_aggregate_runs():
    runs = [
        {"accuracy": 0.9, "f1": 0.88},
        {"accuracy": 0.92, "f1": 0.90},
        {"accuracy": 0.91, "f1": 0.89},
    ]
    summary = aggregate_runs(runs)
    assert "accuracy" in summary
    assert "f1" in summary
    assert abs(summary["accuracy"]["mean"] - 0.91) < 0.01
    assert summary["accuracy"]["std"] > 0


def test_analyze_errors():
    texts = ["a", "b", "c", "d"]
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 1, 1]
    errors = analyze_errors(texts, y_true, y_pred)
    assert len(errors) == 2
    assert set(errors["error_type"].tolist()) == {"false_positive", "false_negative"}
