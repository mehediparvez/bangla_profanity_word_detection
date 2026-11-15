"""Extra analysis used in the thesis: common-subset scores, confidence intervals,
a paired test between two models, and a check of the comments the LLM did not classify.

Everything works on plain label/prediction lists, so it can run on predictions saved
from any of the four approaches.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


def toxic_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def subset_scores(y_true, y_pred, mask):
    """Accuracy, weighted F1 and toxic-class F1 on the rows where mask is True."""
    y_true, y_pred, mask = np.asarray(y_true), np.asarray(y_pred), np.asarray(mask, bool)
    t, p = y_true[mask], y_pred[mask]
    return {
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(t, p)),
        "weighted_f1": float(weighted_f1(t, p)),
        "toxic_f1": float(toxic_f1(t, p)),
    }


def bootstrap_ci(y_true, y_pred, metric=weighted_f1, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap interval for a metric, resampling test comments with replacement."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = metric(y_true[idx], y_pred[idx])
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, y_pred)), float(low), float(high)


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact McNemar test on the comments where exactly one model is right.

    Returns (only_a_correct, only_b_correct, two-sided p-value).
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a = int(np.sum(a_ok & ~b_ok))
    only_b = int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return only_a, only_b, p


def coverage_check(y_true, classified_mask):
    """Class balance of the comments that were and were not classified."""
    y_true, mask = np.asarray(y_true), np.asarray(classified_mask, bool)
    out = {}
    for name, sel in (("classified", mask), ("unclassified", ~mask)):
        n = int(sel.sum())
        toxic = int(y_true[sel].sum())
        out[name] = {"n": n, "toxic": toxic, "toxic_share": toxic / n if n else float("nan")}
    return out


def sample_errors(texts, y_true, y_pred, error_type, k=10, seed=0):
    """Random sample of false positives ('false_positive') or false negatives ('false_negative')."""
    want_pred = 1 if error_type == "false_positive" else 0
    idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p and p == want_pred]
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(k, len(idx)), replace=False) if idx else []
    return [{"index": int(i), "text": texts[i], "true": int(y_true[i]), "pred": int(y_pred[i])} for i in pick]
