import numpy as np

from src.analysis import (
    bootstrap_ci, coverage_check, mcnemar_exact, sample_errors, subset_scores,
)
from src.lexicon import build_lexicon, predict, select_lexicon, tokenize


def test_subset_scores_uses_only_masked_rows():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]
    s = subset_scores(y_true, y_pred, [True, True, False, False])
    assert s["n"] == 2 and s["accuracy"] == 1.0


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.where(rng.random(300) < 0.9, y, 1 - y)
    est, lo, hi = bootstrap_ci(y, p, n_boot=300)
    assert lo <= est <= hi
    assert bootstrap_ci(y, p, n_boot=300) == (est, lo, hi)


def test_mcnemar_counts_and_identical_models():
    y = [1, 1, 0, 0, 1]
    a = [1, 1, 0, 1, 1]   # one error (index 3)
    b = [0, 0, 0, 0, 1]   # two errors (index 0, 1)
    only_a, only_b, p = mcnemar_exact(y, a, b)
    assert (only_a, only_b) == (2, 1) and 0 < p <= 1
    assert mcnemar_exact(y, a, a) == (0, 0, 1.0)


def test_coverage_check_splits_by_mask():
    out = coverage_check([1, 0, 1, 1], [True, True, False, False])
    assert out["classified"]["toxic"] == 1 and out["unclassified"]["toxic"] == 2


def test_sample_errors_picks_requested_type_only():
    texts = list("abcd")
    out = sample_errors(texts, [0, 1, 1, 0], [1, 0, 1, 0], "false_positive", k=5)
    assert [e["text"] for e in out] == ["a"]


def test_lexicon_learns_toxic_words_and_ignores_rare_or_mixed_words():
    texts = ["খারাপ কথা", "খারাপ লোক", "ভালো কথা", "ভালো লোক", "বিরল খারাপ"]
    labels = [1, 1, 0, 0, 1]
    lex = build_lexicon(texts, labels, min_count=2, min_toxic_ratio=0.9)
    assert "খারাপ" in lex and "ভালো" not in lex and "বিরল" not in lex
    assert predict(["এটা খারাপ", "এটা ভালো"], lex) == [1, 0]


def test_tokenize_splits_on_punctuation():
    assert tokenize("ভালো, খারাপ!") == ["ভালো", "খারাপ"]


def test_select_lexicon_returns_best_validation_score():
    fit = ["খারাপ"] * 6 + ["ভালো"] * 6
    fit_y = [1] * 6 + [0] * 6
    lex, settings, score = select_lexicon(fit, fit_y, ["খারাপ", "ভালো"], [1, 0])
    assert "খারাপ" in lex and score == 1.0 and "min_count" in settings
