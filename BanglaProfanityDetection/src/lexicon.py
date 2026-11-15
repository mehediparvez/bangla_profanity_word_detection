"""Bangla word-list baseline.

The list is learned from the training split only: a word goes into the list when it is
frequent enough and appears mostly in toxic comments. The two cut-offs are chosen on the
validation split, so the test set plays no part in building or tuning the list.
"""

import re
from collections import Counter

from sklearn.metrics import f1_score

# Keep letters, digits and Bangla combining marks; everything else separates words
_SPLIT = re.compile(r"[^\w়া-্ৗ]+", re.UNICODE)


def tokenize(text):
    return [w for w in _SPLIT.split(str(text).lower()) if w]


def build_lexicon(texts, labels, min_count=5, min_toxic_ratio=0.9):
    """Words seen in at least `min_count` comments, of which at least `min_toxic_ratio` are toxic."""
    total, toxic = Counter(), Counter()
    for text, label in zip(texts, labels):
        for word in set(tokenize(text)):
            total[word] += 1
            if label == 1:
                toxic[word] += 1
    return {
        w for w, n in total.items()
        if n >= min_count and toxic[w] / n >= min_toxic_ratio
    }


def predict(texts, lexicon):
    return [1 if any(w in lexicon for w in tokenize(t)) else 0 for t in texts]


def select_lexicon(
    fit_texts, fit_labels, val_texts, val_labels,
    min_counts=(3, 5, 10), ratios=(0.7, 0.8, 0.9, 0.95),
):
    """Grid-search the two cut-offs on the validation split (weighted F1).

    Returns (lexicon, settings, validation_f1).
    """
    best = None
    for min_count in min_counts:
        for ratio in ratios:
            lex = build_lexicon(fit_texts, fit_labels, min_count, ratio)
            score = f1_score(val_labels, predict(val_texts, lex), average="weighted", zero_division=0)
            if best is None or score > best[2]:
                best = (lex, {"min_count": min_count, "min_toxic_ratio": ratio}, score)
    return best
