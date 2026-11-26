"""Balance the predictions over the test passages, using no labels.

The blend picks some models far more often than others: on train.csv, which
has 150 passages of each model, it names gpt-5.6-luna 205 times and
claude-opus-5 only 103 times. If the test set is also about equal parts of
each model, one offset per model, added to the log-probabilities, can undo
that. `balance_offsets` finds the offsets that make the average predicted
probability of every model 1/18 over the passages being predicted. It uses
no labels, only the passages themselves, so it can run on the test set.

On train.csv, balancing each held-out fold on its own lifts the GLM blend
from 0.626 to 0.647 (better in 4 of 5 folds). It assumes balance: when the
models' counts in a fold were made to differ by up to 40%, the gain was
gone, and at 60% it hurt.
"""

import numpy as np

EPS = 1e-12


def balance_offsets(log_probs, iterations=200, step=0.5):
    """Offsets b such that softmax(log_probs + b) averages 1/k for each of the k classes."""
    k = log_probs.shape[1]
    offsets = np.zeros(k)
    for _ in range(iterations):
        logits = log_probs + offsets
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        offsets -= step * np.log(probs.mean(axis=0) * k + EPS)
    return offsets


class Balanced:
    """A model whose test predictions are balanced with balance_offsets."""

    def __init__(self, build_base):
        self.build_base = build_base

    def fit(self, texts, y):
        self.base_ = self.build_base().fit(texts, y)
        self.classes_ = self.base_.classes_
        return self

    def predict_proba(self, texts):
        log_probs = np.log(self.base_.predict_proba(texts) + EPS)
        offsets = balance_offsets(log_probs)
        print("balance: offsets " + ", ".join(f"{c}:{b:+.2f}" for c, b in zip(self.classes_, offsets)),
              flush=True)  # fmt: skip
        logits = log_probs + offsets
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]
