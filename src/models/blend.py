"""Combine several models by adding their log-probabilities."""

import numpy as np

EPS = 1e-12


class LogProbBlend:
    """Weighted sum of the members' log-probabilities, renormalised.

    Adding log-probabilities multiplies probabilities, so a class scores high
    only if every member finds it plausible. A weight below 1 makes a member's
    opinion count for less.
    """

    def __init__(self, members):
        """members: list of (build_model, weight) pairs."""
        self.members = members

    def fit(self, texts, y):
        self.models_ = [build().fit(texts, y) for build, _ in self.members]
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, texts):
        total = sum(
            weight * np.log(model.predict_proba(texts) + EPS)
            for model, (_, weight) in zip(self.models_, self.members)
        )
        total -= total.max(axis=1, keepdims=True)
        probs = np.exp(total)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]
