"""Combine several models by adding their log-probabilities."""

import time

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
        self.models_ = []
        for build, _ in self.members:
            started = time.time()
            print(f"blend: fitting {build.__name__} ...", flush=True)
            self.models_.append(build().fit(texts, y))
            print(f"blend: fitted {build.__name__} in {time.time() - started:.0f}s", flush=True)
        self.classes_ = self.models_[0].classes_
        for model, (build, _) in zip(self.models_, self.members):
            if not np.array_equal(model.classes_, self.classes_):
                raise ValueError(f"{build.__name__} knows classes {model.classes_}, not {self.classes_}")
        return self

    def predict_proba(self, texts):
        total = 0.0
        for model, (build, weight) in zip(self.models_, self.members):
            started = time.time()
            total = total + weight * np.log(model.predict_proba(texts) + EPS)
            print(f"blend: {build.__name__} predicted {len(texts)} passages in "
                  f"{time.time() - started:.0f}s", flush=True)  # fmt: skip
        total -= total.max(axis=1, keepdims=True)
        probs = np.exp(total)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]
