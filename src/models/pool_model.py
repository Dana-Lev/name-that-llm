"""Models trained on train.csv plus the public open pool (src/pool.py)."""

import numpy as np
import pandas as pd

from src.models.linear import build_tfidf_logreg
from src.pool import load_pool, sample_pool


def build_big_tfidf_logreg():
    """TF-IDF + logistic regression sized for 100,000+ passages.

    The vocabulary and the number of optimiser steps are capped so the fit
    takes minutes, not hours.
    """
    model = build_tfidf_logreg(max_iter=300, min_df=3)
    for _, vectorizer in model.named_steps["tfidf"].transformer_list:
        vectorizer.set_params(max_features=300_000, dtype=np.float32)
    return model


class PoolAugmented:
    """Fit a model on the given passages plus a sample of the open pool.

    Same interface as the other models (fit, predict_proba, classes_). The
    pool sample is at most `per_label` passages of each model.
    """

    def __init__(self, build_model=build_big_tfidf_logreg, per_label=8000, matched=False):
        self.build_model = build_model
        self.per_label = per_label
        self.matched = matched

    def fit(self, texts, y):
        extra = sample_pool(load_pool(), self.per_label, self.matched)
        all_texts = pd.concat([pd.Series(list(texts)), extra["text"]], ignore_index=True)
        all_labels = np.concatenate([np.asarray(y), extra["label"].to_numpy()])
        self.model_ = self.build_model().fit(all_texts, all_labels)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, texts):
        return self.model_.predict_proba(texts)

    def predict(self, texts):
        return self.model_.predict(texts)


def build_pool_tfidf_logreg():
    return PoolAugmented()
