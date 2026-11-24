"""A blend, then the likelihood under the GLM models for the passages it thinks may be GLM's.

Scoring a passage under both GLM models costs about 0.03 s per token on
Kaggle's T4 x2, plus about 53 minutes of reading the weights per run, so all
~4,000 test passages would take about 6.5 hours. Instead the blend runs
first, and only the passages to which it gives the most glm-5.3 plus
glm-5.3-flash probability are scored: at most `fraction` of them, fewer if
the time left would not cover them. On train.csv the top 30% hold 96% of the
real GLM passages, and adding the GLM classifier's log-probabilities to those
at weight 1.5 lifts the blend from 0.592 to 0.626 (glm-5.3 recall 0.20 to
0.57, glm-5.3-flash 0.37 to 0.56).

The GLM classifier is a logistic regression on the likelihood features. It is
fitted on the train.csv features that train_glm_likelihood.ipynb saved
(glm_<model>_train.csv, attach that notebook's output), so train.csv is not
scored again. If the GLM run fails or runs out of time, the blend's own
predictions stand.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import LABELS, find_data_dir, load_train

STARTED = time.time()  # about when the notebook started: src.models is imported first thing
EPS = 1e-12
GLM_LABELS = [LABELS.index("glm-5.3"), LABELS.index("glm-5.3-flash")]
# seconds per run (reading the weights) and per token row, measured on train.csv
COST = {"glm53_flash": (1350.0, 0.0113), "glm53": (1820.0, 0.0199)}


def find_glm_train(model, root="/kaggle/input"):
    """The train.csv features of `model` saved by train_glm_likelihood.ipynb."""
    hits = sorted(Path(root).rglob(f"glm_{model}_train.csv"))
    if not hits:
        raise FileNotFoundError(
            f"no glm_{model}_train.csv under {root}: attach the output of ntl-train-glm-likelihood"
        )
    return hits[0]


def token_rows(texts):
    """Rows ntl-segrun will process per passage: the prompt prefix plus about 1.3 tokens per word."""
    words = np.array([len(str(text).split()) for text in texts])
    return 13 + np.minimum(320, np.ceil(1.35 * words))


def affordable(rows, models, seconds):
    """How many of the passages (in the given order) all `models` can score in `seconds`."""
    fixed = sum(COST[m][0] for m in models)
    per_row = sum(COST[m][1] for m in models)
    if seconds <= fixed:
        return 0
    return int(np.searchsorted(np.cumsum(rows) * per_row, seconds - fixed, side="right"))


class GatedGLMBlend:
    def __init__(self, build_base, weight=1.5, fraction=0.3, C=0.1, hours=8.0, margin_s=1800):
        self.build_base = build_base
        self.weight = weight
        self.fraction = fraction
        self.C = C
        self.hours = hours  # Kaggle stops a submission at 9 hours
        self.margin_s = margin_s

    def fit(self, texts, y):
        from src.features.glm import SEGMENTS
        from src.glm_experiment import feature_matrix

        self.base_ = self.build_base().fit(texts, y)
        self.classes_ = self.base_.classes_
        train = load_train(find_data_dir())
        assert list(train["text"]) == list(texts), "the GLM features are those of train.csv, in its order"
        ids = train["id"].astype(str)
        tables = {m: pd.read_csv(find_glm_train(m), dtype={"id": str}).set_index("id").reindex(ids)
                  for m in SEGMENTS}  # fmt: skip
        # one classifier on both models, and one on Flash alone in case GLM-5.3 does not finish
        self.classifiers_ = {}
        for models in (tuple(SEGMENTS), ("glm53_flash",)):
            X = feature_matrix({m: tables[m] for m in models})
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
            )
            classifier.fit(X.to_numpy(), np.asarray(y))
            assert np.array_equal(classifier.classes_, self.classes_)
            self.classifiers_[models] = (classifier, list(X.columns))
        return self

    def predict_proba(self, texts):
        from src.features.glm import SEGMENTS, score
        from src.glm_experiment import feature_matrix

        texts = list(texts)
        probs = self.base_.predict_proba(texts)
        log_probs = np.log(probs + EPS)

        mass = probs[:, GLM_LABELS].sum(axis=1)
        order = np.argsort(-mass, kind="stable")
        wanted = max(1, int(round(self.fraction * len(texts))))
        rows = token_rows([texts[i] for i in order])
        seconds = self.hours * 3600 - (time.time() - STARTED) - self.margin_s
        count = min(wanted, affordable(rows, tuple(SEGMENTS), seconds))
        models = tuple(SEGMENTS)
        if count == 0:
            count, models = min(wanted, affordable(rows, ("glm53_flash",), seconds)), ("glm53_flash",)
        print(f"GLM: scoring {count} of {len(texts)} passages under {models} "
              f"({seconds / 3600:.1f} h left)", flush=True)  # fmt: skip
        if count == 0:
            return probs
        chosen = order[:count]
        ids = [str(i) for i in chosen]

        tables = {}
        for model in models:
            budget = self.hours * 3600 - (time.time() - STARTED) - self.margin_s / 2
            try:
                tables[model] = score([texts[i] for i in chosen], model, "glm_test", ids=ids,
                                      budget_s=budget).reindex(ids)  # fmt: skip
            except Exception as error:
                print(f"GLM: {model} failed ({error!r}); using what finished", flush=True)
                break
        key = tuple(m for m in models if m in tables)
        if key not in self.classifiers_:
            print("GLM: no usable run; the blend's predictions stand", flush=True)
            return probs
        classifier, columns = self.classifiers_[key]
        X = feature_matrix({m: tables[m] for m in key})[columns]
        complete = X.notna().all(axis=1).to_numpy()
        rows_done = chosen[complete]
        glm = classifier.predict_proba(X.to_numpy()[complete])
        log_probs[rows_done] += self.weight * np.log(glm + EPS)
        print(f"GLM: added to {len(rows_done)} passages", flush=True)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        out = np.exp(log_probs)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]
