"""Experiment: family specialists trained on the open pool (src/models/family_model.py).

Run on Kaggle with a GPU, deberta-v3-base and the open pool attached:

    from src.family_experiment import main; main()

For each family (GPT, Gemini, Claude) it trains, on open-pool passages of
that family's models only:

- DeBERTa-v3-base, 2 epochs, loss weighted per model, saved as family_<name>/
  so a submission can load it;
- TF-IDF and style logistic regressions, which are cheap enough to refit in a
  submission.

Each is scored on all of train.csv: the pool has no answers to train.csv's
prompts, so this is a held-out test. The printed score is balanced accuracy
among the family's own train.csv passages, choosing only among its members.
probs_family_<name>_<model>.npy holds each specialist's probabilities for
every train.csv passage (columns in FAMILIES order), for src.blend_oof-style
tuning locally against the blend's out-of-fold predictions.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.data import find_data_dir, load_train
from src.models.family_model import FAMILIES
from src.models.linear import build_style_logreg
from src.models.pool_model import build_big_tfidf_logreg
from src.models.transformer import free_gpu
from src.pool import load_pool, sample_pool
from src.pool_encoder import build_pool_encoder

DEBERTA_PER_LABEL = 8000
LINEAR_PER_LABEL = 15000


def family_score(y, probs, ids):
    """Balanced accuracy on the family's passages, choosing among its members only."""
    rows = np.isin(y, ids)
    predicted = np.asarray(ids)[probs[rows].argmax(axis=1)]
    return balanced_accuracy_score(y[rows], predicted)


def main(out_dir=".", families=tuple(FAMILIES), deberta=True, linear=True):
    out_dir = Path(out_dir)
    df = load_train(find_data_dir())
    y = df["label"].to_numpy()
    pool = load_pool()
    rows = []

    def record(family, model, probs, n_train, started):
        np.save(out_dir / f"probs_family_{family}_{model}.npy", probs)
        score = family_score(y, probs, FAMILIES[family])
        minutes = (time.time() - started) / 60
        rows.append({"family": family, "model": model, "n_train": n_train, "minutes": round(minutes),
                     "family_score": score})  # fmt: skip
        print(f"{family:7s} {model:8s} {score:.4f} ({n_train} passages, {minutes:.0f} min)", flush=True)

    for family in families:
        ids = FAMILIES[family]
        members = pool[pool["label"].isin(ids)]
        counts = members["model"].value_counts().to_dict()
        print(f"{family}: pool passages per model {counts}", flush=True)

        if deberta:
            started = time.time()
            sample = sample_pool(members, DEBERTA_PER_LABEL, matched=True)
            model = build_pool_encoder(epochs=2, balanced=True, time_limit_hours=2.5)
            model.fit(sample["text"], sample["label"])
            assert list(model.classes_) == ids
            model.save(out_dir / f"family_{family}")
            record(family, "deberta", model.predict_proba(df["text"]), len(sample), started)
            del model
            free_gpu()

        if linear:
            sample = sample_pool(members, LINEAR_PER_LABEL)
            for name, build in (("tfidf", build_big_tfidf_logreg), ("style", build_style_logreg)):
                started = time.time()
                model = build().fit(sample["text"], sample["label"])
                assert list(model.classes_) == ids
                record(family, name, model.predict_proba(df["text"]), len(sample), started)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "family_experiment.csv", index=False)
    return table
