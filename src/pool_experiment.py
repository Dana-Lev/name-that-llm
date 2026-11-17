"""Experiment: how much is the public open pool worth to our cheap models?

Run on Kaggle (CPU is enough) with the open pool dataset attached:

    from src.pool_experiment import main; main()

Each variant trains the TF-IDF model and the style model on a sample of the
pool, with no passage from train.csv, and is scored on all of train.csv. The
pool has no answers to train.csv's prompts, so this is a held-out test and
needs no folds. The last variants add the real training passages, so they
are cross-validated with the project's folds.

Outputs: pool_experiment.csv (the table) and probs_<variant>_<model>.npy
(class probabilities for every train.csv passage, for blending locally).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.data import LABELS, find_data_dir, load_train, make_folds
from src.models.linear import build_style_logreg
from src.models.pool_model import build_big_tfidf_logreg
from src.pool import load_pool, sample_pool

EPS = 1e-12
# name -> (passages per label, typography-matched sampling)
VARIANTS = {
    "pool_2500": (2500, False),
    "pool_2500_matched": (2500, True),
    "pool_8000": (8000, False),
}
CV_PER_LABEL = 2500  # pool passages per label added to each training fold


def fit_predict(train_texts, train_labels, test_texts):
    """Fit both models; return {model name: probabilities with 18 columns}."""
    out = {}
    for name, build in (("tfidf", build_big_tfidf_logreg), ("style", build_style_logreg)):
        model = build().fit(train_texts, train_labels)
        probs = np.zeros((len(test_texts), len(LABELS)))
        probs[:, model.classes_] = model.predict_proba(test_texts)
        out[name] = probs
    out["blend"] = np.exp(np.log(out["tfidf"] + EPS) + 0.5 * np.log(out["style"] + EPS))
    return out


def main(out_dir="."):
    out_dir = Path(out_dir)
    df = load_train(find_data_dir())
    y = df["label"].to_numpy()
    pool = load_pool()
    print(f"pool: {len(pool)} passages; train.csv: {len(df)}", flush=True)
    rows = []

    def record(variant, n_train, probs_by_model, started):
        for model, probs in probs_by_model.items():
            score = balanced_accuracy_score(y, probs.argmax(axis=1))
            rows.append({"variant": variant, "model": model, "n_train": n_train, "score": score})
            np.save(out_dir / f"probs_{variant}_{model}.npy", probs / probs.sum(axis=1, keepdims=True))
            print(f"{variant:22s} {model:6s} {score:.4f}", flush=True)
        print(f"  ({variant} took {time.time() - started:.0f}s)", flush=True)

    # 1. Pool only, scored on all of train.csv.
    for variant, (per_label, matched) in VARIANTS.items():
        started = time.time()
        sample = sample_pool(pool, per_label, matched)
        record(variant, len(sample), fit_predict(sample["text"], sample["label"], df["text"]), started)

    # 2. Train.csv alone, and train.csv plus pool, cross-validated on train.csv.
    folds = make_folds(df)
    extra = sample_pool(pool, CV_PER_LABEL)
    for variant, add_pool in (("train_only_cv", False), ("train_plus_pool_cv", True)):
        started = time.time()
        oof = {name: np.zeros((len(df), len(LABELS))) for name in ("tfidf", "style", "blend")}
        for fold in np.unique(folds):
            fit_rows = df[folds != fold]
            texts, labels = fit_rows["text"], fit_rows["label"]
            if add_pool:
                texts = pd.concat([texts, extra["text"]])
                labels = pd.concat([labels, extra["label"]])
            for name, probs in fit_predict(texts, labels, df["text"][folds == fold]).items():
                oof[name][folds == fold] = probs
        n_train = int((folds != 0).sum()) + (len(extra) if add_pool else 0)
        record(variant, n_train, oof, started)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "pool_experiment.csv", index=False)
    return table
