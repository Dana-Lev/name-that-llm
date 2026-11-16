"""Experiment: how well do frozen-LLM features name the model?

Run on Kaggle with a GPU and an open language model attached:

    from src.llm_experiment import main; main()

It extracts the features of every training passage once, then cross-validates
a logistic regression on each block of features with the project's folds.
Everything lands in the output folder: the features (llm_features.npz), the
out-of-fold probabilities of each variant (oof_llm_<block>.npy) and a table
(llm_experiment.csv). The oof files go into oof/ locally for blending with
src.blend_oof.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import LABELS, find_data_dir, load_train, make_folds
from src.evaluate import fold_scores
from src.features.llm import extract, find_llm

C_VALUES = (0.01, 0.1, 1.0)


def cross_validate_block(X, y, folds, C):
    oof = np.zeros((len(y), len(LABELS)))
    for fold in np.unique(folds):
        train, valid = folds != fold, folds == fold
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        )
        model.fit(X[train], y[train])
        oof[np.ix_(valid, model.classes_)] = model.predict_proba(X[valid])
    return oof


def main(out_dir="."):
    out_dir = Path(out_dir)
    df = load_train(find_data_dir())
    folds = make_folds(df)
    y = df["label"].to_numpy()
    source = find_llm()
    print(f"language model: {source}")

    features = extract(df["text"], source)
    np.savez_compressed(out_dir / "llm_features.npz", **features)

    layer_names = [name for name in features if name != "stats"]
    blocks = dict(features)
    blocks["all_layers"] = np.hstack([features[name] for name in layer_names])
    blocks["all_layers+stats"] = np.hstack([blocks["all_layers"], features["stats"]])

    rows = []
    for name, X in blocks.items():
        best = None
        for C in C_VALUES:
            oof = cross_validate_block(X, y, folds, C)
            scores = fold_scores(y, oof.argmax(axis=1), folds)
            rows.append({"features": name, "n_features": X.shape[1], "C": C,
                         "cv_mean": scores.mean(), "cv_std": scores.std()})  # fmt: skip
            print(f"{name:18s} C={C:<5} {scores.mean():.4f} +- {scores.std():.4f}", flush=True)
            if best is None or scores.mean() > best[0]:
                best = (scores.mean(), oof)
        np.save(out_dir / f"oof_llm_{name.replace('+', '_')}.npy", best[1])

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "llm_experiment.csv", index=False)
    return table
