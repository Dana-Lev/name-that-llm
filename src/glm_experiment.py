"""Experiment: how well does the likelihood under the GLM models name the model?

Run on Kaggle (GPU T4 x2; inputs in src/features/glm.py):

    from src.glm_experiment import main; main()

Scores every train.csv passage under GLM-5.3-Flash, then GLM-5.3 (about an
hour together), saves the features (glm_<model>_train.csv: the submission
reuses them instead of scoring train.csv again) and cross-validates a
logistic regression on them with the project's folds. The best variant's
out-of-fold probabilities land in probs_glm.npy, for oof/glm.npy and
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
from src.features.glm import SEGMENTS, score

C_VALUES = (0.01, 0.1, 1.0)


def feature_matrix(tables):
    """Features of each model side by side, plus Flash minus GLM-5.3 for each.

    The difference asks the question directly: which of the two GLMs finds
    this passage more likely.
    """
    parts = [table.add_prefix(f"{model}:") for model, table in tables.items()]
    if len(tables) == 2:
        (_, a), (_, b) = tables.items()
        parts.append((a - b).add_prefix("diff:"))
    return pd.concat(parts, axis=1)


def cross_validate(X, y, folds, C):
    oof = np.zeros((len(y), len(LABELS)))
    for fold in np.unique(folds):
        train, valid = folds != fold, folds == fold
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        )
        model.fit(X[train], y[train])
        oof[np.ix_(valid, model.classes_)] = model.predict_proba(X[valid])
    return oof


def report_glm_recall(y, predicted):
    for name in ("glm-5.3", "glm-5.3-flash", "claude-fable-5"):
        label = LABELS.index(name)
        print(f"  recall {name}: {(predicted[y == label] == label).mean():.2f}", flush=True)


def main(out_dir="."):
    out_dir = Path(out_dir)
    df = load_train(find_data_dir())
    folds = make_folds(df)
    y = df["label"].to_numpy()
    ids = df["id"].astype(str).tolist()

    tables = {}
    for model in SEGMENTS:
        try:
            table = score(df["text"], model, out_dir / "glm_out", ids=ids)
        except Exception as error:  # e.g. GLM-5.3's datasets not attached: go on with Flash
            print(f"{model} failed: {error!r}", flush=True)
            continue
        table.to_csv(out_dir / f"glm_{model}_train.csv")
        print(f"{model}: {len(table)}/{len(df)} passages scored", flush=True)
        tables[model] = table.reindex(ids)
    if not tables:
        raise RuntimeError("no GLM model could be run")

    X_all = feature_matrix(tables)
    complete = X_all.notna().all(axis=1).to_numpy()
    if not complete.all():
        print(f"only {complete.sum()} passages have every feature; scoring on those", flush=True)
    variants = {model: feature_matrix({model: table}) for model, table in tables.items()}
    if len(tables) == 2:
        variants["both"] = X_all

    rows, best = [], None
    for name, X in variants.items():
        X = X.to_numpy()[complete]
        for C in C_VALUES:
            oof = cross_validate(X, y[complete], folds[complete], C)
            scores = fold_scores(y[complete], oof.argmax(axis=1), folds[complete])
            rows.append({"features": name, "n_features": X.shape[1], "C": C,
                         "cv_mean": scores.mean(), "cv_std": scores.std()})  # fmt: skip
            print(f"{name:12s} C={C:<5} {scores.mean():.4f} +- {scores.std():.4f}", flush=True)
            report_glm_recall(y[complete], oof.argmax(axis=1))
            if best is None or scores.mean() > best[0]:
                best = (scores.mean(), name, C, oof)

    probs = np.full((len(df), len(LABELS)), 1.0 / len(LABELS))  # unscored passages: uniform
    probs[complete] = best[3]
    np.save(out_dir / "probs_glm.npy", probs)
    print(f"best: {best[1]} C={best[2]} {best[0]:.4f}; saved probs_glm.npy", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "glm_experiment.csv", index=False)
    return table
