"""Score a weighted blend of models from their saved out-of-fold predictions.

    python -m src.blend_oof tfidf_logreg=1 style_logreg=0.5 deberta_base=0.5

Each name is a file oof/<name>.npy written by src.train. The blend adds
weight * log-probability, the same rule as src/models/blend.py, so a blend
can be tried in seconds without refitting any model.
"""

import argparse

import numpy as np

from src.data import load_train, make_folds
from src.evaluate import fold_scores
from src.train import log_experiment, report

EPS = 1e-12


def blend(oofs, weights):
    return sum(w * np.log(oof + EPS) for oof, w in zip(oofs, weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("members", nargs="+", help="name=weight, e.g. deberta_base=0.5")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--owner", default="")
    parser.add_argument("--log-as", default=None, help="append to experiments.csv under this name")
    args = parser.parse_args()

    names, weights = zip(*(member.split("=") for member in args.members))
    weights = [float(w) for w in weights]
    df = load_train(args.data_dir)
    folds = make_folds(df)
    y_true = df["label"].to_numpy()

    oofs = [np.load(f"oof/{name}.npy") for name in names]
    for name, oof in zip(names, oofs):
        report(f"{name} alone", fold_scores(y_true, oof.argmax(axis=1), folds))
    scores = fold_scores(y_true, blend(oofs, weights).argmax(axis=1), folds)
    report("blend", scores)
    if args.log_as:
        log_experiment("experiments.csv", args.owner, args.log_as, scores, " ".join(args.members))


if __name__ == "__main__":
    main()
