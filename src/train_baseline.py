"""Cross-validate the TF-IDF + logistic regression baseline.

    python -m src.train_baseline --data-dir data --owner A

Prints balanced accuracy per fold, saves out-of-fold probabilities to oof/,
and appends one row to experiments.csv.
"""

import argparse
import csv
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

from src.data import LABELS, load_train, make_folds
from src.evaluate import balanced_accuracy, fold_scores, lab_confusion, within_lab_error_share
from src.models.linear import build_tfidf_logreg


def cross_validate(df, folds, build_model):
    """Return out-of-fold class probabilities, shape (n_rows, 18)."""
    oof = np.zeros((len(df), len(LABELS)))
    for fold in np.unique(folds):
        train, valid = df[folds != fold], df[folds == fold]
        model = build_model()
        model.fit(train["text"], train["label"])
        # A fold's training part can miss a class; map columns by class id.
        oof[np.ix_(folds == fold, model.classes_)] = model.predict_proba(valid["text"])
    return oof


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def log_experiment(path, owner, name, scores, notes):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                date.today().isoformat(),
                owner,
                name,
                git_commit(),
                f"{scores.mean():.4f}",
                f"{scores.std():.4f}",
                "",
                notes,
            ]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--name", default="tfidf_logreg")
    parser.add_argument("--owner", default="")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--no-log", action="store_true", help="don't append to experiments.csv")
    args = parser.parse_args()

    df = load_train(args.data_dir)
    folds = make_folds(df)
    oof = cross_validate(df, folds, lambda: build_tfidf_logreg(C=args.C))

    y_true = df["label"].to_numpy()
    y_pred = oof.argmax(axis=1)
    scores = fold_scores(y_true, y_pred, folds)

    print(f"folds: {np.round(scores, 4)}")
    print(f"balanced accuracy: {scores.mean():.4f} +- {scores.std():.4f}")
    print(f"pooled out-of-fold: {balanced_accuracy(y_true, y_pred):.4f}")
    print(f"mistakes that stay inside the true lab: {within_lab_error_share(y_true, y_pred):.1%}")
    print(lab_confusion(y_true, y_pred))

    Path("oof").mkdir(exist_ok=True)
    np.save(Path("oof") / f"{args.name}.npy", oof)
    if not args.no_log:
        log_experiment("experiments.csv", args.owner, args.name, scores, f"C={args.C}")


if __name__ == "__main__":
    main()
