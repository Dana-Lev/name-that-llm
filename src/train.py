"""Cross-validate a model from src.models.MODELS.

    python -m src.train --model tfidf_style_blend --owner Dana

Prints balanced accuracy per fold, plain and with the one-model-per-prompt
assignment (src/prompts.py), saves out-of-fold probabilities to oof/, and
appends two rows to experiments.csv.
"""

import argparse
import csv
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

from src.data import LABELS, load_train, make_folds
from src.evaluate import fold_scores, lab_confusion, within_lab_error_share
from src.models import MODELS
from src.prompts import assign_within_groups, cluster_prompts


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


def assign_by_fold(df, folds, oof):
    """Apply the assignment to each validation fold, with prompts found by clustering.

    The clustering sees only that fold's texts and never prompt_id, which is
    what will happen on the test set.
    """
    predicted = np.zeros(len(df), dtype=int)
    for fold in np.unique(folds):
        valid = folds == fold
        groups = cluster_prompts(df["text"][valid])
        predicted[valid] = assign_within_groups(oof[valid], groups)
    return predicted


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


def report(title, scores):
    print(f"{title}: {scores.mean():.4f} +- {scores.std():.4f}  folds {np.round(scores, 4)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="tfidf_style_blend", choices=sorted(MODELS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--owner", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--no-log", action="store_true", help="don't append to experiments.csv")
    args = parser.parse_args()

    df = load_train(args.data_dir)
    folds = make_folds(df)
    oof = cross_validate(df, folds, MODELS[args.model])
    y_true = df["label"].to_numpy()

    plain = oof.argmax(axis=1)
    assigned = assign_by_fold(df, folds, oof)
    # Upper bound for the assignment: the true prompts instead of clusters.
    assigned_true = assign_within_groups(oof, df["prompt_id"].to_numpy())

    plain_scores = fold_scores(y_true, plain, folds)
    assigned_scores = fold_scores(y_true, assigned, folds)
    report("plain argmax", plain_scores)
    report("assignment, clustered prompts", assigned_scores)
    report("assignment, true prompts", fold_scores(y_true, assigned_true, folds))
    share = within_lab_error_share(y_true, assigned)
    print(f"after assignment, mistakes that stay inside the true lab: {share:.1%}")
    print(lab_confusion(y_true, assigned))

    Path("oof").mkdir(exist_ok=True)
    np.save(Path("oof") / f"{args.model}.npy", oof)
    if not args.no_log:
        log_experiment("experiments.csv", args.owner, args.model, plain_scores, args.notes)
        log_experiment(
            "experiments.csv",
            args.owner,
            f"{args.model}+assign",
            assigned_scores,
            "one model per clustered prompt",
        )


if __name__ == "__main__":
    main()
