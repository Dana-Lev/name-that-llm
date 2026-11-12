"""Fit a model on all of train.csv and write submission.parquet.

    python -m src.predict --model tfidf_style_blend --data-dir data

This is what the Kaggle submission notebooks run.
"""

import argparse

import pandas as pd

from src.data import LABELS, find_data_dir, load_test, load_train
from src.models import MODELS
from src.prompts import assign_within_groups, cluster_prompts


def check_submission(submission, test):
    """Fail here rather than have Kaggle reject the file."""
    if list(submission.columns) != ["id", "model"]:
        raise ValueError(f"columns must be id, model; got {list(submission.columns)}")
    if submission.isna().any().any():
        raise ValueError("submission has null values")
    if submission["id"].duplicated().any() or set(submission["id"]) != set(test["id"]):
        raise ValueError("every test id must appear exactly once")
    unknown = set(submission["model"]) - set(LABELS)
    if unknown:
        raise ValueError(f"labels outside the 18 valid ones: {sorted(unknown)}")


def make_submission(model="tfidf_logreg", assign=False, data_dir=None, out="submission.parquet"):
    """With assign=True, group the test passages by prompt and let no model
    repeat inside a group (see src/prompts.py)."""
    data_dir = find_data_dir() if data_dir is None else data_dir
    train, test = load_train(data_dir), load_test(data_dir)

    fitted = MODELS[model]().fit(train["text"], train["label"])
    # Columns of predict_proba follow fitted.classes_, which are label ids.
    probs = fitted.predict_proba(test["text"])
    if assign:
        groups = cluster_prompts(test["text"])
        columns = assign_within_groups(probs, groups)
        print(f"grouped {len(test)} passages into {groups.max() + 1} prompts")
    else:
        columns = probs.argmax(axis=1)
    predicted = fitted.classes_[columns]

    submission = pd.DataFrame({"id": test["id"], "model": [LABELS[i] for i in predicted]})
    check_submission(submission, test)
    submission.to_parquet(out, index=False)
    print(f"trained on {len(train)} passages, wrote {len(submission)} predictions to {out}")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="tfidf_logreg", choices=sorted(MODELS))
    parser.add_argument("--assign", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="submission.parquet")
    args = parser.parse_args()
    make_submission(args.model, args.assign, args.data_dir, args.out)
