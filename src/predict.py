"""Fit the baseline on all of train.csv and write submission.parquet.

    python -m src.predict --data-dir data

This is what the Kaggle submission notebook runs.
"""

import argparse

import pandas as pd

from src.data import LABELS, find_data_dir, load_test, load_train
from src.models.linear import build_tfidf_logreg


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


def make_submission(data_dir=None, out="submission.parquet", build_model=build_tfidf_logreg):
    data_dir = find_data_dir() if data_dir is None else data_dir
    train, test = load_train(data_dir), load_test(data_dir)

    model = build_model()
    model.fit(train["text"], train["label"])
    predicted = model.predict(test["text"])

    submission = pd.DataFrame({"id": test["id"], "model": [LABELS[i] for i in predicted]})
    check_submission(submission, test)
    submission.to_parquet(out, index=False)
    print(f"trained on {len(train)} passages, wrote {len(submission)} predictions to {out}")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="submission.parquet")
    args = parser.parse_args()
    make_submission(args.data_dir, args.out)
