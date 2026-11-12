"""End-to-end check of folds, cross-validation and scoring on synthetic data."""

import numpy as np
import pandas as pd

from src.data import LABEL_TO_ID, LABELS, make_folds
from src.evaluate import (
    balanced_accuracy,
    fold_scores,
    lab_confusion,
    model_confusion,
    within_lab_error_share,
)
from src.models.linear import build_tfidf_logreg
from src.train_baseline import cross_validate


def synthetic_train(n_prompts=30, seed=0):
    """Every model answers every prompt; each model has one tell-tale word."""
    rng = np.random.default_rng(seed)
    filler = "the of and to in is that it for as with was on be by".split()
    rows = []
    for p in range(n_prompts):
        for model in LABELS:
            words = list(rng.choice(filler, size=40)) + [f"topic{p}"] * 3
            words += [f"tell{LABEL_TO_ID[model]}"] * 2
            rng.shuffle(words)
            rows.append(
                {
                    "id": f"{p}-{model}",
                    "prompt_id": f"p{p}",
                    "text": " ".join(words),
                    "model": model,
                }
            )
    df = pd.DataFrame(rows)
    df["label"] = df["model"].map(LABEL_TO_ID)
    return df


def test_labels():
    assert len(LABELS) == 18
    assert len(set(LABELS)) == 18


def test_folds_keep_prompts_together():
    df = synthetic_train()
    folds = make_folds(df)
    assert set(folds) == {0, 1, 2, 3, 4}
    assert (df.groupby("prompt_id")["label"].count() == 18).all()
    assert (pd.Series(folds).groupby(df["prompt_id"].to_numpy()).nunique() == 1).all()
    assert np.array_equal(folds, make_folds(df))


def test_cross_validation_learns_the_signal():
    df = synthetic_train()
    folds = make_folds(df)
    oof = cross_validate(df, folds, build_tfidf_logreg)
    assert oof.shape == (len(df), 18)
    assert np.allclose(oof.sum(axis=1), 1.0)
    y_true, y_pred = df["label"].to_numpy(), oof.argmax(axis=1)
    assert fold_scores(y_true, y_pred, folds).mean() > 0.9


def test_confusion_tables():
    y_true = np.array([0, 0, 1, 5, 5, 17])
    y_pred = np.array([0, 1, 1, 5, 0, 17])
    assert balanced_accuracy(y_true, y_pred) == (0.5 + 1 + 0.5 + 1) / 4
    assert model_confusion(y_true, y_pred).to_numpy().sum() == 6
    labs = lab_confusion(y_true, y_pred)
    assert labs.shape == (6, 6)
    assert labs.loc["OpenAI", "OpenAI"] == 3
    assert labs.loc["Google", "OpenAI"] == 1
    # two mistakes: gpt-5.5 -> gpt-5.6-luna (same lab) and gemini -> gpt-5.5 (other lab)
    assert within_lab_error_share(y_true, y_pred) == 0.5


def test_make_submission(tmp_path):
    from src.predict import make_submission

    train = synthetic_train(n_prompts=10)
    train.drop(columns="label").to_csv(tmp_path / "train.csv", index=False)
    test = train.tail(18)[["id", "text"]]
    test.to_csv(tmp_path / "test.csv", index=False)

    out = tmp_path / "submission.parquet"
    submission = make_submission(tmp_path, out)

    written = pd.read_parquet(out)
    assert list(written.columns) == ["id", "model"]
    assert list(written["id"]) == list(test["id"])
    assert set(written["model"]) <= set(LABELS)
    # the test rows are training rows, so the model should get them right
    assert (submission["model"].to_numpy() == train.tail(18)["model"].to_numpy()).mean() > 0.9


def test_notebook_is_up_to_date():
    """The committed notebook must match src/; rebuild it after changing src/."""
    from src.build_notebook import NOTEBOOK, render

    assert NOTEBOOK.read_text() == render(), "run: python -m src.build_notebook"
