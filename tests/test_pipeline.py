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
from src.train import cross_validate


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


def write_synthetic_data(folder):
    train = synthetic_train(n_prompts=10)
    train.drop(columns="label").to_csv(folder / "train.csv", index=False)
    test = train.tail(18)[["id", "text"]]
    test.to_csv(folder / "test.csv", index=False)
    return train, test


def test_make_submission(tmp_path):
    from src.predict import make_submission

    train, test = write_synthetic_data(tmp_path)
    out = tmp_path / "submission.parquet"
    submission = make_submission("tfidf_logreg", data_dir=tmp_path, out=out)

    written = pd.read_parquet(out)
    assert list(written.columns) == ["id", "model"]
    assert list(written["id"]) == list(test["id"])
    assert set(written["model"]) <= set(LABELS)
    # the test rows are training rows, so the model should get them right
    assert (submission["model"].to_numpy() == train.tail(18)["model"].to_numpy()).mean() > 0.9


def test_make_submission_with_assignment(tmp_path):
    from src.predict import make_submission

    train, test = write_synthetic_data(tmp_path)
    submission = make_submission(
        "tfidf_style_blend", assign=True, data_dir=tmp_path, out=tmp_path / "s.parquet"
    )
    # the 18 test rows answer one prompt, so every model must appear exactly once
    assert sorted(submission["model"]) == sorted(LABELS)


def test_style_features_handle_odd_text():
    from src.features.stylometry import FEATURE_NAMES, style_features

    features = style_features(["", "word", "Two sentences here. And a second one!\n\n- a list item"])
    assert features.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    names = dict(zip(FEATURE_NAMES, features[2]))
    # two sentences, then the list item counts as a third segment
    assert names["n_sentences"] == 3
    assert names["n_list_items"] == 1
    assert names["blank_line_per_100_chars"] > 0


def test_every_model_gives_probabilities():
    from src.models import MODELS

    df = synthetic_train(n_prompts=6)
    for name, build in MODELS.items():
        probs = build().fit(df["text"], df["label"]).predict_proba(df["text"][:5])
        assert probs.shape == (5, 18), name
        assert np.allclose(probs.sum(axis=1), 1.0), name


def test_assignment_never_repeats_a_class_in_a_group():
    from src.prompts import assign_within_groups

    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(18), size=36)
    groups = np.repeat([0, 1], 18)
    predicted = assign_within_groups(probs, groups)
    assert sorted(predicted[:18]) == list(range(18))
    assert sorted(predicted[18:]) == list(range(18))


def test_assignment_fixes_a_repeated_guess():
    from src.prompts import assign_within_groups

    # both passages look most like class 0, but the second is nearly as likely class 1
    probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.45, 0.05]])
    assert list(probs.argmax(axis=1)) == [0, 0]
    assert list(assign_within_groups(probs, [7, 7])) == [0, 1]
    # in separate groups nothing changes
    assert list(assign_within_groups(probs, [1, 2])) == [0, 0]


def test_assignment_allows_repeats_in_oversized_groups():
    from src.prompts import assign_within_groups

    probs = np.tile([[0.8, 0.1, 0.1]], (6, 1))
    predicted = assign_within_groups(probs, np.zeros(6, dtype=int))
    # 6 passages and 3 classes: each class is used exactly twice
    assert sorted(predicted) == [0, 0, 1, 1, 2, 2]


def test_cluster_prompts_groups_passages_by_topic():
    from src.prompts import cluster_prompts

    topics = {
        "physics": "quantum electron photon energy momentum particle wave",
        "biology": "cell protein enzyme membrane gene organism tissue",
        "poetry": "moonlight river sorrow autumn whisper lantern",
    }
    rng = np.random.default_rng(0)
    texts, truth = [], []
    for topic, vocabulary in topics.items():
        for _ in range(6):
            words = list(rng.choice(vocabulary.split(), size=12)) + ["the", "of", "and"]
            texts.append(" ".join(words))
            truth.append(topic)
    groups = cluster_prompts(texts)
    assert len(set(groups)) == 3
    assert (pd.Series(groups).groupby(truth).nunique() == 1).all()
    assert list(cluster_prompts(["only one passage"])) == [0]


def test_notebooks_are_up_to_date():
    """The committed notebooks must match src/; rebuild them after changing src/."""
    from src.build_notebook import NOTEBOOKS, ROOT, render

    for name in NOTEBOOKS:
        assert (ROOT / "notebooks" / name).read_text() == render(name), (
            "run: python -m src.build_notebook"
        )
