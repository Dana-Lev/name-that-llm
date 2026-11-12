"""Loading the competition data and building the cross-validation folds."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# The 18 valid labels, grouped by lab. Order is fixed: column k of every
# probability matrix in this project refers to LABELS[k].
LABS = {
    "OpenAI": ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"],
    "Google": [
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    ],
    "Anthropic": [
        "claude-fable-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
    ],
    "Qwen": ["qwen3.8-max"],
    "ZAI": ["glm-5.3", "glm-5.3-flash"],
    "Moonshot AI": ["kimi-k3"],
}
LABELS = [model for models in LABS.values() for model in models]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
LAB_OF = {model: lab for lab, models in LABS.items() for model in models}

N_SPLITS = 5
SEED = 42


def find_data_dir():
    """The folder with train.csv: $NTL_DATA_DIR, Kaggle's input folder, or ./data."""
    candidates = [
        os.environ.get("NTL_DATA_DIR"),
        "/kaggle/input/competitions/name-that-llm",
        "/kaggle/input/name-that-llm",
        "data",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "train.csv").exists():
            return Path(candidate)
    raise FileNotFoundError("train.csv not found; set NTL_DATA_DIR or put the data in ./data")


def load_train(data_dir="data"):
    """Read train.csv and add an integer `label` column."""
    df = pd.read_csv(Path(data_dir) / "train.csv")
    unknown = set(df["model"]) - set(LABELS)
    if unknown:
        raise ValueError(f"train.csv has labels outside the 18 valid ones: {sorted(unknown)}")
    df["label"] = df["model"].map(LABEL_TO_ID)
    return df


def load_test(data_dir="data"):
    return pd.read_csv(Path(data_dir) / "test.csv")


def make_folds(df, n_splits=N_SPLITS, seed=SEED):
    """Assign each row a fold id in [0, n_splits).

    All passages answering the same prompt land in the same fold, so every
    validation prompt is unseen during training, as in the hidden test set.
    Folds are also stratified by model to keep the class mix similar.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(
        splitter.split(df["text"], df["label"], groups=df["prompt_id"])
    ):
        folds[valid_idx] = fold
    return folds
