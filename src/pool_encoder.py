"""Fine-tune DeBERTa-v3-base on the public open pool and save it.

Run on Kaggle with a GPU, the deberta-v3-base weights and the open pool
attached:

    from src.pool_encoder import main; main()

The model learns from pool passages only, never from train.csv, so scoring it
on all of train.csv is a held-out test (the pool has no answers to
train.csv's prompts). Outputs:

- <name>/: the trained model. Attach this notebook's output to a
  submission notebook, where SavedEncoder loads it instead of training.
- probs_<name>.npy: class probabilities for every train.csv passage, for
  choosing blend weights locally.
- <name>.csv: the score.

`name` is deberta_pool by default. build=build_pool_qwen_lora trains a LoRA
on Qwen 3 4B instead (attach Qwen 3 instead of deberta-v3-base).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.data import LABELS, find_data_dir, load_train
from src.models.transformer import FineTunedEncoder
from src.pool import load_pool, sample_pool

PER_LABEL = 4000  # about 70,000 passages; the smallest class has about 3,500


def build_pool_encoder(**kwargs):
    settings = dict(
        epochs=2,
        lr=3e-5,
        batch_size=16,
        max_length=384,  # the longest train.csv passage is 351 tokens
        fast=True,
        verbose=True,
        progress_every=250,
        time_limit_hours=9,  # Kaggle stops a run at 12 hours
    )
    settings.update(kwargs)
    return FineTunedEncoder("microsoft/deberta-v3-base", **settings)


def build_pool_qwen_lora(**kwargs):
    """Qwen 3 4B, first 18 of its 36 layers, with a rank-16 LoRA on every linear layer.

    One epoch: each step costs far more than with DeBERTa. The loss and the
    time left are printed every 100 steps; if the run would pass
    time_limit_hours it stops there and still saves.
    """
    from src.features.llm import find_llm

    settings = dict(
        epochs=1,
        lr=2e-4,
        head_lr_factor=5,
        batch_size=16,
        max_length=384,
        fast=True,
        verbose=True,
        progress_every=100,
        time_limit_hours=8,
        balanced=True,
        lora_rank=16,
        lora_alpha=32,
        keep_layers=18,
        gradient_checkpointing=True,
    )
    settings.update(kwargs)
    pretrained = settings.pop("pretrained", None) or find_llm()
    return FineTunedEncoder(pretrained, **settings)


def main(out_dir=".", per_label=PER_LABEL, matched=True, name="deberta_pool", build=None, **encoder_settings):
    out_dir = Path(out_dir)
    df = load_train(find_data_dir())
    sample = sample_pool(load_pool(), per_label, matched)
    print(f"training on {len(sample)} pool passages; testing on {len(df)} train.csv passages", flush=True)

    started = time.time()
    model = (build or build_pool_encoder)(**encoder_settings).fit(sample["text"], sample["label"])
    minutes = (time.time() - started) / 60
    model.save(out_dir / name)  # before scoring, so the weights survive a later error
    print(f"trained in {minutes:.0f} min; saved to {out_dir / name}", flush=True)

    probs = np.zeros((len(df), len(LABELS)))
    probs[:, model.classes_] = model.predict_proba(df["text"])
    np.save(out_dir / f"probs_{name}.npy", probs)
    score = balanced_accuracy_score(df["label"], probs.argmax(axis=1))
    print(f"balanced accuracy on train.csv: {score:.4f}", flush=True)

    table = pd.DataFrame(
        [{"model": name, "n_train": len(sample), "minutes": round(minutes), "score": score}]
    )
    table.to_csv(out_dir / f"{name}.csv", index=False)
    return table
