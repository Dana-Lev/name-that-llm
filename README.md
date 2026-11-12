# Name That LLM

Our entry for the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm):
given a short passage (40–250 words), predict which of 18 language models wrote it. The metric is
balanced accuracy.

This is a project for the Workshop in Machine Learning and Data Analysis (0368-3548) at Tel Aviv
University, Fall 2026–27. The schedule and roles are in [PLAN.md](PLAN.md); the proposal is in
[docs/](docs/).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the competition files from the
[Data page](https://www.kaggle.com/competitions/name-that-llm/data) (you must accept the
competition rules first) and put `train.csv` and `test.csv` in `data/`. The data is not committed.

## Run

```bash
python -m pytest tests                              # checks the pipeline on synthetic data
python -m src.train --model tfidf_style_blend --owner <your name>
```

`src.train` cross-validates one model from `src/models/__init__.py`. It prints balanced accuracy per
fold, saves out-of-fold probabilities to `oof/`, and appends rows to `experiments.csv`.

## Method so far

1. **TF-IDF model.** Word and character n-gram counts into logistic regression.
2. **Style model.** 48 hand-made features (sentence lengths, punctuation rates, formatting) into
   logistic regression.
3. **Blend.** The two models' log-probabilities are added, with the style model at half weight.
4. **One model per prompt.** In `train.csv` no prompt has two passages from the same model. We
   group the test passages by topic to recover the prompts, then choose the labelling of each group
   in which no model repeats (`src/prompts.py`). This assumes the hidden test set is built like the
   training set, which only the leaderboard can confirm.

| Model | Plain | With one model per prompt |
|---|---|---|
| Style features only | 0.282 | 0.334 |
| TF-IDF only | 0.343 | 0.450 |
| Blend of both | 0.384 | 0.496 |

Balanced accuracy, mean of 5 prompt-grouped folds; the standard deviation across folds is 0.01–0.03.
Random guessing scores 0.056.

## Submit

Kaggle runs submissions with internet off, so each submission notebook carries a copy of the code
it needs. The notebooks are generated from `src/`; never edit them by hand.

```bash
python -m src.build_notebook                        # rewrites the notebooks in notebooks/
```

| Notebook | What it submits |
|---|---|
| `submission_tfidf_logreg.ipynb` | TF-IDF model |
| `submission_blend.ipynb` | Blend |
| `submission_blend_assign.ipynb` | Blend with one model per prompt |

Upload a notebook to the competition on Kaggle, attach the competition data, keep internet off,
save a version and submit it. `python -m src.predict --model tfidf_style_blend --assign` does the
same locally.

## Validation

Every experiment uses the same 5 folds from `src.data.make_folds`. All passages that answer one
prompt stay in one fold, because the hidden test set uses prompts that are not in the training set.
Scores are reported as mean ± std of balanced accuracy across folds.

## Layout

```
src/
  data.py                 labels, loading, folds
  evaluate.py             balanced accuracy, model and lab confusion tables
  features/tfidf.py       word and character n-gram features
  features/stylometry.py  hand-made style features
  models/linear.py        logistic regression on TF-IDF and on style features
  models/blend.py         adds the log-probabilities of several models
  models/__init__.py      the list of models by name
  prompts.py              groups passages by prompt; one model per prompt
  train.py                cross-validation entry point
  predict.py              fit on all data, write submission.parquet
  build_notebook.py       generates the Kaggle submission notebooks
tests/                    pipeline tests on synthetic data
notebooks/                the generated Kaggle submission notebooks
configs/                  one file per experiment (to come)
docs/                     proposal
experiments.csv           one row per experiment
```
