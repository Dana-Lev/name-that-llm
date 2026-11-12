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
python -m src.train_baseline --data-dir data --owner <your name>
```

The baseline is TF-IDF over word and character n-grams with logistic regression. The script prints
balanced accuracy per fold, saves out-of-fold probabilities to `oof/`, and appends a row to
`experiments.csv`.

## Submit

Kaggle runs submissions with internet off, so the submission notebook carries a copy of the code it
needs. It is generated from `src/`; never edit it by hand.

```bash
python -m src.build_notebook                        # writes notebooks/submission_tfidf_logreg.ipynb
```

Upload the notebook to the competition on Kaggle, attach the competition data, keep internet off,
save a version and submit it. The notebook fits the baseline on all of `train.csv` and writes
`submission.parquet`. `python -m src.predict --data-dir data` does the same locally.

## Validation

Every experiment uses the same 5 folds from `src.data.make_folds`. All passages that answer one
prompt stay in one fold, because the hidden test set uses prompts that are not in the training set.
Scores are reported as mean ± std of balanced accuracy across folds.

## Layout

```
src/
  data.py            labels, loading, folds
  evaluate.py        balanced accuracy, model and lab confusion tables
  features/tfidf.py  word and character n-gram features
  models/linear.py   TF-IDF + logistic regression
  train_baseline.py  cross-validation entry point
  predict.py         fit on all data, write submission.parquet
  build_notebook.py  generates the Kaggle submission notebook
tests/               pipeline test on synthetic data
configs/             one file per experiment (to come)
notebooks/           the generated Kaggle submission notebook; EDA (to come)
docs/                proposal
experiments.csv      one row per experiment
```
