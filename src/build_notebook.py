"""Build the Kaggle submission notebooks from the code in src/.

    python -m src.build_notebook

Kaggle runs submissions with internet off, so a notebook can't clone this
repo. Instead each source file becomes a `%%writefile` cell that recreates it
inside the notebook, and the last cell calls src.predict.make_submission.
Never edit the notebooks by hand: change src/ and rebuild.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only what a submission needs, in import order.
SOURCES = [
    "src/__init__.py",
    "src/data.py",
    "src/evaluate.py",
    "src/features/__init__.py",
    "src/features/tfidf.py",
    "src/features/stylometry.py",
    "src/features/llm.py",
    "src/models/linear.py",
    "src/models/blend.py",
    "src/models/transformer.py",
    "src/pool.py",
    "src/models/pool_model.py",
    "src/models/__init__.py",
    "src/prompts.py",
    "src/predict.py",
    "src/llm_experiment.py",
    "src/pool_experiment.py",
    "src/pool_encoder.py",
]

# notebook file name -> (title, arguments of make_submission)
NOTEBOOKS = {
    "submission_tfidf_logreg.ipynb": (
        "TF-IDF + logistic regression",
        {"model": "tfidf_logreg", "assign": False},
    ),
    "submission_blend.ipynb": (
        "TF-IDF + style features",
        {"model": "tfidf_style_blend", "assign": False},
    ),
    "submission_blend_assign.ipynb": (
        "TF-IDF + style features, one model per prompt",
        {"model": "tfidf_style_blend", "assign": True},
    ),
    # Needs a GPU session and deberta-v3-base attached as an input.
    "submission_deberta_blend.ipynb": (
        "TF-IDF + style features + fine-tuned DeBERTa",
        {"model": "tfidf_style_deberta_blend", "assign": False},
    ),
    # Needs a GPU session, deberta-v3-base and an open language model (Qwen 3 4B) attached.
    "submission_full_blend.ipynb": (
        "TF-IDF + style features + fine-tuned DeBERTa + language-model features",
        {"model": "full_blend", "assign": False},
    ),
    # As the full blend, plus the open pool dataset attached (crtproduction/name-that-llm-open-pool).
    "submission_pool_blend.ipynb": (
        "the full blend with TF-IDF trained on the public open pool",
        {"model": "pool_blend", "assign": False},
    ),
    # Needs a GPU session, Qwen 3 4B, the open pool dataset, and the output of the
    # ntl-train-deberta-pool notebook (train_deberta_pool.ipynb) attached.
    "submission_pool_deberta_blend.ipynb": (
        "TF-IDF and DeBERTa trained on the public open pool, with style and language-model features",
        {"model": "pool_deberta_blend", "assign": False},
    ),
    # Needs a GPU session, Qwen 3 4B, the open pool dataset, and the outputs of both
    # ntl-train-deberta-pool-long and ntl-train-deberta-pool-big attached.
    "submission_pool_two_deberta_blend.ipynb": (
        "TF-IDF and two DeBERTas trained on the public open pool, with style and language-model features",
        {"model": "pool_two_deberta_blend", "assign": False},
    ),
    # As above, plus the output of ntl-train-qwen-lora-pool attached.
    "submission_pool_qwen_lora_blend.ipynb": (
        "TF-IDF, two DeBERTas and a LoRA-tuned Qwen trained on the public open pool, with style "
        "and language-model features",
        {"model": "pool_qwen_lora_blend", "assign": False},
    ),
    # A probe, not a contender: the LoRA-tuned Qwen alone (0.547 on train.csv), to see whether
    # it holds up on the test prompts. Needs a GPU session, Qwen 3 4B and the output of
    # ntl-train-qwen-lora-pool attached; about 10 minutes.
    # The 0.581 blend plus the LoRA Qwen at weight 0.5; the same inputs as the blend above.
    "submission_pool_qwen_lora_light_blend.ipynb": (
        "the two-DeBERTa blend with the LoRA-tuned Qwen at a low weight",
        {"model": "pool_qwen_lora_light_blend", "assign": False},
    ),
    "submission_qwen_lora.ipynb": (
        "the LoRA-tuned Qwen alone",
        {"model": "qwen_lora_pool", "assign": False},
    ),
}

# Notebooks that run an experiment instead of writing a submission:
# file name -> (title, what it does, code of the last cell)
EXPERIMENTS = {
    # Needs a GPU session, deberta-v3-base and the open pool dataset attached.
    # Its output (the trained model) is an input of later submission notebooks.
    "train_deberta_pool.ipynb": (
        "DeBERTa trained on the public open pool",
        "then fine-tune DeBERTa-v3-base on the public open pool, save it, and score it on "
        "`train.csv`. This trains a model for later submissions: do not submit it.",
        "from src.pool_encoder import main\n\nmain()",
    ),
    # The same with about twice the passages and 3 epochs instead of 2: about 2.5 hours on a
    # T4, against 51 minutes for the first run, whose loss was still falling.
    "train_deberta_pool_long.ipynb": (
        "DeBERTa trained longer on the public open pool",
        "then fine-tune DeBERTa-v3-base on up to 8,000 open-pool passages per model (about "
        "136,000) for 3 epochs, save it, and score it on `train.csv`. This trains a model for "
        "later submissions: do not submit it.",
        'from src.pool_encoder import main\n\nmain(name="deberta_pool_long", per_label=8000, epochs=3)',
    ),
    # Up to 12,000 passages per model (about 192,000) for 3 epochs, about 3.2 hours on a T4.
    # The loss is weighted per model: the smallest two have only 3,500 and 4,600 passages.
    "train_deberta_pool_big.ipynb": (
        "DeBERTa trained on most of the public open pool",
        "then fine-tune DeBERTa-v3-base on up to 12,000 open-pool passages per model (about "
        "192,000) for 3 epochs, with the loss weighted so every model counts the same, save "
        "it, and score it on `train.csv`. This trains a model for later submissions: do not "
        "submit it.",
        'from src.pool_encoder import main\n\n'
        'main(name="deberta_pool_big", per_label=12000, epochs=3, balanced=True)',
    ),
    # Needs a GPU session, Qwen 3 4B (Models: qwen-lm/qwen-3, transformers 4b) and the open
    # pool dataset attached; not deberta-v3-base. One epoch over about 71,500 passages; the
    # log prints the time left every 100 steps, and the run stops at 8 hours and still saves.
    "train_qwen_lora_pool.ipynb": (
        "Qwen 3 4B fine-tuned with LoRA on the public open pool",
        "then fine-tune the first 18 layers of Qwen 3 4B with LoRA on up to 4,000 open-pool "
        "passages per model (about 71,500), save the LoRA weights, and score it on "
        "`train.csv`. This trains a model for later submissions: do not submit it.",
        'from src.pool_encoder import build_pool_qwen_lora, main\n\n'
        'main(name="qwen_lora_pool", build=build_pool_qwen_lora)',
    ),
    # CPU is enough. Needs the open pool dataset attached (crtproduction/name-that-llm-open-pool).
    "experiment_open_pool.ipynb": (
        "training on the public open pool",
        "then train the TF-IDF and style models on the public open pool of extra passages "
        "and score them on `train.csv`. This is an experiment: do not submit it.",
        "from src.pool_experiment import main\n\ntable = main()\n"
        'table.pivot(index="variant", columns="model", values="score").round(4)',
    ),
    # Needs a GPU session and an open language model (e.g. Qwen 3) attached.
    "experiment_llm_features.ipynb": (
        "features from a frozen language model",
        "then extract language-model features for `train.csv` and cross-validate them. "
        "This is an experiment: do not submit it.",
        "from src.llm_experiment import main\n\ntable = main()\n"
        'table.sort_values("cv_mean", ascending=False)',
    ),
}

INTRO = """# Name That LLM: {title}

Generated by `python -m src.build_notebook` from
https://github.com/Dana-Lev/name-that-llm. The cells below recreate the `src/`
package, {what}"""

SUBMISSION = "then fit on `train.csv` and write `submission.parquet`."

RUN = """from src.predict import make_submission

submission = make_submission(model={model!r}, assign={assign!r})
submission["model"].value_counts()"""


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def build(title, what, last_cell):
    cells = [cell("markdown", INTRO.format(title=title, what=what))]
    texts = {source: (ROOT / source).read_text() for source in SOURCES}
    # %%writefile refuses an empty body, so the setup cell creates empty files.
    empty = [source for source in SOURCES if not texts[source].strip()]
    folders = sorted({str(Path(source).parent) for source in SOURCES})
    setup = (
        "from pathlib import Path\n\n"
        f"for folder in {folders!r}:\n"
        "    Path(folder).mkdir(parents=True, exist_ok=True)\n"
        f"for empty_file in {empty!r}:\n"
        "    Path(empty_file).touch()"
    )
    cells.append(cell("code", setup))
    for source in SOURCES:
        if source not in empty:
            cells.append(cell("code", f"%%writefile {source}\n{texts[source]}"))
    cells.append(cell("code", last_cell))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def render(name):
    if name in EXPERIMENTS:
        notebook = build(*EXPERIMENTS[name])
    else:
        title, arguments = NOTEBOOKS[name]
        notebook = build(title, SUBMISSION, RUN.format(**arguments))
    return json.dumps(notebook, indent=1) + "\n"


if __name__ == "__main__":
    for name in [*NOTEBOOKS, *EXPERIMENTS]:
        (ROOT / "notebooks" / name).write_text(render(name))
        print(f"wrote notebooks/{name}")
