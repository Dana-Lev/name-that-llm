"""Every model that can be cross-validated or submitted, by name."""

from src.models.blend import LogProbBlend
from src.models.linear import build_llm_logreg, build_style_logreg, build_tfidf_logreg
from src.models.pool_model import build_pool_tfidf_logreg
from src.models.transformer import SavedEncoder, build_deberta_base, build_deberta_small


def build_tfidf_style_blend(style_weight=0.5):
    return LogProbBlend([(build_tfidf_logreg, 1.0), (build_style_logreg, style_weight)])


def build_tfidf_style_deberta_blend(style_weight=0.5, deberta_weight=0.5):
    return LogProbBlend(
        [
            (build_tfidf_logreg, 1.0),
            (build_style_logreg, style_weight),
            (build_deberta_base, deberta_weight),
        ]
    )


def build_full_blend(style_weight=0.5, deberta_weight=0.5, llm_weight=0.5):
    return LogProbBlend(
        [
            (build_tfidf_logreg, 1.0),
            (build_style_logreg, style_weight),
            (build_deberta_base, deberta_weight),
            (build_llm_logreg, llm_weight),
        ]
    )


def build_pool_blend(style_weight=0.5, deberta_weight=0.5, llm_weight=0.5):
    """The full blend with TF-IDF trained on train.csv plus the open pool."""
    return LogProbBlend(
        [
            (build_pool_tfidf_logreg, 1.0),
            (build_style_logreg, style_weight),
            (build_deberta_base, deberta_weight),
            (build_llm_logreg, llm_weight),
        ]
    )


def build_pool_deberta_blend(style_weight=0.5, deberta_weight=1.0, llm_weight=0.5):
    """TF-IDF and DeBERTa both trained on the open pool.

    The DeBERTa was trained in train_deberta_pool.ipynb and is only loaded
    here, so that notebook's output must be attached. Weights from
    src.blend_oof: the best ones lie on a flat plateau (0.573-0.576), and the
    DeBERTa trained on train.csv adds nothing once this one is in.
    """
    return LogProbBlend(
        [
            (build_pool_tfidf_logreg, 1.0),
            (build_style_logreg, style_weight),
            (SavedEncoder, deberta_weight),
            (build_llm_logreg, llm_weight),
        ]
    )


def load_deberta_pool_long():
    """DeBERTa trained on 136,000 pool passages (train_deberta_pool_long.ipynb)."""
    return SavedEncoder("deberta_pool_long")


def load_deberta_pool_big():
    """DeBERTa trained on 194,000 pool passages with a per-model loss weight
    (train_deberta_pool_big.ipynb)."""
    return SavedEncoder("deberta_pool_big")


def build_pool_two_deberta_blend(style_weight=0.5, long_weight=1.0, big_weight=1.0, llm_weight=0.5):
    """Both pool DeBERTas, which make different mistakes: the big one,
    trained with a per-model loss weight, finds the rare gemini-3.7-flash far
    more often (recall 0.44 against 0.26) and gemini-3.8-flash less often.
    Each alone in the blend scores 0.578 and 0.586, the two together 0.592
    (src.blend_oof). Needs the outputs of both training notebooks attached.
    """
    return LogProbBlend(
        [
            (build_pool_tfidf_logreg, 1.0),
            (build_style_logreg, style_weight),
            (load_deberta_pool_long, long_weight),
            (load_deberta_pool_big, big_weight),
            (build_llm_logreg, llm_weight),
        ]
    )


MODELS = {
    "tfidf_logreg": build_tfidf_logreg,
    "style_logreg": build_style_logreg,
    "tfidf_style_blend": build_tfidf_style_blend,
    "deberta_small": build_deberta_small,
    "deberta_base": build_deberta_base,
    "tfidf_style_deberta_blend": build_tfidf_style_deberta_blend,
    "llm_logreg": build_llm_logreg,
    "full_blend": build_full_blend,
    "pool_tfidf_logreg": build_pool_tfidf_logreg,
    "pool_blend": build_pool_blend,
    "deberta_pool": SavedEncoder,
    "pool_deberta_blend": build_pool_deberta_blend,
    "pool_two_deberta_blend": build_pool_two_deberta_blend,
}

# Kaggle inputs each model needs besides the competition data (see
# src.predict.find_inputs): the open pool, an open language model, the
# deberta-v3-base weights, or an encoder saved by a train_deberta_pool*
# notebook ("encoder:<name>" for one of several attached).
INPUTS = {
    "deberta_small": ("deberta",),
    "deberta_base": ("deberta",),
    "tfidf_style_deberta_blend": ("deberta",),
    "llm_logreg": ("llm",),
    "full_blend": ("deberta", "llm"),
    "pool_tfidf_logreg": ("pool",),
    "pool_blend": ("pool", "deberta", "llm"),
    "deberta_pool": ("encoder",),
    "pool_deberta_blend": ("pool", "llm", "encoder"),
    "pool_two_deberta_blend": ("pool", "llm", "encoder:deberta_pool_long", "encoder:deberta_pool_big"),
}

# Models too heavy for a quick test: they fine-tune or run a pretrained
# network, or train on the open pool. Run them on Kaggle.
NEURAL = {
    "deberta_small",
    "deberta_base",
    "tfidf_style_deberta_blend",
    "llm_logreg",
    "full_blend",
    "pool_tfidf_logreg",
    "pool_blend",
    "deberta_pool",
    "pool_deberta_blend",
    "pool_two_deberta_blend",
}
