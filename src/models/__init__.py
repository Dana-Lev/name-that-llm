"""Every model that can be cross-validated or submitted, by name."""

from src.models.blend import LogProbBlend
from src.models.linear import build_llm_logreg, build_style_logreg, build_tfidf_logreg
from src.models.transformer import build_deberta_base, build_deberta_small


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


MODELS = {
    "tfidf_logreg": build_tfidf_logreg,
    "style_logreg": build_style_logreg,
    "tfidf_style_blend": build_tfidf_style_blend,
    "deberta_small": build_deberta_small,
    "deberta_base": build_deberta_base,
    "tfidf_style_deberta_blend": build_tfidf_style_deberta_blend,
    "llm_logreg": build_llm_logreg,
    "full_blend": build_full_blend,
}

# Models that fine-tune a pretrained network: minutes per fit, need torch.
NEURAL = {"deberta_small", "deberta_base", "tfidf_style_deberta_blend", "llm_logreg", "full_blend"}
