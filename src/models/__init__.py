"""Every model that can be cross-validated or submitted, by name."""

from src.models.blend import LogProbBlend
from src.models.linear import build_style_logreg, build_tfidf_logreg


def build_tfidf_style_blend(style_weight=0.5):
    return LogProbBlend([(build_tfidf_logreg, 1.0), (build_style_logreg, style_weight)])


MODELS = {
    "tfidf_logreg": build_tfidf_logreg,
    "style_logreg": build_style_logreg,
    "tfidf_style_blend": build_tfidf_style_blend,
}
