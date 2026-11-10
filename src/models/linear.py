"""Linear baseline: TF-IDF features into multinomial logistic regression."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.tfidf import build_tfidf


def build_tfidf_logreg(C=10.0, max_iter=2000, **tfidf_kwargs):
    """class_weight="balanced" matches the metric, which weights every model equally."""
    return Pipeline(
        [
            ("tfidf", build_tfidf(**tfidf_kwargs)),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")),
        ]
    )
