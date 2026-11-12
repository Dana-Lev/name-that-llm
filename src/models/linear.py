"""Linear models: logistic regression on TF-IDF features and on style features."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features.stylometry import style_features
from src.features.tfidf import build_tfidf


def build_tfidf_logreg(C=10.0, max_iter=3000, **tfidf_kwargs):
    """class_weight="balanced" matches the metric, which weights every model equally."""
    return Pipeline(
        [
            ("tfidf", build_tfidf(**tfidf_kwargs)),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")),
        ]
    )


def build_style_logreg(C=1.0, max_iter=3000):
    """The style features are on very different scales, so standardise them first."""
    return Pipeline(
        [
            ("style", FunctionTransformer(style_features)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")),
        ]
    )
