"""TF-IDF features over word and character n-grams."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf(word_ngrams=(1, 2), char_ngrams=(2, 5), min_df=2):
    """Word n-grams capture vocabulary; character n-grams capture punctuation
    and formatting habits. Case is kept because capitalisation is a style cue."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngrams,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngrams,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=False,
    )
    return FeatureUnion([("word", word), ("char", char)])
