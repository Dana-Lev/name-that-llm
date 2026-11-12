"""Hand-made writing-style features: sentence shape, punctuation habits, formatting.

TF-IDF sees which words and character sequences appear. These features add what
it can't see: how long and how varied the sentences are, how the passage starts
and ends, and how often each punctuation mark is used per 100 characters.
"""

import re

import numpy as np

PUNCTUATION = {
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "em_dash": "—",
    "en_dash": "–",
    "hyphen": "-",
    "open_paren": "(",
    "close_paren": ")",
    "straight_double_quote": '"',
    "straight_single_quote": "'",
    "curly_apostrophe": "’",
    "curly_open_quote": "“",
    "question": "?",
    "exclamation": "!",
    "ellipsis_char": "…",
    "three_dots": "...",
    "asterisk": "*",
    "backtick": "`",
    "hash": "#",
    "dollar": "$",
    "backslash": "\\",
    "slash": "/",
    "percent": "%",
    "equals": "=",
    "newline": "\n",
    "blank_line": "\n\n",
    "double_space": "  ",
}

SHAPE_NAMES = [
    "n_chars",
    "n_words",
    "n_sentences",
    "sentence_len_mean",
    "sentence_len_std",
    "sentence_len_max",
    "sentence_len_min",
    "word_len_mean",
    "word_len_std",
    "long_word_share",
    "distinct_word_share",
    "uppercase_share",
    "digit_share",
    "non_ascii_share",
    "starts_uppercase",
    "ends_with_sentence_mark",
    "ends_with_period",
    "pronoun_share",
    "n_list_items",
    "n_bold_spans",
    "n_latex_spans",
]
FEATURE_NAMES = SHAPE_NAMES + [f"{name}_per_100_chars" for name in PUNCTUATION]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PRONOUN = re.compile(r"\b(?:I|we|you)\b")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", flags=re.M)
_BOLD = re.compile(r"\*\*[^*]+\*\*")
_LATEX = re.compile(r"\$[^$]+\$")


def style_vector(text):
    """The style features of one passage, in FEATURE_NAMES order."""
    words = text.split()
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    sentence_lens = np.array([len(s.split()) for s in sentences] or [0])
    word_lens = np.array([len(w) for w in words] or [0])
    n_chars = max(len(text), 1)
    n_words = max(len(words), 1)
    last = text.rstrip()[-1:]

    shape = [
        len(text),
        len(words),
        len(sentences),
        sentence_lens.mean(),
        sentence_lens.std(),
        sentence_lens.max(),
        sentence_lens.min(),
        word_lens.mean(),
        word_lens.std(),
        (word_lens > 8).mean(),
        len({w.lower() for w in words}) / n_words,
        sum(c.isupper() for c in text) / n_chars,
        sum(c.isdigit() for c in text) / n_chars,
        sum(ord(c) > 127 for c in text) / n_chars,
        float(text[:1].isupper()),
        float(last in ".!?" and last != ""),
        float(last == "."),
        len(_PRONOUN.findall(text)) / n_words,
        len(_LIST_ITEM.findall(text)),
        len(_BOLD.findall(text)),
        len(_LATEX.findall(text)),
    ]
    punctuation = [text.count(mark) / n_chars * 100 for mark in PUNCTUATION.values()]
    return shape + punctuation


def style_features(texts):
    """Matrix of style features, one row per passage."""
    return np.array([style_vector(text) for text in texts], dtype=np.float64)
