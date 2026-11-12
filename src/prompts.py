"""Using the structure of the data: every model answers each prompt at most once.

In train.csv no prompt has two passages from the same model. If the test set
is built the same way, the predictions for one prompt should all be different
models. The test set has no prompt_id, so we first group passages by topic
(passages answering one prompt share vocabulary), then pick the best labelling
of each group in which no model repeats.

Whether the hidden test set follows this structure is an assumption; the
leaderboard is the only way to check it.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

EPS = 1e-12


def cluster_prompts(texts, threshold=0.90):
    """Group passages that seem to answer the same prompt; returns a group id per passage.

    Average-linkage clustering on cosine distance between TF-IDF vectors with
    stop words removed, so only topic words count. Merging stops at `threshold`,
    so the number of prompts doesn't have to be known. 0.90 was chosen on
    train.csv, where the true prompts are known.
    """
    texts = list(texts)
    if len(texts) < 2:
        return np.zeros(len(texts), dtype=int)
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1)
    vectors = vectorizer.fit_transform(texts).toarray()
    # A passage with no topic words has a zero vector, which has no cosine distance.
    empty = ~vectors.any(axis=1)
    vectors[empty, 0] = 1.0
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
    )
    return clustering.fit_predict(vectors)


def assign_within_groups(probs, groups):
    """Label each group so that no class repeats, maximising total log-probability.

    This is the assignment problem, solved exactly by the Hungarian algorithm.
    A group larger than the number of classes can't be a single prompt (the
    clustering merged several), so there each class may be used once per
    merged prompt: ceil(group size / number of classes) times.
    """
    probs = np.asarray(probs)
    groups = np.asarray(groups)
    n_classes = probs.shape[1]
    log_probs = np.log(probs + EPS)
    predicted = np.zeros(len(probs), dtype=int)
    for group in np.unique(groups):
        rows = np.where(groups == group)[0]
        copies = int(np.ceil(len(rows) / n_classes))
        row_idx, col_idx = linear_sum_assignment(-np.tile(log_probs[rows], (1, copies)))
        predicted[rows[row_idx]] = col_idx % n_classes
    return predicted
