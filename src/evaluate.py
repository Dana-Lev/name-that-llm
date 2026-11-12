"""Scoring: balanced accuracy (the competition metric) and confusion tables."""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.data import LAB_OF, LABELS


def balanced_accuracy(y_true, y_pred):
    """Mean per-class recall over the classes present in y_true."""
    return balanced_accuracy_score(y_true, y_pred)


def fold_scores(y_true, y_pred, folds):
    """Balanced accuracy of each validation fold."""
    return np.array(
        [balanced_accuracy(y_true[folds == f], y_pred[folds == f]) for f in np.unique(folds)]
    )


def model_confusion(y_true, y_pred):
    """18x18 table: rows are the true model, columns the predicted model."""
    ids = np.arange(len(LABELS))
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=ids), index=LABELS, columns=LABELS)


def lab_confusion(y_true, y_pred):
    """The same table collapsed to the 6 labs."""
    table = model_confusion(y_true, y_pred)
    # An array, not a list: pandas reads a list of strings as column names.
    labs = np.array([LAB_OF[m] for m in LABELS])
    return table.groupby(labs, sort=False).sum().T.groupby(labs, sort=False).sum().T


def within_lab_error_share(y_true, y_pred):
    """Fraction of the mistakes where the predicted model is from the true lab."""
    table = lab_confusion(y_true, y_pred)
    errors = len(y_true) - int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
    if errors == 0:
        return 0.0
    model_table = model_confusion(y_true, y_pred).to_numpy()
    same_lab_total = int(np.trace(table.to_numpy()))
    same_lab_errors = same_lab_total - int(np.trace(model_table))
    return same_lab_errors / errors
