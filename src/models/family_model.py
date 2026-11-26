"""Family specialists: one model per maker, trained only on that maker's models.

Most of the blend's remaining errors are within a family: gemini-3.7-flash is
called gemini-3.8-flash or gemini-3.6-flash, the three GPT-5.6 variants are
called each other. On train.csv, resolving the GPT family perfectly would lift
the GLM blend from 0.626 to 0.717 and the Gemini family to 0.699, so how the
blend splits a family's probability among its members matters more than which
family it picks, which it mostly gets right.

A specialist learns only that split, from open-pool passages of the family's
models. `refine` keeps the blend's total probability for the family and
re-splits it among the members with the specialist's opinion added, so a
passage the blend gives to another maker is left alone.
"""

import numpy as np

from src.data import LABELS

EPS = 1e-12

# family -> label ids, in increasing order (the column order of a specialist's probabilities)
FAMILIES = {
    name: sorted(LABELS.index(label) for label in labels)
    for name, labels in {
        "gpt": ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"],
        "gemini": ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
        "claude": ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"],
    }.items()
}


def refine(probs, family, specialist, weight):
    """Re-split each passage's probability for `family` with the specialist's added at `weight`.

    probs: (n, 18) probabilities of the blend; specialist: (n, k) probabilities
    over the family's labels in FAMILIES order. The family's total stays the
    same, so only the choice among its members can change.
    """
    ids = FAMILIES[family]
    out = probs.copy()
    total = probs[:, ids].sum(axis=1, keepdims=True)
    logits = np.log(probs[:, ids] + EPS) + weight * np.log(specialist + EPS)
    logits -= logits.max(axis=1, keepdims=True)
    split = np.exp(logits)
    out[:, ids] = total * split / split.sum(axis=1, keepdims=True)
    return out
