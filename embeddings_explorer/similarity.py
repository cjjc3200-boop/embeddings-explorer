"""Vector maths: similarity, nearest neighbours, and analogies.

All inputs are assumed to be the L2-normalised vectors from `embedder.embed`,
which is what makes the dot products below equal cosine similarities.
"""
from __future__ import annotations

import numpy as np


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the (n, n) cosine-similarity matrix of `vectors`.

    With normalised vectors this is just the Gram matrix `V @ V.T`.
    Diagonal entries are 1.0 (a vector is perfectly similar to itself).
    """
    return vectors @ vectors.T


def nearest(
    query: np.ndarray,
    corpus: np.ndarray,
    labels: list[str],
    k: int = 3,
) -> list[tuple[str, float]]:
    """Return the `k` corpus entries most similar to `query`, as (label, score)."""
    scores = corpus @ query  # cosine similarity, since both are normalised
    order = np.argsort(-scores)[:k]
    return [(labels[i], float(scores[i])) for i in order]


def analogy(a: str, b: str, c: str, words: list[str]) -> list[tuple[str, float]]:
    """Solve `a - b + c = ?` in vector space.

    The classic example: king - man + woman ~ queen. `words` is the vocabulary to
    search; the three operands must be in it. The operands themselves are
    excluded from the results.
    """
    from .embedder import embed

    vocab = embed(words)
    index = {w: i for i, w in enumerate(words)}
    for w in (a, b, c):
        if w not in index:
            raise KeyError(f"{w!r} is not in the word list; add it and retry")

    target = vocab[index[a]] - vocab[index[b]] + vocab[index[c]]
    target = target / (np.linalg.norm(target) or 1.0)

    scores = vocab @ target
    order = np.argsort(-scores)
    return [(words[i], float(scores[i])) for i in order if words[i] not in (a, b, c)][:5]
