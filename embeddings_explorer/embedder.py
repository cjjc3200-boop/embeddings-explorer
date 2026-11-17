"""Embedding back-end — the only file that talks to the model.

Everything here is deliberately small and readable. The job of this module is
to turn a list of strings into an L2-normalised numpy array of vectors, so the
rest of the project can do plain linear algebra on them.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

# Small, fast, 384-dimensional. Swap for a bigger model to see clusters sharpen.
MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Load the model once and reuse it (downloading ~130 MB on first run)."""
    return TextEmbedding(model_name=MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """Embed `texts` into a (n, dim) array of L2-normalised float32 vectors.

    Normalising to unit length matters: with unit vectors, cosine similarity
    collapses to a plain dot product, so `embeddings @ query` is all we need.
    """
    if not texts:
        raise ValueError("embed() needs at least one string")

    raw = np.array(list(_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against a zero vector
    return raw / norms


def dim() -> int:
    """How many numbers per vector (384 for bge-small)."""
    return embed(["dimension probe"]).shape[1]
