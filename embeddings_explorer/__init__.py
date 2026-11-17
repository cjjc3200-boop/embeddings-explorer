"""Embeddings Explorer — a small toolkit for seeing what embeddings are."""
from .embedder import embed, dim, MODEL_NAME
from .similarity import cosine_matrix, nearest, analogy

__all__ = ["embed", "dim", "MODEL_NAME", "cosine_matrix", "nearest", "analogy"]
