"""Offline tests for the pure-maths parts (no model download needed).

Run with:  python -m pytest   (or: python test_similarity.py)
"""
from __future__ import annotations

import numpy as np

from embeddings_explorer.similarity import cosine_matrix, nearest
from embeddings_explorer.plot import pca_2d


def _unit(*rows):
    v = np.array(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_cosine_matrix_diagonal_is_one():
    v = _unit([1, 0, 0], [0, 1, 0], [0.7, 0.7, 0])
    sims = cosine_matrix(v)
    assert np.allclose(np.diag(sims), 1.0)
    assert np.allclose(sims, sims.T)  # symmetric


def test_nearest_orders_by_similarity():
    labels = ["east", "north", "almost_east"]
    v = _unit([1, 0], [0, 1], [0.99, 0.1])
    top = nearest(v[0], v, labels, k=2)
    assert top[0][0] == "east"       # a vector is nearest to itself
    assert top[1][0] == "almost_east"
    assert top[0][1] > top[1][1]


def test_nearest_respects_k():
    v = _unit([1, 0], [0, 1], [-1, 0])
    assert len(nearest(v[0], v, ["a", "b", "c"], k=2)) == 2


def test_pca_2d_shape_and_centering():
    v = _unit([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1])
    pts = pca_2d(v)
    assert pts.shape == (4, 2)
    # PCA is centred, so the projection should be centred too.
    assert np.allclose(pts.mean(axis=0), 0, atol=1e-5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall tests passed")
