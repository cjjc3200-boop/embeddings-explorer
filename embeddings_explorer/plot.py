"""Project high-dimensional vectors down to a 2D map you can look at.

We use PCA (via numpy's SVD — no extra dependency) to find the two directions
of greatest variance and flatten every point onto them. This is a *shadow* of a
384-dimensional space: nearness on the map is meaningful, exact positions are not.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Return the (n, 2) projection of `vectors` onto their top-2 principal axes."""
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    # SVD: columns of `vt` are the principal directions, most variance first.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def save_map(vectors: np.ndarray, labels: list[str], path: str = "map.png") -> str:
    """Render a labelled 2D PCA scatter plot of `vectors` and save it to `path`."""
    points = pca_2d(vectors)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.scatter(points[:, 0], points[:, 1], s=55, color="#1F3864", zorder=3)
    for (x, y), label in zip(points, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_title("Embeddings Explorer — 2D PCA map", fontsize=12, color="#1F3864")
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#cccccc", linewidth=0.7)
    ax.axvline(0, color="#cccccc", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
