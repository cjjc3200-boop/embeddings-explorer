#!/usr/bin/env python3
"""Embeddings Explorer CLI.

Examples:
    python explore.py                                  # similarity + neighbours
    python explore.py --plot                           # also save a 2D PCA map
    python explore.py --analogy king man woman         # vector arithmetic
    python explore.py --words dog puppy cat car engine # your own phrases
"""
from __future__ import annotations

import argparse

from embeddings_explorer.embedder import embed, dim, MODEL_NAME
from embeddings_explorer.similarity import cosine_matrix

# A small default set chosen to show three clear clusters:
# animals, vehicles, and finance.
DEFAULT_WORDS = [
    "dog", "puppy", "cat",           # animals
    "car", "truck", "engine",        # vehicles
    "bank", "money", "interest rate",  # finance
]


def print_matrix(words: list[str]) -> None:
    vectors = embed(words)
    sims = cosine_matrix(vectors)
    width = max(len(w) for w in words) + 1
    print(f"\nSimilarity matrix ({len(words)} phrases, dim={dim()}):\n")
    print(" " * width + "  ".join(f"{i+1:>5}" for i in range(len(words))))
    for i, word in enumerate(words):
        row = "  ".join(f"{sims[i][j]:5.2f}" for j in range(len(words)))
        print(f"{word:<{width}}{row}")
    print("\n  1-based column key: " + ", ".join(f"{i+1}={w}" for i, w in enumerate(words)))


def print_neighbours(words: list[str]) -> None:
    vectors = embed(words)

    print("\nNearest neighbours:\n")
    for i, word in enumerate(words):
        scores = vectors @ vectors[i]
        order = [j for j in scores.argsort()[::-1] if j != i][:3]
        trail = ", ".join(f"{words[j]} ({scores[j]:.2f})" for j in order)
        print(f"  {word:<16}->  {trail}")


def main() -> None:
    p = argparse.ArgumentParser(description="Explore text embeddings as geometry.")
    p.add_argument("--words", nargs="+", help="phrases to embed (default: a demo set)")
    p.add_argument("--plot", action="store_true", help="save a 2D PCA map to map.png")
    p.add_argument(
        "--analogy",
        nargs=3,
        metavar=("A", "B", "C"),
        help="solve A - B + C = ? (e.g. --analogy king man woman)",
    )
    p.add_argument("--top-k", type=int, default=3, help="neighbours to show (default 3)")
    args = p.parse_args()

    words = args.words or DEFAULT_WORDS

    print(f"Model: {MODEL_NAME}  ->  {dim()} numbers per phrase")

    if args.analogy:
        from embeddings_explorer.similarity import analogy

        a, b, c = args.analogy
        pool = list(dict.fromkeys(words + [a, b, c]))
        print(f"\nAnalogy:  {a} - {b} + {c}  = ?\n")
        for word, score in analogy(a, b, c, pool):
            print(f"  {word:<16} {score:.3f}")
        return

    print_matrix(words)
    print_neighbours(words)

    if args.plot:
        from embeddings_explorer.plot import save_map

        path = save_map(embed(words), words)
        print(f"\nSaved 2D map to {path}")


if __name__ == "__main__":
    main()
