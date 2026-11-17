# Embeddings Explorer

A tiny, readable toolkit for **seeing what embeddings actually are**.

An embedding turns text into a list of numbers (a *vector*) whose direction
encodes meaning. This project takes a handful of words or phrases, embeds them
locally, and shows you the geometry:

- **Similarity matrix** — how close every phrase is to every other
- **Nearest neighbours** — what a phrase is closest to
- **Analogies** — `king - man + woman ≈ queen`
- **2D map** — the whole set projected down to a plane you can look at

Everything runs **on your machine** with [fastembed](https://github.com/qdrant/fastembed)
(ONNX-based, no PyTorch). No API key.

## The idea in one picture

Embeddings place words in a high-dimensional space where **related meanings sit
close together**. "dog" and "puppy" land near each other; "dog" and "interest
rate" land far apart. Similarity is measured by the **angle** between two
vectors (**cosine similarity**), not their length — so we L2-normalise every
vector first and cosine becomes a plain dot product.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

```bash
# 1. similarity matrix + nearest neighbours (no plot)
python explore.py

# 2. save a 2D PCA map to map.png
python explore.py --plot

# 3. a word analogy
python explore.py --analogy king man woman

# 4. your own phrases
python explore.py --words dog puppy cat car engine "interest rate"
```

## What you should notice

- **Related clusters, not synonyms.** The model groups *meaning*, so "car" and
  "engine" sit closer than "car" and "banana".
- **Direction, not distance.** Because vectors are normalised, a long document
  and a short phrase can still be "close" if they mean the same thing.
- **PCA is lossy.** The 2D map is a *shadow* of a 384-dimensional space — it
  flattens a lot. Nearby points on the map are meaningful; exact positions are
  not.

## The vocabulary, mapped to the code

| Term | Meaning | Where |
|------|---------|-------|
| **Vector** | ordered list of numbers; `bge-small-en-v1.5` gives **384** | `embedder.py` |
| **Cosine similarity** | angle between vectors, -1..1 | `similarity.py` |
| **L2 normalisation** | scaling each vector to length 1 so cosine = dot product | `embedder.py` |
| **Nearest neighbours** | the top-k closest phrases to a query | `similarity.py` |
| **Analogy** | `a - b + c` arithmetic in vector space | `similarity.py` |
| **PCA** | projecting 384-D down to 2-D to visualise | `plot.py` |

## Example output

```
Model: BAAI/bge-small-en-v1.5  ->  384 numbers per phrase

Nearest neighbours:
  dog             ->  puppy (0.87), cat (0.73), truck (0.69)
  car             ->  truck (0.74), engine (0.74), cat (0.69)
  bank            ->  money (0.76), interest rate (0.67), car (0.60)

Analogy:  king - man + woman  = ?
  queen            0.713
  throne           0.620
```

## Ideas to extend it

- Swap `bge-small` for a bigger model and watch clusters sharpen.
- Add a *clustering* view (k-means) to colour the map by group.
- Feed whole sentences instead of single words.
- Try a **cross-lingual** model and see if "dog" and "perro" land together.

## Tests

The pure-maths parts (similarity, nearest neighbours, PCA) are covered by
`test_similarity.py` and need no model download:

```bash
pip install pytest
pytest -q
```
