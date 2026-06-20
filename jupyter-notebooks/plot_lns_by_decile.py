"""
Standalone 'LNS by frequency decile' plot -- just the lower-left panel of
charts/lns_metric_comparison.png, saved as its own figure.

Compares Euclidean vs cosine LNS at k=50, grouped by corpus-frequency decile.
No correlation annotations in the legend, per request.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM")
A_VEC = ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_vectors.tsv"
B_VEC = ROOT / "embeddings/bert_128d_filtered/e2e_vectors.tsv"
TOKEN_FILE = ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_metadata.tsv"
CORPUS = ROOT / "datasets/ROCstory/roc_train_corpus_bert.txt"

OUT_PNG = ROOT / "charts/lns_by_frequency_decile.png"
K = 50


def load_vectors(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32, delimiter="\t")


def load_tokens(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text().splitlines()]


def token_frequencies(tokens: list[str]) -> np.ndarray:
    cnt: Counter[str] = Counter()
    vocab = set(tokens)
    with CORPUS.open("r") as f:
        for line in f:
            for tok in line.split():
                if tok in vocab:
                    cnt[tok] += 1
    return np.array([cnt.get(t, 0) for t in tokens], dtype=np.int64)


def knn_sets(e: np.ndarray, k: int, metric: str) -> np.ndarray:
    v = e.shape[0]
    if metric == "euclidean":
        sq = (e * e).sum(axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (e @ e.T)
        np.fill_diagonal(d2, np.inf)
        idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
    elif metric == "cosine":
        en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        sim = en @ en.T
        np.fill_diagonal(sim, -np.inf)
        idx = np.argpartition(sim, kth=v - 1 - k, axis=1)[:, v - k:]
    else:
        raise ValueError(metric)
    return idx.astype(np.int64)


def jaccard_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v, k = a.shape
    out = np.empty(v, dtype=np.float64)
    a = np.sort(a, axis=1)
    b = np.sort(b, axis=1)
    for i in range(v):
        inter = np.intersect1d(a[i], b[i], assume_unique=True).size
        out[i] = inter / (2 * k - inter)
    return out


def main() -> None:
    print("Loading embeddings...")
    ea = load_vectors(A_VEC)
    eb = load_vectors(B_VEC)
    tokens = load_tokens(TOKEN_FILE)

    print(f"Computing k={K} neighbourhoods and LNS for Euclidean + cosine...")
    lns = {}
    for metric in ("euclidean", "cosine"):
        t0 = time.time()
        nn_a = knn_sets(ea, K, metric)
        nn_b = knn_sets(eb, K, metric)
        lns[metric] = jaccard_batch(nn_a, nn_b)
        print(f"  {metric}: mean={lns[metric].mean():.4f}  [{time.time()-t0:.1f}s]")

    freq = token_frequencies(tokens)
    mask = freq > 0
    nz_freq = freq[mask]
    edges = np.quantile(nz_freq, np.linspace(0, 1, 11))
    bi = np.clip(np.searchsorted(edges, nz_freq, side="right") - 1, 0, 9)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    xpos = np.arange(10)
    w = 0.38
    for i, metric in enumerate(("euclidean", "cosine")):
        means = []
        for d in range(10):
            sel = bi == d
            means.append(float(lns[metric][mask][sel].mean())
                         if sel.any() else np.nan)
        color = "#4c72b0" if metric == "euclidean" else "#dd8452"
        bars = ax.bar(xpos + (i - 0.5) * w, means, w, color=color,
                      edgecolor="black", label=metric)
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, m + 0.002,
                        f"{m:.2f}", ha="center", va="bottom",
                        fontsize=8, alpha=0.75)

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"D{i+1}" for i in range(10)])
    ax.set_xlabel("frequency decile  (D1 = rarest, D10 = most frequent)")
    ax.set_ylabel("mean LNS(w)")
    ax.set_title(f"LNS by frequency decile  (k={K})")
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
