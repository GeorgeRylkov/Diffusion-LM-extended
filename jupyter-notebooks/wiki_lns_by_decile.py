"""
LNS by corpus-frequency decile for the Wikipedia top-5180 embeddings.

Uses:
  embeddings/wiki_bert_128d_top5180/frozen_vectors.tsv   (k-NN space A)
  embeddings/wiki_bert_128d_top5180/e2e_vectors.tsv      (k-NN space B)
  embeddings/wiki_bert_128d_top5180/frozen_metadata.tsv  (vocab)
  embeddings/wiki_bert_128d_filtered/token_frequencies.tsv (token -> count)

Produces charts/lns_wiki_by_frequency_decile.png, mirroring
charts/lns_by_frequency_decile.png for ROCStories.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

ROOT    = Path("/Users/grylkov/git/hse/Diffusion-LM")
A_VEC   = ROOT / "embeddings/wiki_bert_128d_top5180/frozen_vectors.tsv"
B_VEC   = ROOT / "embeddings/wiki_bert_128d_top5180/e2e_vectors.tsv"
TOK_FILE = ROOT / "embeddings/wiki_bert_128d_top5180/frozen_metadata.tsv"
FREQ_TSV = ROOT / "embeddings/wiki_bert_128d_filtered/token_frequencies.tsv"
OUT_PNG  = ROOT / "charts/lns_wiki_by_frequency_decile.png"

K_VALUES = (10, 50)


def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.name}  {arr.shape}  in {time.time()-t0:.1f}s", flush=True)
    return arr


def load_tokens(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text().splitlines()]


def load_frequencies(path: Path) -> dict[str, int]:
    freq: dict[str, int] = {}
    for ln in path.read_text().splitlines()[1:]:  # skip header
        parts = ln.split("\t")
        if len(parts) == 3:
            freq[parts[1]] = int(parts[2])
    return freq


def knn_sets(e: np.ndarray, k: int) -> np.ndarray:
    """Cosine k-NN indices, shape (V, k), excluding self."""
    v = e.shape[0]
    en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    sim = en @ en.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(sim, kth=v - 1 - k, axis=1)[:, v - k:].astype(np.int64)


def jaccard_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v, k = a.shape
    out = np.empty(v, dtype=np.float64)
    a = np.sort(a, axis=1); b = np.sort(b, axis=1)
    for i in range(v):
        inter = np.intersect1d(a[i], b[i], assume_unique=True).size
        out[i] = inter / (2 * k - inter)
    return out


def decile_means(freq: np.ndarray, lns: np.ndarray) -> tuple[list, list, list[tuple]]:
    mask = freq > 0
    nz_f = freq[mask]; nz_l = lns[mask]
    edges = np.quantile(nz_f, np.linspace(0, 1, 11))
    bi = np.clip(np.searchsorted(edges, nz_f, side="right") - 1, 0, 9)
    means, medians, ranges = [], [], []
    for d in range(10):
        sel = bi == d
        means.append(float(nz_l[sel].mean()) if sel.any() else np.nan)
        medians.append(float(np.median(nz_l[sel])) if sel.any() else np.nan)
        ranges.append((int(nz_f[sel].min()), int(nz_f[sel].max()))
                      if sel.any() else (0, 0))
    return means, medians, ranges


def main() -> None:
    print("Loading embeddings:")
    ea = load_vectors(A_VEC)
    eb = load_vectors(B_VEC)
    tokens = load_tokens(TOK_FILE)
    v = len(tokens)

    print("Loading frequencies:")
    freq_map = load_frequencies(FREQ_TSV)
    freq = np.array([freq_map.get(t, 0) for t in tokens], dtype=np.int64)
    print(f"  {(freq > 0).sum():,} / {v:,} tokens have non-zero count")

    lns_by_k: dict[int, np.ndarray] = {}
    for k in K_VALUES:
        print(f"\nComputing LNS k={k} (cosine) …")
        t0 = time.time()
        nn_a = knn_sets(ea, k)
        nn_b = knn_sets(eb, k)
        lns = jaccard_batch(nn_a, nn_b)
        lns_by_k[k] = lns
        rnd = (k*k / (v-1)) / (2*k - k*k/(v-1))
        print(f"  mean={lns.mean():.4f}  median={np.median(lns):.4f}  "
              f"random={rnd:.4f}  mean/random={lns.mean()/rnd:.1f}x  "
              f"[{time.time()-t0:.1f}s]")

        means, medians, ranges = decile_means(freq, lns)
        print(f"  {'decile':>6}  {'freq range':>20}  {'mean LNS':>9}  {'median LNS':>10}")
        for d in range(10):
            lo, hi = ranges[d]
            print(f"  D{d+1:<4d}  [{lo:>8,}, {hi:>9,}]  {means[d]:9.3f}  {medians[d]:10.3f}")

    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(6.6 * len(K_VALUES), 5.5),
                             sharey=False)
    colors = ["#4c72b0", "#dd8452"]

    for ax, k, color in zip(axes, K_VALUES, colors):
        lns = lns_by_k[k]
        means, medians, ranges = decile_means(freq, lns)
        xpos = np.arange(10)
        w = 0.38
        bars_m  = ax.bar(xpos - w/2, means,   w, color=color,
                         edgecolor="black", label="mean")
        bars_md = ax.bar(xpos + w/2, medians, w, color=color, alpha=0.55,
                         edgecolor="black", label="median")
        for b, val in zip(bars_m, means):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width()/2, val + 0.003,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"D{i+1}" for i in range(10)])
        ax.set_xlabel("frequency decile  (D1 = rarest, D10 = most frequent)")
        ax.set_ylabel("LNS(w)")
        ax.set_title(f"LNS by frequency decile  (k={k})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9, frameon=False, loc="upper left")

    fig.suptitle(
        "Per-token LNS by corpus-frequency decile  "
        "(Wikipedia, bert_frozen vs bert_e2e, cosine, V=5180)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"\nsaved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
