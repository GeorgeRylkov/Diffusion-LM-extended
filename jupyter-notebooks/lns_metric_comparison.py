"""
Compare LNS under Euclidean vs cosine metric.

The Embedding Comparator paper uses Euclidean distance for NLP embeddings,
but for Diffusion-LM / modern BERT work cosine is the more common choice.
If the two metrics agree closely, we'd prefer cosine going forward as it
removes per-token norm as a confound. If they disagree, that disagreement
itself is informative (it means norm drift contributes to the apparent
local-neighbourhood change).

For each token w and k=50 we compute
    LNS_euc(w)  = J(k-NN_euc_frozen(w),  k-NN_euc_e2e(w))
    LNS_cos(w)  = J(k-NN_cos_frozen(w),  k-NN_cos_e2e(w))

and then summarise (mean, quantiles, per-decile) and plot them side by side.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM")
A_VEC = ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_vectors.tsv"
A_TOK = ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_metadata.tsv"
B_VEC = ROOT / "embeddings/bert_128d_filtered/e2e_vectors.tsv"
B_TOK = ROOT / "embeddings/bert_128d_filtered/e2e_metadata.tsv"
CORPUS = ROOT / "datasets/ROCstory/roc_train_corpus_bert.txt"
OUT_PNG = ROOT / "charts/lns_metric_comparison.png"
K = 50


# ---------- helpers --------------------------------------------------------- #

def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.parent.name}  {arr.shape}  in {time.time()-t0:.1f}s",
          flush=True)
    return arr


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
    """Indices of the k nearest non-self neighbours per row.  Shape (V, k)."""
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


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i, n = 0, len(x)
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + j + 1)
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a); rb = rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


# ---------- main ------------------------------------------------------------ #

def main() -> None:
    print("Loading embeddings:")
    ea = load_vectors(A_VEC)
    eb = load_vectors(B_VEC)
    ta = load_tokens(A_TOK); tb = load_tokens(B_TOK)
    assert ta == tb
    tokens = ta; v = len(tokens)

    norm_a = np.linalg.norm(ea, axis=1)
    norm_b = np.linalg.norm(eb, axis=1)
    print(f"\nnorms (frozen): mean={norm_a.mean():.3f}  std={norm_a.std():.3f}  "
          f"cv={norm_a.std()/norm_a.mean():.3f}")
    print(f"norms (e2e):    mean={norm_b.mean():.3f}  std={norm_b.std():.3f}  "
          f"cv={norm_b.std()/norm_b.mean():.3f}")

    lns = {}
    for metric in ("euclidean", "cosine"):
        print(f"\n[{metric}] computing k-NN and LNS (k={K})")
        t0 = time.time()
        nn_a = knn_sets(ea, K, metric)
        nn_b = knn_sets(eb, K, metric)
        l = jaccard_batch(nn_a, nn_b)
        lns[metric] = l
        print(f"  mean={l.mean():.4f}  median={np.median(l):.4f}  "
              f"std={l.std():.4f}  [{time.time()-t0:.1f}s]")
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            print(f"  p{int(q*100):02d} = {np.quantile(l, q):.4f}")

    freq = token_frequencies(tokens)
    log_freq = np.log10(np.maximum(freq, 1))

    # Agreement between the two metrics.
    rho = spearman(lns["euclidean"], lns["cosine"])
    pear = float(np.corrcoef(lns["euclidean"], lns["cosine"])[0, 1])
    print(f"\nAgreement between metrics:  Pearson={pear:+.3f}  Spearman={rho:+.3f}")

    # Frequency-LNS correlation under each metric.
    rho_f_e = spearman(lns["euclidean"], freq.astype(float))
    rho_f_c = spearman(lns["cosine"],    freq.astype(float))
    print(f"LNS vs log-freq Spearman:  euclidean={rho_f_e:+.3f}  "
          f"cosine={rho_f_c:+.3f}")

    # Per-token delta.
    delta = lns["cosine"] - lns["euclidean"]
    print(f"\ncosine - euclidean:  mean={delta.mean():+.4f}  "
          f"median={np.median(delta):+.4f}  "
          f"P(cos>euc)={float((delta>0).mean()):.3f}  "
          f"P(cos=euc)={float((delta==0).mean()):.3f}")
    for q in (0.01, 0.05, 0.5, 0.95, 0.99):
        print(f"  delta p{int(q*100):02d} = {np.quantile(delta, q):+.4f}")

    # ---- plot ---- #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: overlayed histograms of the two LNS distributions.
    ax = axes[0, 0]
    bins = np.linspace(0, max(lns["euclidean"].max(), lns["cosine"].max()) + 1e-3, 50)
    ax.hist(lns["euclidean"], bins=bins, histtype="step", lw=2,
            color="#4c72b0", label=f"Euclidean (mean={lns['euclidean'].mean():.3f})")
    ax.hist(lns["cosine"], bins=bins, histtype="step", lw=2,
            color="#dd8452", label=f"cosine    (mean={lns['cosine'].mean():.3f})")
    ax.set_xlabel(f"LNS(w)  (k={K})")
    ax.set_ylabel("# tokens")
    ax.set_title("Per-token LNS distribution\n"
                 "(Euclidean vs cosine neighbourhoods)")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)

    # Panel B: scatter Euclidean vs cosine LNS, with y=x diagonal.
    ax = axes[0, 1]
    sc = ax.scatter(lns["euclidean"], lns["cosine"], s=7, alpha=0.35,
                    c=log_freq, cmap="viridis", edgecolor="none")
    mx = float(max(lns["euclidean"].max(), lns["cosine"].max()))
    ax.plot([0, mx], [0, mx], color="black", lw=1, ls="--", label="y = x")
    ax.set_xlabel("LNS (Euclidean)")
    ax.set_ylabel("LNS (cosine)")
    ax.set_title(f"Per-token agreement\nSpearman rho = {rho:+.3f}  "
                 f"Pearson = {pear:+.3f}")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("log10(corpus frequency)")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel C: LNS by frequency decile, both metrics.
    ax = axes[1, 0]
    mask = freq > 0
    nz_freq = freq[mask]
    edges = np.quantile(nz_freq, np.linspace(0, 1, 11))
    bi = np.clip(np.searchsorted(edges, nz_freq, side="right") - 1, 0, 9)
    xpos = np.arange(10)
    w = 0.38
    for i, metric in enumerate(("euclidean", "cosine")):
        means = []
        for d in range(10):
            sel = bi == d
            means.append(float(lns[metric][mask][sel].mean())
                         if sel.any() else np.nan)
        color = "#4c72b0" if metric == "euclidean" else "#dd8452"
        ax.bar(xpos + (i - 0.5) * w, means, w, color=color,
               edgecolor="black", label=metric)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"D{i+1}" for i in range(10)])
    ax.set_xlabel("frequency decile (D1 = rarest, D10 = most frequent)")
    ax.set_ylabel("mean LNS(w)")
    ax.set_title(f"LNS by frequency decile  (k={K})")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel D: tokens where the two metrics disagree the most.
    ax = axes[1, 1]
    # Largest positive delta = cosine says "preserved" but Euclidean says "rebuilt".
    biggest = np.argsort(np.abs(delta))[-20:][::-1]
    y = np.arange(len(biggest))[::-1]
    w = 0.4
    ax.barh(y - w/2, lns["euclidean"][biggest], w, color="#4c72b0",
            label="Euclidean", edgecolor="black")
    ax.barh(y + w/2, lns["cosine"][biggest], w, color="#dd8452",
            label="cosine", edgecolor="black")
    labels = [f"{tokens[i]}  (f={int(freq[i])})" for i in biggest]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("LNS(w)")
    ax.set_title("Top-20 tokens with largest |cosine - Euclidean| disagreement")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"LNS robustness: Euclidean vs cosine  "
        f"(bert_frozen_v2 vs bert_e2e, k={K}, V={v})",
        y=1.00, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"\nsaved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
