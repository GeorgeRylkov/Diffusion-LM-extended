"""
Local Neighborhood Similarity (LNS) between two embedding spaces.

LNS is from the Embedding Comparator paper (Boggust et al. 2022, IUI '22,
https://vis.csail.mit.edu/pubs/embedding-comparator.pdf). For each token w,

    LNS(w) = J(k-NN_1(w), k-NN_2(w))

where k-NN_i(w) is the set of the k nearest neighbours of w in space i, and
J is the Jaccard index

    J(A, B) = |A n B| / |A u B|.

LNS(w) = 1 means the two spaces agree perfectly about w's local neighbourhood.
LNS(w) = 0 means the two neighbourhoods are disjoint. A random baseline (two
independent uniform neighbourhoods of size k drawn from V) is approximately
E[LNS_rand] = k / (2V - k) -- tiny for k=50, V=5180 (~0.00485).

The paper uses Euclidean distance for text embeddings; we use cosine after
confirming that Euclidean and cosine k-NN agree to within Spearman 0.86 on
these spaces (see charts/lns_metric_comparison.png).

We compare
    bert_frozen_v2_128d_filtered   vs   bert_e2e_128d_filtered

and also correlate LNS with training-corpus token frequency to see whether
high- or low-frequency tokens are the most affected by e2e fine-tuning.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM")

SPACE_A = dict(
    name="bert_frozen_v2",
    vectors=ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_vectors.tsv",
    tokens=ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_metadata.tsv",
    color="#1f77b4",
)
SPACE_B = dict(
    name="bert_e2e",
    vectors=ROOT / "embeddings/bert_128d_filtered/e2e_vectors.tsv",
    tokens=ROOT / "embeddings/bert_128d_filtered/e2e_metadata.tsv",
    color="#ff7f0e",
)

CORPUS = ROOT / "datasets/ROCstory/roc_train_corpus_bert.txt"

K = 50
METRIC = "cosine"      # cosine chosen after Euclidean/cosine robustness check
                       # (see charts/lns_metric_comparison.png): the two agree
                       # to within Spearman 0.86, and cosine removes per-token
                       # norm as a confound.


# ---------- loaders --------------------------------------------------------- #

def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.parent.name}  {arr.shape}  in {time.time() - t0:.1f}s",
          flush=True)
    return arr


def load_tokens(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text().splitlines()]


def token_frequencies(tokens: list[str]) -> dict[str, int]:
    """Count occurrences of every token in the ROCStories BERT-tokenized corpus.

    The corpus already contains BERT wordpieces (with `##` prefixes) separated
    by whitespace, so a simple split is the right tokenization here.
    """
    cnt: Counter[str] = Counter()
    t0 = time.time()
    vocab = set(tokens)
    with CORPUS.open("r") as f:
        for line in f:
            for tok in line.split():
                if tok in vocab:
                    cnt[tok] += 1
    print(f"  counted corpus frequencies in {time.time() - t0:.1f}s  "
          f"(nonzero tokens: {sum(1 for c in cnt.values() if c > 0):,} / "
          f"{len(tokens):,})", flush=True)
    return cnt


# ---------- LNS core -------------------------------------------------------- #

def knn_sets(e: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """For every row, return indices of its k nearest neighbours (excl. self).

    Shape: (V, k), dtype int64. Not sorted; used only via set membership.
    """
    v = e.shape[0]
    if metric == "euclidean":
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b . Only relative order matters,
        # so we can use -(a.b) + 0.5*||a||^2 (the ||b||^2 term is the same for
        # all b fixing a). But simplest and still cheap: full squared-dist mat.
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


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank (like scipy.stats.rankdata 'average'), no SciPy dep."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Tie correction: replace runs of equal values with their mean rank.
    xs = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            mean_rank = 0.5 * (i + j + 1)
            ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def jaccard_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard between two (V, k) index matrices.

    Both have exactly k non-self neighbours. |A|=|B|=k, so

        |A n B| = inter
        |A u B| = 2k - inter
        J       = inter / (2k - inter)
    """
    v, k = a.shape
    out = np.empty(v, dtype=np.float64)
    a = np.sort(a, axis=1)
    b = np.sort(b, axis=1)
    # Vectorised intersection via merge-count: np.in1d is not row-wise, so
    # we do a simple Python loop; 5180 rows at k=50 is < 0.5 s anyway.
    for i in range(v):
        inter = np.intersect1d(a[i], b[i], assume_unique=True).size
        out[i] = inter / (2 * k - inter)
    return out


# ---------- plots ----------------------------------------------------------- #

def plot_lns(lns: np.ndarray, freqs: np.ndarray, tokens: list[str],
             random_baseline: float) -> Path:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # Panel A: distribution of LNS(w) across the vocabulary.
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(lns, bins=40, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.axvline(float(lns.mean()), color="black", lw=1.5, ls="--",
               label=f"mean = {lns.mean():.3f}")
    ax.axvline(float(np.median(lns)), color="gray", lw=1.2, ls=":",
               label=f"median = {np.median(lns):.3f}")
    ax.axvline(random_baseline, color="red", lw=1.2, ls="-.",
               label=f"random = {random_baseline:.4f}")
    ax.set_xlabel(f"LNS(w)  (Jaccard over k={K} {METRIC} neighbours)")
    ax.set_ylabel("# tokens")
    ax.set_title("Distribution of per-token LNS\n"
                 "(1 = identical neighbourhood, 0 = disjoint)")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)

    # Panel B: LNS CDF + quantiles.
    ax = fig.add_subplot(gs[0, 1])
    xs = np.sort(lns)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color="#4c72b0", lw=2)
    for q_val, label in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
        qq = float(np.quantile(lns, q_val))
        ax.axvline(qq, color="gray", lw=0.8, ls=":")
        ax.text(qq, q_val, f" {label}={qq:.2f}", fontsize=8, va="center")
    ax.set_xlabel("LNS(w)")
    ax.set_ylabel("cumulative fraction of tokens")
    ax.set_title("CDF of LNS")
    ax.grid(True, alpha=0.3)

    # Panel C: LNS vs token frequency (log scale).
    ax = fig.add_subplot(gs[0, 2])
    mask = freqs > 0
    ax.scatter(freqs[mask], lns[mask], s=6, alpha=0.25,
               color="#4c72b0", edgecolor="none")
    logf = np.log10(freqs[mask])
    bin_edges = np.linspace(logf.min(), logf.max(), 14)
    bin_idx = np.digitize(logf, bin_edges)
    centers, medians, p25s, p75s = [], [], [], []
    for bi in range(1, len(bin_edges)):
        sel = bin_idx == bi
        if sel.sum() < 5:
            continue
        centers.append(10 ** ((bin_edges[bi - 1] + bin_edges[bi]) / 2))
        medians.append(float(np.median(lns[mask][sel])))
        p25s.append(float(np.quantile(lns[mask][sel], 0.25)))
        p75s.append(float(np.quantile(lns[mask][sel], 0.75)))
    centers = np.array(centers)
    ax.plot(centers, medians, color="black", lw=2, label="binned median")
    ax.fill_between(centers, p25s, p75s, color="black", alpha=0.15,
                    label="IQR")
    rho = _spearman(freqs[mask].astype(np.float64), lns[mask])
    ax.set_xscale("log")
    ax.set_xlabel("corpus frequency (log)")
    ax.set_ylabel("LNS(w)")
    ax.set_title(f"LNS vs token frequency\n"
                 f"Spearman rho = {rho:+.3f}")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # Panel D: LNS aggregated by frequency *decile*.
    ax = fig.add_subplot(gs[1, 0])
    nz_freqs = freqs[mask]
    nz_lns = lns[mask]
    quantile_edges = np.quantile(nz_freqs, np.linspace(0, 1, 11))
    # np.quantile can give equal edges for repeated values; ensure strict-ish.
    bin_idx = np.clip(np.searchsorted(quantile_edges, nz_freqs,
                                      side="right") - 1, 0, 9)
    decile_means, decile_medians = [], []
    for d in range(10):
        sel = bin_idx == d
        decile_means.append(float(nz_lns[sel].mean()) if sel.any() else np.nan)
        decile_medians.append(float(np.median(nz_lns[sel])) if sel.any()
                              else np.nan)
    xpos = np.arange(10)
    ax.bar(xpos - 0.18, decile_means, 0.36, color="#4c72b0",
           edgecolor="black", label="mean")
    ax.bar(xpos + 0.18, decile_medians, 0.36, color="#dd8452",
           edgecolor="black", label="median")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"D{i+1}" for i in range(10)])
    ax.set_xlabel("frequency decile  (D1 = rarest, D10 = most frequent)")
    ax.set_ylabel("LNS(w)")
    ax.set_title("LNS by frequency decile")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel E: bottom-20 LNS tokens (most-changed local neighbourhood).
    ax = fig.add_subplot(gs[1, 1])
    order_low = np.argsort(lns)[:20]
    y = np.arange(len(order_low))[::-1]
    ax.barh(y, lns[order_low], color="#c44e52", edgecolor="black")
    labels = [f"{tokens[i]}  (f={int(freqs[i])})" for i in order_low]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("LNS(w)")
    ax.set_title("Tokens with the LEAST-similar neighbourhood\n"
                 "(biggest change from frozen to e2e)")
    ax.grid(True, axis="x", alpha=0.3)

    # Panel F: top-20 LNS tokens (neighbourhood preserved).
    ax = fig.add_subplot(gs[1, 2])
    order_high = np.argsort(-lns)[:20]
    y = np.arange(len(order_high))[::-1]
    ax.barh(y, lns[order_high], color="#55a868", edgecolor="black")
    labels = [f"{tokens[i]}  (f={int(freqs[i])})" for i in order_high]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("LNS(w)")
    ax.set_title("Tokens with the MOST-similar neighbourhood\n"
                 "(unchanged by fine-tuning)")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"Local Neighborhood Similarity  (bert_frozen_v2  vs  bert_e2e, "
        f"k={K}, {METRIC}, V={len(tokens)})",
        y=1.00, fontsize=13, fontweight="bold",
    )
    out = ROOT / "charts" / "lns_bert_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nsaved plot -> {out}")
    return out


# ---------- main ------------------------------------------------------------ #

def main() -> None:
    print("Loading embeddings:")
    ea = load_vectors(SPACE_A["vectors"])
    eb = load_vectors(SPACE_B["vectors"])
    ta = load_tokens(SPACE_A["tokens"])
    tb = load_tokens(SPACE_B["tokens"])
    assert ta == tb, "Vocab must match between spaces for LNS."
    tokens = ta
    v = len(tokens)
    print(f"vocab size: {v}   dim: {ea.shape[1]}   k: {K}   metric: {METRIC}")

    print("\nComputing k-NN for each space:")
    t0 = time.time()
    nn_a = knn_sets(ea, k=K, metric=METRIC)
    print(f"  NN for {SPACE_A['name']}  in {time.time() - t0:.1f}s")
    t0 = time.time()
    nn_b = knn_sets(eb, k=K, metric=METRIC)
    print(f"  NN for {SPACE_B['name']}  in {time.time() - t0:.1f}s")

    print("\nComputing per-token Jaccard:")
    t0 = time.time()
    lns = jaccard_batch(nn_a, nn_b)
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nCorpus frequencies:")
    fc = token_frequencies(tokens)
    freqs = np.array([fc.get(t, 0) for t in tokens], dtype=np.int64)

    # Random baseline: two uniform k-subsets drawn independently from V-1 rows.
    # E[|A n B|] = k * k / (V - 1); hence E[LNS_rand] = E[inter]/(2k - E[inter]).
    exp_inter = K * K / (v - 1)
    rnd = exp_inter / (2 * K - exp_inter)

    print(f"\n--- per-token LNS summary ---")
    print(f"  mean   = {lns.mean():.4f}")
    print(f"  median = {np.median(lns):.4f}")
    print(f"  std    = {lns.std():.4f}")
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"  p{int(q*100):02d}    = {np.quantile(lns, q):.4f}")
    print(f"  random baseline = {rnd:.4f}  "
          f"(mean relative to random: {lns.mean() / rnd:.1f}x)")

    mask = freqs > 0
    nz_freqs = freqs[mask]; nz_lns = lns[mask]
    edges = np.quantile(nz_freqs, np.linspace(0, 1, 11))
    bin_idx = np.clip(np.searchsorted(edges, nz_freqs, side="right") - 1, 0, 9)
    print(f"\n--- LNS by frequency decile ({mask.sum()} nonzero tokens) ---")
    print(f"  {'decile':>6s} {'freq range':>18s} {'n':>5s} "
          f"{'mean':>7s} {'median':>7s}")
    for d in range(10):
        sel = bin_idx == d
        if not sel.any():
            continue
        lo, hi = nz_freqs[sel].min(), nz_freqs[sel].max()
        print(f"    D{d+1:<4d} {f'[{lo}, {hi}]':>18s} {sel.sum():>5d} "
              f"{nz_lns[sel].mean():7.3f} {np.median(nz_lns[sel]):7.3f}")

    print("\n--- 15 tokens with LOWEST LNS (most-changed neighbourhood) ---")
    order_low = np.argsort(lns)[:15]
    for i in order_low:
        print(f"  LNS={lns[i]:.3f}  freq={int(freqs[i]):>6d}   token={tokens[i]!r}")
    print("\n--- 15 tokens with HIGHEST LNS (least-changed neighbourhood) ---")
    order_high = np.argsort(-lns)[:15]
    for i in order_high:
        print(f"  LNS={lns[i]:.3f}  freq={int(freqs[i]):>6d}   token={tokens[i]!r}")

    out_tsv = ROOT / "charts" / "lns_bert_per_token.tsv"
    with out_tsv.open("w") as f:
        f.write("token\tfrequency\tlns\n")
        for tok, fr, l in zip(tokens, freqs, lns):
            f.write(f"{tok}\t{int(fr)}\t{l:.6f}\n")
    print(f"\nsaved per-token table -> {out_tsv}")

    out_json = ROOT / "charts" / "lns_bert_comparison.json"
    out_json.write_text(json.dumps(dict(
        k=K, metric=METRIC, vocab=v, dim=int(ea.shape[1]),
        mean=float(lns.mean()), median=float(np.median(lns)),
        std=float(lns.std()),
        quantiles={f"p{int(q*100):02d}": float(np.quantile(lns, q))
                   for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
        random_baseline=rnd,
    ), indent=2))
    print(f"saved summary    -> {out_json}")

    plot_lns(lns, freqs, tokens, random_baseline=rnd)


if __name__ == "__main__":
    main()
