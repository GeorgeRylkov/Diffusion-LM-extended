"""
Hubness analysis of token-embedding matrices.

Hubness (Radovanovic, Nanopoulos, Ivanovic 2010) is the tendency of some
points in a high-dimensional space to appear as nearest neighbours of many
other points (hubs), while other points almost never appear in any k-NN list
(anti-hubs). Hubness is a well-known symptom of the curse of dimensionality
and an important probe for token-embedding quality in Diffusion-LM-style
models, because the rounding / nearest-token step is literally a k=1 NN
lookup into the embedding matrix.

For each embedding matrix E (rows = tokens, unit-normalised) and a given k,
we compute
            N_k(x) = #{ y != x : x in k-NN(y) under cosine distance }
and report

  - Skewness S_k of the {N_k(x)} distribution over the vocabulary.
    S_k ~ 0 for an isotropic point cloud; S_k grows with hubness.
  - Mean N_k (trivially = k, sanity check).
  - Max N_k and top-1% hub share: what fraction of *all* k-NN hits land
    on the top 1% most-frequent hubs. 1% under perfect uniformity.
  - Anti-hub fraction: fraction of tokens with N_k = 0, i.e. tokens that
    are never the nearest neighbour of anything.
  - "Robin Hood" index (Gini-like): cumulative fraction of N_k mass held
    by the top-q fraction of tokens, reported at q = 0.01 and q = 0.10.

Also produces a 3-panel figure:
  1. log-scale histogram of N_k for both spaces,
  2. Lorenz curve of N_k (cumulative share vs. token rank),
  3. summary bar chart with the main metrics side-by-side.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM")

# (short_name, vectors_path, metadata_path, color)
SPACES = [
    ("bert_frozen_v2",
     ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_vectors.tsv",
     ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_metadata.tsv",
     "#1f77b4"),
    ("bert_e2e",
     ROOT / "embeddings/bert_128d_filtered/e2e_vectors.tsv",
     ROOT / "embeddings/bert_128d_filtered/e2e_metadata.tsv",
     "#ff7f0e"),
]

K_VALUES = (1, 5, 10)
K_FOR_PLOTS = 10


# ---------- data loading ---------------------------------------------------- #

def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.parent.name}  {arr.shape}  in {time.time() - t0:.1f}s",
          flush=True)
    return arr


def load_tokens(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text().splitlines()]


# ---------- hubness core ---------------------------------------------------- #

def knn_incidence(e: np.ndarray, k: int) -> np.ndarray:
    """For every row x, find its k nearest neighbours (excl. self) and count
    how many times each row appears as someone else's NN.

    Uses cosine distance, computed as dot product on unit-normalised rows.
    Returns an int array N_k of length V.
    """
    v = e.shape[0]
    en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    # (V, V) similarity matrix; V=5180 so this is ~100 MB float32, fine.
    sim = en @ en.T
    np.fill_diagonal(sim, -np.inf)

    # argpartition for the k largest similarities per row; no need to sort.
    # kth index = v-1-k places the top-k in positions [v-k:].
    idx = np.argpartition(sim, kth=v - 1 - k, axis=1)[:, v - k:]  # (V, k)

    n = np.bincount(idx.ravel(), minlength=v)
    return n.astype(np.int64)


def skewness(x: np.ndarray) -> float:
    """Third standardised moment."""
    x = x.astype(np.float64)
    mu = x.mean()
    sd = x.std()
    if sd == 0:
        return 0.0
    return float(((x - mu) ** 3).mean() / sd ** 3)


def lorenz(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of total mass held by the top-q fraction.

    Returns (q, share) sorted so that q=0 -> share=0 and q=1 -> share=1.
    """
    y = np.sort(x)[::-1].astype(np.float64)
    tot = y.sum()
    cum = np.cumsum(y) / (tot if tot > 0 else 1.0)
    q = np.arange(1, len(y) + 1) / len(y)
    q = np.concatenate([[0.0], q])
    cum = np.concatenate([[0.0], cum])
    return q, cum


def cum_at(q: np.ndarray, cum: np.ndarray, target: float) -> float:
    idx = int(np.searchsorted(q, target))
    idx = min(idx, len(cum) - 1)
    return float(cum[idx])


def analyse(name: str, e: np.ndarray, tokens: list[str]) -> dict:
    print(f"\n[{name}]  shape={e.shape}")
    stats: dict[str, object] = dict(vocab=int(e.shape[0]), dim=int(e.shape[1]))
    per_k = {}
    for k in K_VALUES:
        t0 = time.time()
        n = knn_incidence(e, k)
        sk = skewness(n)
        q, cum = lorenz(n)
        top1pct = cum_at(q, cum, 0.01)
        top10pct = cum_at(q, cum, 0.10)
        zero_frac = float((n == 0).mean())
        top_hub_idx = np.argsort(-n)[:5]
        top_hubs = [(tokens[i], int(n[i])) for i in top_hub_idx]
        print(f"  k={k:>2}: skew={sk:6.2f}  max N_k={int(n.max()):>5}  "
              f"top1%-share={top1pct:.3f}  top10%-share={top10pct:.3f}  "
              f"anti-hub frac={zero_frac:.3f}  [{time.time()-t0:.1f}s]")
        print(f"        top-5 hubs: " +
              ", ".join(f"{tok!r}:{c}" for tok, c in top_hubs))
        per_k[k] = dict(
            skew=sk,
            mean=float(n.mean()),
            std=float(n.std()),
            max=int(n.max()),
            top1pct_share=top1pct,
            top10pct_share=top10pct,
            antihub_frac=zero_frac,
            top_hubs=top_hubs,
        )
        if k == K_FOR_PLOTS:
            stats["N_k_sample"] = n
            stats["lorenz_q"] = q
            stats["lorenz_cum"] = cum
    stats["per_k"] = per_k
    return stats


# ---------- plotting -------------------------------------------------------- #

def plot_everything(results: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: histogram of N_k (log y).
    ax = axes[0]
    for (name, _p, _m, color) in SPACES:
        n = results[name]["N_k_sample"]
        bins = np.arange(0, n.max() + 2)
        ax.hist(n, bins=bins, histtype="step", linewidth=1.8,
                color=color, label=f"{name}  (skew={results[name]['per_k'][K_FOR_PLOTS]['skew']:.2f})")
    ax.set_yscale("log")
    ax.axvline(K_FOR_PLOTS, color="black", lw=0.7, ls=":", alpha=0.7)
    ax.text(K_FOR_PLOTS, ax.get_ylim()[1] * 0.5,
            f" mean = k = {K_FOR_PLOTS}",
            fontsize=8, color="black", alpha=0.7)
    ax.set_xlabel(f"$N_{{{K_FOR_PLOTS}}}(x)$  = # times x is a NN "
                  f"(k={K_FOR_PLOTS}, cosine)")
    ax.set_ylabel("# tokens  (log)")
    ax.set_title("Distribution of neighbour-count\n"
                 "(right tail = hubs, spike at 0 = anti-hubs)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: Lorenz curve.
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":", label="equality")
    for (name, _p, _m, color) in SPACES:
        q = results[name]["lorenz_q"]
        cum = results[name]["lorenz_cum"]
        ax.plot(q, cum, color=color, lw=2, label=name)
    ax.set_xlabel("fraction of tokens (ranked by $N_k$, descending)")
    ax.set_ylabel(f"cumulative share of total $N_{{{K_FOR_PLOTS}}}$ mass")
    ax.set_title("Lorenz curve of hub mass\n"
                 "(more bowed = more concentrated)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)

    # Panel 3: summary bars.
    ax = axes[2]
    metric_keys = [
        ("skew", "skewness $S_k$"),
        ("top1pct_share", "top-1% share"),
        ("antihub_frac", "anti-hub frac ($N_k=0$)"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.38
    for i, (name, _p, _m, color) in enumerate(SPACES):
        d = results[name]["per_k"][K_FOR_PLOTS]
        vals = [d[k] for k, _ in metric_keys]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      color=color, edgecolor="black", label=name)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metric_keys], fontsize=9)
    ax.set_title(f"Hubness summary  (k = {K_FOR_PLOTS})\n"
                 "(lower skew / top-1% / anti-hub = better)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Hubness of Diffusion-LM BERT-tiny embeddings  "
        "(ROCStories, filtered vocab, 128-dim)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = ROOT / "charts" / "hubness_bert_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nsaved plot -> {out}")
    return out


# ---------- main ------------------------------------------------------------ #

def main() -> None:
    print("Loading embeddings:")
    data = {}
    for name, vpath, mpath, _c in SPACES:
        e = load_vectors(vpath)
        toks = load_tokens(mpath)
        assert len(toks) == e.shape[0], f"{name}: tokens {len(toks)} vs rows {e.shape[0]}"
        data[name] = (e, toks)

    results = {}
    for name, _v, _m, _c in SPACES:
        e, toks = data[name]
        results[name] = analyse(name, e, toks)

    summary = {}
    for name, s in results.items():
        summary[name] = dict(
            vocab=s["vocab"], dim=s["dim"],
            per_k={str(k): v for k, v in s["per_k"].items()},
        )
    out_json = ROOT / "charts" / "hubness_bert_comparison.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved stats -> {out_json}")

    plot_everything(results)

    print("\n" + "=" * 96)
    print(f"Hubness comparison at k={K_FOR_PLOTS}")
    print(f"{'space':20s} {'skew':>7s} {'max':>6s} {'top1%':>8s} {'top10%':>8s} "
          f"{'antihub%':>9s}")
    print("-" * 96)
    for name in results:
        d = results[name]["per_k"][K_FOR_PLOTS]
        print(f"{name:20s} {d['skew']:7.2f} {d['max']:6d} "
              f"{d['top1pct_share']:8.3f} {d['top10pct_share']:8.3f} "
              f"{d['antihub_frac']*100:8.2f}%")
    print("=" * 96)
    print("Lower skew + smaller top-1% share + fewer anti-hubs => more uniform "
          "(better) neighbour distribution.")


if __name__ == "__main__":
    main()
