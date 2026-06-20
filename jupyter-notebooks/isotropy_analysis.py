"""
Isotropy / anisotropy analysis of token embedding matrices.

For each embedding matrix we report:

1) Pairwise cosine similarity distribution between random pairs of distinct
   tokens (classic anisotropy probe, Ethayarajh 2019; Mu & Viswanath 2018):
     - mean cosine   (0 under perfect isotropy)
     - std of cosine (sqrt(1/d) under uniform distribution on the unit sphere)
     - percentiles  p01 / p50 / p99

2) PCA spectrum of the (mean-centered) embedding matrix:
     - fraction of variance explained by the top 1/3/10/50 components
     - effective rank via the entropy of the normalized singular-value-squared
       spectrum:  d_eff = exp(H), with H = -sum p_i log p_i, p_i = s_i^2 / sum s_i^2
     - participation ratio: (sum s_i^2)^2 / sum s_i^4 -- another "effective rank"
       that's less sensitive to the long tail

3) Convenient per-matrix summary printed in a single line.

We also emit a 3-panel figure comparing the four filtered spaces:
    bert_frozen_v2, bert_e2e, gpt2_pca_frozen, gpt2_e2e
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM")

# (name, vectors_path, color, marker)
SPACES = [
    ("bert_frozen_v2_128d_filtered",
     ROOT / "embeddings/bert_frozen_v2_128d_filtered/bert_frozen_v2_vectors.tsv",
     "#1f77b4", "s"),
    ("bert_e2e_128d_filtered",
     ROOT / "embeddings/bert_128d_filtered/e2e_vectors.tsv",
     "#1f77b4", "o"),
    ("gpt2_pca_frozen_128d_filtered",
     ROOT / "embeddings/gpt2_pca_frozen_128d_filtered/gpt2_pca_frozen_vectors.tsv",
     "#d62728", "s"),
    ("gpt2_e2e_128d_filtered",
     ROOT / "embeddings/gpt2_e2e_128d_filtered/gpt2_e2e_vectors.tsv",
     "#d62728", "o"),
]


def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.parent.name}  {arr.shape}  in {time.time()-t0:.1f}s",
          flush=True)
    return arr


def sample_pair_cosines(e: np.ndarray, n_pairs: int = 200_000,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Cosine similarities between `n_pairs` random distinct token pairs."""
    rng = rng or np.random.default_rng(0)
    v = e.shape[0]
    # Unit-normalize rows once.
    en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    i = rng.integers(0, v, n_pairs)
    j = rng.integers(0, v, n_pairs)
    # Resample j where it collides with i (rare for v >> 1).
    mask = i == j
    while mask.any():
        j[mask] = rng.integers(0, v, mask.sum())
        mask = i == j
    # Cosine = dot of unit rows.
    cos = np.einsum("ij,ij->i", en[i], en[j])
    return cos.astype(np.float64)


def pca_spectrum(e: np.ndarray) -> np.ndarray:
    """Return squared singular values of the mean-centered matrix, sorted desc."""
    x = e - e.mean(axis=0, keepdims=True)
    # SVD directly on the (V, d) matrix; s has length min(V, d) = d.
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    return (s.astype(np.float64)) ** 2


def effective_rank(eig: np.ndarray) -> tuple[float, float]:
    """Entropy-based effective rank and participation ratio.

    eig is nonneg variance contributions (squared singular values).
    """
    p = eig / eig.sum()
    nz = p > 0
    H = float(-(p[nz] * np.log(p[nz])).sum())
    d_eff = float(np.exp(H))
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    return d_eff, pr


def analyse(name: str, e: np.ndarray, n_pairs: int = 200_000) -> dict:
    print(f"\n[{name}]  shape={e.shape}")
    cos = sample_pair_cosines(e, n_pairs=n_pairs)
    eig = pca_spectrum(e)
    d = e.shape[1]
    total = eig.sum()
    cumsum = np.cumsum(eig) / total
    top1 = float(cumsum[0])
    top3 = float(cumsum[2])
    top10 = float(cumsum[9])
    top50 = float(cumsum[min(49, len(cumsum) - 1)])
    d_eff, pr = effective_rank(eig)

    row_norms = np.linalg.norm(e, axis=1)

    stats = dict(
        vocab=int(e.shape[0]),
        dim=int(d),
        row_norm_mean=float(row_norms.mean()),
        row_norm_std=float(row_norms.std()),
        mean_cos=float(cos.mean()),
        std_cos=float(cos.std()),
        iso_ratio_vs_uniform=float(cos.std() / np.sqrt(1.0 / d)),
        p01=float(np.quantile(cos, 0.01)),
        p50=float(np.quantile(cos, 0.50)),
        p99=float(np.quantile(cos, 0.99)),
        p_neg=float((cos < 0).mean()),
        top1=top1,
        top3=top3,
        top10=top10,
        top50=top50,
        d_eff_entropy=d_eff,
        d_eff_participation=pr,
    )
    print(f"  pair-cosine: mean={stats['mean_cos']:+.4f}  std={stats['std_cos']:.4f} "
          f"(uniform-sphere std=sqrt(1/d)={1.0/np.sqrt(d):.4f})")
    print(f"             : p01={stats['p01']:+.3f}  p50={stats['p50']:+.3f}  "
          f"p99={stats['p99']:+.3f}  frac(cos<0)={stats['p_neg']:.3f}")
    print(f"  PCA      : top1={top1:.3f}  top3={top3:.3f}  top10={top10:.3f}  "
          f"top50={top50:.3f}")
    print(f"  effective rank: entropy d_eff={d_eff:.1f}/{d}  "
          f"participation ratio={pr:.1f}")
    return dict(stats=stats, cosines=cos, eig=eig)


def plot_everything(results: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))

    # Panel 1: cosine-distribution histograms.
    ax = axes[0]
    bins = np.linspace(-0.5, 1.0, 101)
    for (name, _p, color, marker) in SPACES:
        r = results[name]
        ax.hist(r["cosines"], bins=bins, histtype="step", linewidth=1.8,
                color=color, label=None,
                linestyle="-" if marker == "o" else "--",
                density=True)
    ax.axvline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_xlabel("cosine similarity (random token pairs)")
    ax.set_ylabel("density")
    ax.set_title("Pairwise cosine distribution\n(isotropic would center on 0)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.0)

    # Manual legend (color x line-style).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2, ls="--", label="BERT-tiny frozen-v2"),
        Line2D([0], [0], color="#1f77b4", lw=2, ls="-",  label="BERT-tiny e2e"),
        Line2D([0], [0], color="#d62728", lw=2, ls="--", label="GPT-2 PCA-frozen"),
        Line2D([0], [0], color="#d62728", lw=2, ls="-",  label="GPT-2 e2e"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, frameon=False)

    # Panel 2: cumulative PCA variance.
    ax = axes[1]
    for (name, _p, color, marker) in SPACES:
        r = results[name]
        cum = np.cumsum(r["eig"]) / r["eig"].sum()
        ax.plot(np.arange(1, len(cum) + 1), cum,
                color=color, lw=2,
                linestyle="-" if marker == "o" else "--")
    ax.set_xscale("log")
    ax.set_xlabel("PCA component rank  (log)")
    ax.set_ylabel("cumulative fraction of variance")
    ax.set_title("PCA spectrum  (steeper = more anisotropic)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1.02)

    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

    # Panel 3: summary bar chart -- mean cos, top-1 PCA frac, effective rank.
    ax = axes[2]
    names = [s[0] for s in SPACES]
    short = {
        "bert_frozen_v2_128d_filtered": "BERT-tiny\nfrozen",
        "bert_e2e_128d_filtered": "BERT-tiny\ne2e",
        "gpt2_pca_frozen_128d_filtered": "GPT-2\nPCA-frozen",
        "gpt2_e2e_128d_filtered": "GPT-2\ne2e",
    }
    labels = [short[n] for n in names]
    colors = [s[2] for s in SPACES]
    patterns = ["///" if s[3] == "s" else "" for s in SPACES]

    x = np.arange(len(names))
    width = 0.28

    mean_cos = [results[n]["stats"]["mean_cos"] for n in names]
    d_eff_norm = [results[n]["stats"]["d_eff_entropy"] / results[n]["stats"]["dim"]
                  for n in names]
    pr_norm = [results[n]["stats"]["d_eff_participation"] / results[n]["stats"]["dim"]
               for n in names]

    bars1 = ax.bar(x - width, mean_cos, width, label="mean cosine",
                   color=[c for c in colors],
                   edgecolor="black")
    bars2 = ax.bar(x,          d_eff_norm, width,
                   label="effective rank (entropy) / d",
                   color=[c for c in colors], alpha=0.75,
                   edgecolor="black")
    bars3 = ax.bar(x + width,  pr_norm, width,
                   label="participation ratio / d",
                   color=[c for c in colors], alpha=0.40,
                   edgecolor="black", hatch="///")

    for bs in (bars1, bars2, bars3):
        for b in bs:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.02 if h >= 0 else h - 0.04,
                    f"{h:+.2f}" if bs is bars1 else f"{h:.2f}",
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=8)

    # Reference line for perfect isotropy (both d_eff/d and PR/d = 1).
    ax.axhline(1.0, color="green", lw=0.8, ls="--", alpha=0.6)
    ax.text(len(names) - 0.5, 1.01, "isotropic = 1.0",
            color="green", fontsize=8, ha="right", va="bottom")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("Isotropy summary\n"
                 "(lower mean cos / higher d_eff, PR = more isotropic)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    fig.suptitle(
        "Isotropy / anisotropy of Diffusion-LM token embeddings  (ROCStories, filtered vocab)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out = ROOT / "charts" / "isotropy_analysis.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nsaved plot -> {out}")
    return out


def main() -> None:
    print("Loading embeddings:")
    data = {}
    for name, path, _c, _m in SPACES:
        data[name] = load_vectors(path)

    results = {}
    for name, _p, _c, _m in SPACES:
        results[name] = analyse(name, data[name])

    # Dump machine-readable summary.
    out_json = ROOT / "charts" / "isotropy_analysis.json"
    out_json.write_text(json.dumps(
        {k: v["stats"] for k, v in results.items()}, indent=2))
    print(f"\nsaved stats -> {out_json}")

    plot_everything(results)

    print("\n" + "=" * 100)
    print(f"{'space':32s} {'meanCos':>8s} {'stdCos':>8s} {'iso/uni':>8s} "
          f"{'top1':>6s} {'top10':>7s} {'d_eff':>7s} {'PR':>7s} {'gap':>6s}")
    print("-" * 100)
    for name in results:
        s = results[name]["stats"]
        gap = s['d_eff_entropy'] - s['d_eff_participation']
        print(f"{name:32s} {s['mean_cos']:+8.4f} {s['std_cos']:8.4f} "
              f"{s['iso_ratio_vs_uniform']:8.2f} "
              f"{s['top1']:6.3f} {s['top10']:7.3f} "
              f"{s['d_eff_entropy']:7.1f} "
              f"{s['d_eff_participation']:7.1f} "
              f"{gap:6.1f}")
    print("=" * 100)
    print("gap = d_eff(entropy) - PR  -->  large gap means heavy head + long flat tail")


if __name__ == "__main__":
    main()
