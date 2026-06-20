"""
Hubness analysis for two ROCStories e2e embedding matrices (different RNG seeds).

Same pipeline as `hubness_analysis.py` / `wiki_hubness_lns.py`:
  N_k(x) = # of other rows y for which x is in y's cosine k-NN set (excluding self).

Reports skewness S_k, max N_k, max/k, top-1% Lorenz share, anti-hub fraction,
and top-5 hub tokens per k in K_VALUES.

Outputs:
  charts/hubness_bert_e2e_sd101_vs_sd69.png
  charts/hubness_bert_e2e_sd101_vs_sd69.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM")
EMB_DIR = ROOT / "embeddings/bert_e2e_sd101_e2e_sd69_filtered"

SPACES = [
    ("e2e_sd_101", EMB_DIR / "e2e_vectors_sd_101.tsv",
     EMB_DIR / "e2e_sd_101_metadata.tsv", "#1f77b4"),
    ("e2e_sd_69", EMB_DIR / "e2e_vectors_sd_69.tsv",
     EMB_DIR / "e2e_sd_69_metadata.tsv", "#ff7f0e"),
]

K_VALUES = (1, 5, 10, 20, 50, 100)


def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.name}  {arr.shape}  in {time.time()-t0:.1f}s", flush=True)
    return arr


def load_tokens(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text().splitlines()]


def knn_incidence(e: np.ndarray, k: int) -> np.ndarray:
    v = e.shape[0]
    en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    sim = en @ en.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(sim, kth=v - 1 - k, axis=1)[:, v - k:]
    n = np.bincount(idx.ravel(), minlength=v)
    return n.astype(np.int64)


def skewness(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    mu = x.mean()
    sd = x.std()
    if sd == 0:
        return 0.0
    return float(((x - mu) ** 3).mean() / sd ** 3)


def lorenz_top_share(x: np.ndarray, q: float) -> float:
    y = np.sort(x)[::-1].astype(np.float64)
    cum = np.cumsum(y) / max(y.sum(), 1.0)
    idx = max(1, int(np.ceil(q * len(y)))) - 1
    return float(cum[idx])


def hubness_scan(name: str, e: np.ndarray, tokens: list[str]) -> dict:
    print(f"\n[{name}]  shape={e.shape}")
    rows = {}
    for k in K_VALUES:
        t0 = time.time()
        n = knn_incidence(e, k)
        sk = skewness(n)
        topshare = lorenz_top_share(n, 0.01)
        zero = float((n == 0).mean())
        top_idx = np.argsort(-n)[:5]
        rows[k] = dict(
            skew=sk,
            mean=float(n.mean()),
            max=int(n.max()),
            max_over_k=float(n.max() / k),
            top1pct_share=topshare,
            antihub_frac=zero,
            top_hubs=[(tokens[i], int(n[i])) for i in top_idx],
        )
        print(f"  k={k:>3}: skew={sk:6.2f}  max={int(n.max()):>5}  "
              f"max/k={n.max()/k:5.2f}  top1%={topshare:.3f}  "
              f"antihub={zero*100:5.2f}%  "
              f"top5={rows[k]['top_hubs']}  [{time.time()-t0:.1f}s]")
    return rows


def plot_hubness(results: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ks = list(K_VALUES)
    short = {"e2e_sd_101": "sd101", "e2e_sd_69": "sd69"}
    for name in results:
        c = "#1f77b4" if "101" in name else "#ff7f0e"
        skews = [results[name][k]["skew"] for k in ks]
        maxok = [results[name][k]["max"] / k for k in ks]
        anti = [results[name][k]["antihub_frac"] * 100 for k in ks]
        axes[0].plot(ks, skews, "o-", color=c, lw=2, label=short[name])
        axes[1].plot(ks, maxok, "o-", color=c, lw=2, label=short[name])
        axes[2].plot(ks, anti, "o-", color=c, lw=2, label=short[name])

    for ax, ttl, ylab in [
        (axes[0], r"Hubness skewness $S_k$ vs k", r"$S_k$"),
        (axes[1], r"Worst hub: $\max N_k / k$ vs k", r"$\max N_k / k$"),
        (axes[2], r"Anti-hub fraction (% with $N_k=0$)", "anti-hubs (%)"),
    ]:
        ax.set_title(ttl)
        ax.set_xlabel("k")
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10, frameon=False)

    fig.suptitle(
        "Hubness: ROCStories e2e embeddings  (seed 101 vs seed 69, V=5180, cosine)",
        y=1.03, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = ROOT / "charts/hubness_bert_e2e_sd101_vs_sd69.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")
    return out


def main() -> None:
    print("Loading e2e seed embeddings:")
    data = {}
    tokens = {}
    for name, vpath, mpath, _c in SPACES:
        data[name] = load_vectors(vpath)
        tokens[name] = load_tokens(mpath)
    assert tokens["e2e_sd_101"] == tokens["e2e_sd_69"], "Vocab order must match."

    results = {}
    for name, _v, _m, _c in SPACES:
        results[name] = hubness_scan(name, data[name], tokens[name])

    out_json = ROOT / "charts/hubness_bert_e2e_sd101_vs_sd69.json"
    serial = {
        name: {
            str(k): {kk: vv for kk, vv in row.items() if kk != "top_hubs"}
            | {"top_hubs": row["top_hubs"]}
            for k, row in res.items()
        }
        for name, res in results.items()
    }
    out_json.write_text(json.dumps(serial, indent=2))
    print(f"saved -> {out_json}")

    print("\n=== Skewness summary ===")
    print(f"{'k':>4}  {'sd101':>8}  {'sd69':>8}  {'Δ':>7}")
    for k in K_VALUES:
        a = results["e2e_sd_101"][k]["skew"]
        b = results["e2e_sd_69"][k]["skew"]
        print(f"{k:>4}  {a:8.3f}  {b:8.3f}  {b-a:+7.3f}")

    plot_hubness(results)


if __name__ == "__main__":
    main()
