"""
Hubness + LNS for GPT-2 128d filtered embeddings (ROCStories):
  gpt2_pca_frozen_128d_filtered  vs  gpt2_e2e_128d_filtered

V = 5520, d = 128. Cosine k-NN throughout (same as BERT analyses).

Outputs:
  charts/hubness_gpt2_128d.png, charts/hubness_gpt2_128d.json
  charts/lns_gpt2_128d_k10.png, charts/lns_gpt2_128d_k10.json,
      charts/lns_gpt2_128d_k10_per_token.tsv
  charts/lns_gpt2_128d_k50.png, charts/lns_gpt2_128d_k50.json,
      charts/lns_gpt2_128d_k50_per_token.tsv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ROOT = Path(__file__).resolve().parents[1]
DIR_F = ROOT / "embeddings/gpt2_pca_frozen_128d_filtered"
DIR_E = ROOT / "embeddings/gpt2_e2e_128d_filtered"

SPACES = [
    ("gpt2_pca_frozen", DIR_F / "gpt2_pca_frozen_vectors.tsv",
     DIR_F / "gpt2_pca_frozen_metadata.tsv", "#1f77b4"),
    ("gpt2_e2e", DIR_E / "gpt2_e2e_vectors.tsv",
     DIR_E / "gpt2_e2e_metadata.tsv", "#ff7f0e"),
]

K_HUBNESS = (1, 5, 10, 20, 50, 100)
K_LNS = (10, 50)
METRIC = "cosine"


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
    mu, sd = x.mean(), x.std()
    return 0.0 if sd == 0 else float(((x - mu) ** 3).mean() / sd ** 3)


def lorenz_top_share(x: np.ndarray, q: float) -> float:
    y = np.sort(x)[::-1].astype(np.float64)
    cum = np.cumsum(y) / max(y.sum(), 1.0)
    idx = max(1, int(np.ceil(q * len(y)))) - 1
    return float(cum[idx])


def hubness_scan(name: str, e: np.ndarray, tokens: list[str]) -> dict:
    print(f"\n[{name}]  shape={e.shape}")
    rows = {}
    for k in K_HUBNESS:
        t0 = time.time()
        n = knn_incidence(e, k)
        sk = skewness(n)
        topshare = lorenz_top_share(n, 0.01)
        zero = float((n == 0).mean())
        top_idx = np.argsort(-n)[:5]
        rows[k] = dict(
            skew=sk, mean=float(n.mean()), max=int(n.max()),
            max_over_k=float(n.max() / k), top1pct_share=topshare,
            antihub_frac=zero,
            top_hubs=[(tokens[i], int(n[i])) for i in top_idx],
        )
        print(f"  k={k:>3}: skew={sk:6.2f}  max={int(n.max()):>5}  "
              f"max/k={n.max()/k:5.2f}  top1%={topshare:.3f}  "
              f"antihub={zero*100:5.2f}%  {rows[k]['top_hubs']}  "
              f"[{time.time()-t0:.1f}s]")
    return rows


def plot_hubness(results: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ks = list(K_HUBNESS)
    colors = {"gpt2_pca_frozen": "#1f77b4", "gpt2_e2e": "#ff7f0e"}
    for name in results:
        c = colors[name]
        skews = [results[name][k]["skew"] for k in ks]
        maxok = [results[name][k]["max"] / k for k in ks]
        anti = [results[name][k]["antihub_frac"] * 100 for k in ks]
        axes[0].plot(ks, skews, "o-", color=c, lw=2, label=name.replace("_", " "))
        axes[1].plot(ks, maxok, "o-", color=c, lw=2, label=name.replace("_", " "))
        axes[2].plot(ks, anti, "o-", color=c, lw=2, label=name.replace("_", " "))

    for ax, ttl, ylab in [
        (axes[0], r"Hubness skewness $S_k$ vs k", r"$S_k$"),
        (axes[1], r"$\max N_k / k$ vs k", r"$\max N_k / k$"),
        (axes[2], r"Anti-hub % ($N_k=0$)", "%"),
    ]:
        ax.set_title(ttl)
        ax.set_xlabel("k")
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, frameon=False)

    fig.suptitle(
        "Hubness: GPT-2 128d PCA-frozen vs e2e  (ROCStories filtered, V=5520)",
        y=1.03, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = ROOT / "charts/hubness_gpt2_128d.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")
    return out


def knn_sets(e: np.ndarray, k: int) -> np.ndarray:
    v = e.shape[0]
    en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    sim = en @ en.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(sim, kth=v - 1 - k, axis=1)[:, v - k:].astype(np.int64)


def jaccard_batch(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    v = a.shape[0]
    out = np.empty(v, dtype=np.float64)
    a = np.sort(a, axis=1)
    b = np.sort(b, axis=1)
    for i in range(v):
        inter = np.intersect1d(a[i], b[i], assume_unique=True).size
        out[i] = inter / (2 * k - inter)
    return out


def plot_lns(lns: np.ndarray, tokens: list[str], k: int, rnd: float) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    ax = axes[0]
    bins = np.linspace(0, max(0.05, float(lns.max()) + 1e-3), 50)
    ax.hist(lns, bins=bins, color="#9467bd", edgecolor="black", alpha=0.85)
    ax.axvline(float(lns.mean()), color="black", lw=1.5, ls="--",
               label=f"mean = {lns.mean():.3f}")
    ax.axvline(float(np.median(lns)), color="gray", lw=1.2, ls=":",
               label=f"median = {np.median(lns):.3f}")
    ax.axvline(rnd, color="red", lw=1.2, ls="-.", label=f"random = {rnd:.4f}")
    ax.set_xlabel(f"LNS(w)  (k={k}, cosine)")
    ax.set_ylabel("# tokens")
    ax.set_title("LNS distribution")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    xs = np.sort(lns)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color="#9467bd", lw=2)
    ax.set_xlabel("LNS(w)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of LNS")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    order = np.argsort(-lns)[:18]
    y = np.arange(len(order))[::-1]
    ax.barh(y, lns[order], color="#55a868", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels([tokens[i] for i in order], fontsize=8)
    ax.set_xlabel("LNS(w)")
    ax.set_title("Top-18 by LNS")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"LNS: GPT-2 PCA-frozen vs e2e  (k={k}, cosine, V={len(tokens)})",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = ROOT / f"charts/lns_gpt2_128d_k{k}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")
    return out


def main() -> None:
    print("Loading GPT-2 filtered embeddings:")
    data = {}
    tokens = {}
    for name, vpath, mpath, _c in SPACES:
        data[name] = load_vectors(vpath)
        tokens[name] = load_tokens(mpath)
    assert tokens["gpt2_pca_frozen"] == tokens["gpt2_e2e"]
    tok = tokens["gpt2_pca_frozen"]
    v, d = data["gpt2_pca_frozen"].shape
    print(f"vocab={v}  dim={d}")

    hub = {}
    for name, _v, _m, _c in SPACES:
        hub[name] = hubness_scan(name, data[name], tok)

    out_h = ROOT / "charts/hubness_gpt2_128d.json"
    out_h.write_text(json.dumps({
        n: {str(k): {kk: vv for kk, vv in row.items() if kk != "top_hubs"}
            | {"top_hubs": row["top_hubs"]}
            for k, row in res.items()}
        for n, res in hub.items()
    }, indent=2))
    print(f"saved -> {out_h}")
    plot_hubness(hub)

    ef = data["gpt2_pca_frozen"]
    ee = data["gpt2_e2e"]
    for k in K_LNS:
        print(f"\n=== LNS k={k} ===")
        t0 = time.time()
        lns = jaccard_batch(knn_sets(ef, k), knn_sets(ee, k), k)
        rnd = (k * k / (v - 1)) / (2 * k - k * k / (v - 1))
        print(f"  mean={lns.mean():.4f}  median={np.median(lns):.4f}  "
              f"random={rnd:.4f}  mean/rnd={lns.mean()/rnd:.1f}x  "
              f"[{time.time()-t0:.1f}s]")
        out_j = ROOT / f"charts/lns_gpt2_128d_k{k}.json"
        out_j.write_text(json.dumps(dict(
            k=k, metric=METRIC, vocab=v, dim=d,
            mean=float(lns.mean()), median=float(np.median(lns)),
            std=float(lns.std()),
            quantiles={f"p{int(q*100):02d}": float(np.quantile(lns, q))
                       for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
            random_baseline=float(rnd),
        ), indent=2))
        out_tsv = ROOT / f"charts/lns_gpt2_128d_k{k}_per_token.tsv"
        with out_tsv.open("w") as f:
            f.write("token\tlns\n")
            for t, l in zip(tok, lns):
                f.write(f"{t}\t{l:.6f}\n")
        print(f"  saved -> {out_j.name}, {out_tsv.name}")
        plot_lns(lns, tok, k, rnd)


if __name__ == "__main__":
    main()
