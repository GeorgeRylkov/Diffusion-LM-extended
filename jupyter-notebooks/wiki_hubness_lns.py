"""
Hubness + LNS for the Wikipedia BERT 128d filtered embeddings.

Reuses the analysis logic from
    jupyter-notebooks/hubness_analysis.py
    jupyter-notebooks/lns_analysis.py
parameterised onto the wiki vocabulary (V=10440, d=128).

We deliberately skip the frequency-decile LNS panel because we don't have a
local tokenized Wikipedia corpus -- the data lives as raw parquets. The
frequency-independent metrics are the bulk of the story anyway:
  - hubness skewness, max N_k, anti-hub %, top-1 % share at k in {1,5,10,20,50,100};
  - LNS distribution, mean/median/quantiles, top-N hubs and antihubs.

Outputs:
  charts/hubness_wiki_bert128d.png + .json
  charts/lns_wiki_bert128d_k10.png + .json + per_token.tsv
  charts/lns_wiki_bert128d_k50.png + .json + per_token.tsv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM")
SPACE_DIR = ROOT / "embeddings/wiki_bert_128d_filtered"

A_NAME = "wiki_bert_frozen"
B_NAME = "wiki_bert_e2e"

A_VEC = SPACE_DIR / "frozen_vectors.tsv"
A_TOK = SPACE_DIR / "frozen_metadata.tsv"
B_VEC = SPACE_DIR / "e2e_vectors.tsv"
B_TOK = SPACE_DIR / "e2e_metadata.tsv"

K_HUBNESS = (1, 5, 10, 20, 50, 100)
K_LNS = (10, 50)
LNS_METRIC = "cosine"


# ---------- helpers --------------------------------------------------------- #

def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.name}  {arr.shape}  in {time.time()-t0:.1f}s",
          flush=True)
    return arr


def load_tokens(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text().splitlines()]


# ---------- hubness --------------------------------------------------------- #

def knn_incidence(e: np.ndarray, k: int) -> np.ndarray:
    """For every row, find its k nearest cosine neighbours (excl. self) and
    count, per row, how many times it appears as someone's NN."""
    v = e.shape[0]
    en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    sim = en @ en.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(sim, kth=v - 1 - k, axis=1)[:, v - k:]
    n = np.bincount(idx.ravel(), minlength=v)
    return n.astype(np.int64)


def skewness(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    mu = x.mean(); sd = x.std()
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
            top1pct_share=topshare, antihub_frac=zero,
            top_hubs=[(tokens[i], int(n[i])) for i in top_idx],
        )
        print(f"  k={k:>3}: skew={sk:6.2f}  max={int(n.max()):>5}  "
              f"max/k={n.max()/k:5.2f}  top1%={topshare:.3f}  "
              f"antihub={zero*100:5.2f}%   top5={rows[k]['top_hubs']}  "
              f"[{time.time()-t0:.1f}s]")
    return rows


def plot_hubness(results: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    ks = list(K_HUBNESS)
    colors = {"frozen": "#1f77b4", "e2e": "#ff7f0e"}
    short = {A_NAME: "frozen", B_NAME: "e2e"}
    for name in results:
        skews = [results[name][k]["skew"] for k in ks]
        maxs  = [results[name][k]["max"] for k in ks]
        anti  = [results[name][k]["antihub_frac"] * 100 for k in ks]
        c = colors[short[name]]
        axes[0].plot(ks, skews, "o-", color=c, lw=2, label=short[name])
        axes[1].plot(ks, [m/k for m, k in zip(maxs, ks)], "o-",
                     color=c, lw=2, label=short[name])
        axes[2].plot(ks, anti, "o-", color=c, lw=2, label=short[name])

    for ax, ttl, ylab in [
        (axes[0], r"Hubness skewness $S_k$ vs k",        r"$S_k$"),
        (axes[1], r"Worst hub: $\max N_k / k$ vs k",      r"$\max N_k / k$"),
        (axes[2], r"Anti-hub fraction (% with $N_k=0$)", "anti-hubs (%)"),
    ]:
        ax.set_title(ttl)
        ax.set_xlabel("k")
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10, frameon=False)

    fig.suptitle(
        "Hubness of Wikipedia BERT-tiny 128d embeddings  (filtered vocab, V=10440)",
        y=1.03, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = ROOT / "charts/hubness_wiki_bert128d.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")
    return out


# ---------- LNS ------------------------------------------------------------- #

def knn_sets(e: np.ndarray, k: int, metric: str) -> np.ndarray:
    v = e.shape[0]
    if metric == "euclidean":
        sq = (e * e).sum(axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (e @ e.T)
        np.fill_diagonal(d2, np.inf)
        return np.argpartition(d2, kth=k, axis=1)[:, :k].astype(np.int64)
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


def plot_lns(lns: np.ndarray, tokens: list[str], k: int,
             random_baseline: float) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    ax = axes[0]
    bins = np.linspace(0, max(0.05, lns.max() + 1e-3), 50)
    ax.hist(lns, bins=bins, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.axvline(float(lns.mean()), color="black", lw=1.5, ls="--",
               label=f"mean = {lns.mean():.3f}")
    ax.axvline(float(np.median(lns)), color="gray", lw=1.2, ls=":",
               label=f"median = {np.median(lns):.3f}")
    ax.axvline(random_baseline, color="red", lw=1.2, ls="-.",
               label=f"random = {random_baseline:.4f}")
    ax.set_xlabel(f"LNS(w)  (Jaccard over k={k} cosine neighbours)")
    ax.set_ylabel("# tokens")
    ax.set_title("LNS distribution")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    xs = np.sort(lns); ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color="#4c72b0", lw=2)
    for q_val, label in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
        qq = float(np.quantile(lns, q_val))
        ax.axvline(qq, color="gray", lw=0.8, ls=":")
        ax.text(qq, q_val, f" {label}={qq:.2f}", fontsize=8, va="center")
    ax.set_xlabel("LNS(w)")
    ax.set_ylabel("cumulative fraction of tokens")
    ax.set_title("CDF of LNS")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    order_high = np.argsort(-lns)[:18]
    y = np.arange(len(order_high))[::-1]
    ax.barh(y, lns[order_high], color="#55a868", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels([tokens[i] for i in order_high], fontsize=8)
    ax.set_xlabel("LNS(w)")
    ax.set_title("Top-18 LNS tokens (most-preserved local geometry)")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"Local Neighborhood Similarity (Wikipedia BERT-tiny, "
        f"k={k}, cosine, V={len(tokens)})",
        y=1.02, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = ROOT / f"charts/lns_wiki_bert128d_k{k}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")
    return out


# ---------- main ------------------------------------------------------------ #

def main() -> None:
    print("Loading Wikipedia BERT-128d embeddings:")
    ea = load_vectors(A_VEC); eb = load_vectors(B_VEC)
    ta = load_tokens(A_TOK);  tb = load_tokens(B_TOK)
    assert ta == tb, "Vocabs of frozen and e2e must match for LNS."
    tokens = ta; v = len(tokens); d = ea.shape[1]
    print(f"vocab={v}  dim={d}")
    print(f"norms: frozen mean={np.linalg.norm(ea, axis=1).mean():.3f} "
          f"std={np.linalg.norm(ea, axis=1).std():.3f}   "
          f"e2e    mean={np.linalg.norm(eb, axis=1).mean():.3f} "
          f"std={np.linalg.norm(eb, axis=1).std():.3f}")

    # Hubness scan.
    hub_results = {}
    for name, e in ((A_NAME, ea), (B_NAME, eb)):
        hub_results[name] = hubness_scan(name, e, tokens)

    out_json = ROOT / "charts/hubness_wiki_bert128d.json"
    out_json.write_text(json.dumps({
        name: {str(k): {kk: vv for kk, vv in row.items() if kk != "top_hubs"}
               | {"top_hubs": row["top_hubs"]}
               for k, row in res.items()}
        for name, res in hub_results.items()
    }, indent=2))
    print(f"saved hubness stats -> {out_json}")
    plot_hubness(hub_results)

    # Hubness comparison print.
    print("\n=== Hubness summary (S_k) ===")
    print(f"{'k':>4}  {'frozen':>8}  {'e2e':>8}  {'Δ':>6}")
    for k in K_HUBNESS:
        f = hub_results[A_NAME][k]["skew"]
        e = hub_results[B_NAME][k]["skew"]
        print(f"{k:>4}  {f:8.3f}  {e:8.3f}  {e-f:+6.3f}")

    # LNS at each k.
    for k in K_LNS:
        print(f"\n=== LNS  k={k}  cosine ===")
        t0 = time.time()
        nn_a = knn_sets(ea, k, LNS_METRIC)
        nn_b = knn_sets(eb, k, LNS_METRIC)
        l = jaccard_batch(nn_a, nn_b)
        rnd = (k * k / (v - 1)) / (2 * k - k * k / (v - 1))
        print(f"  mean={l.mean():.4f}  median={np.median(l):.4f}  "
              f"std={l.std():.4f}  random={rnd:.4f}  "
              f"mean/random={l.mean()/rnd:.1f}x   [{time.time()-t0:.1f}s]")
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            print(f"  p{int(q*100):02d} = {np.quantile(l, q):.4f}")

        order_high = np.argsort(-l)[:15]
        order_low = np.argsort(l)[:15]
        print("  top-15 most preserved:")
        for i in order_high:
            print(f"    LNS={l[i]:.3f}  {tokens[i]!r}")
        print("  top-15 least preserved (LNS=0 means no shared neighbours):")
        for i in order_low:
            print(f"    LNS={l[i]:.3f}  {tokens[i]!r}")

        out_tsv = ROOT / f"charts/lns_wiki_bert128d_k{k}_per_token.tsv"
        with out_tsv.open("w") as f:
            f.write("token\tlns\n")
            for tok, ll in zip(tokens, l):
                f.write(f"{tok}\t{ll:.6f}\n")
        out_json = ROOT / f"charts/lns_wiki_bert128d_k{k}.json"
        out_json.write_text(json.dumps(dict(
            k=k, metric=LNS_METRIC, vocab=v, dim=d,
            mean=float(l.mean()), median=float(np.median(l)),
            std=float(l.std()),
            quantiles={f"p{int(q*100):02d}": float(np.quantile(l, q))
                       for q in (0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95)},
            random_baseline=rnd,
        ), indent=2))
        print(f"  saved -> {out_tsv}, {out_json}")
        plot_lns(l, tokens, k, rnd)


if __name__ == "__main__":
    main()
