"""
LNS vs drift scatter.

We previously computed:
  - LNS(w)  : per-token Jaccard between the k=50 Euclidean neighbourhoods of
              bert_frozen_v2 and bert_e2e (1 = neighbourhood preserved).
  - drift_l2(w), cos_sim(w) : how far a token's vector moved from frozen to e2e,
              both as Euclidean distance and as cosine similarity between the
              two vectors (1 = direction preserved).

This script plots LNS vs both drift measures, colouring by corpus frequency,
and annotates a handful of extreme / representative tokens.

Hypotheses:
  - Tokens with HIGHER drift_l2 (moved further) -> LOWER LNS.
  - Tokens with HIGHER cos_sim (direction preserved) -> HIGHER LNS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM")
DRIFT_TSV = ROOT / "charts/drift_per_token_bert128_v2.csv"   # actually tab-sep
LNS_TSV   = ROOT / "charts/lns_bert_per_token.tsv"
OUT_PNG   = ROOT / "charts/lns_vs_drift_bert128_v2.png"


# ---------- tiny helpers ---------------------------------------------------- #

def read_tsv(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Return (column_order, {col: [values...]})  for a tab-sep file with header."""
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    cols: dict[str, list[str]] = {c: [] for c in header}
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) != len(header):
            continue
        for c, v in zip(header, parts):
            cols[c].append(v)
    return header, cols


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


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def binned_stats(x: np.ndarray, y: np.ndarray, n_bins: int = 14,
                 xlog: bool = False) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    if xlog:
        xv = np.log10(np.maximum(x, 1e-12))
    else:
        xv = x
    edges = np.linspace(xv.min(), xv.max(), n_bins + 1)
    bi = np.clip(np.digitize(xv, edges[1:-1]), 0, n_bins - 1)
    centers, med, p25, p75 = [], [], [], []
    for b in range(n_bins):
        sel = bi == b
        if sel.sum() < 5:
            continue
        c = 0.5 * (edges[b] + edges[b + 1])
        centers.append(10 ** c if xlog else c)
        med.append(float(np.median(y[sel])))
        p25.append(float(np.quantile(y[sel], 0.25)))
        p75.append(float(np.quantile(y[sel], 0.75)))
    return (np.array(centers), np.array(med), np.array(p25), np.array(p75))


# ---------- load and join --------------------------------------------------- #

_, dcols = read_tsv(DRIFT_TSV)
_, lcols = read_tsv(LNS_TSV)

drift_map = {
    tok: (float(l2), float(cs), int(cnt))
    for tok, l2, cs, cnt in zip(dcols["token"], dcols["drift_l2"],
                                dcols["cos_sim"], dcols["count"])
}
lns_map = {
    tok: (float(l), int(fr))
    for tok, l, fr in zip(lcols["token"], lcols["lns"], lcols["frequency"])
}
common = [t for t in lcols["token"] if t in drift_map]
missing = [t for t in lcols["token"] if t not in drift_map]
print(f"tokens in LNS table: {len(lcols['token'])}")
print(f"tokens in drift table: {len(drift_map)}")
print(f"joined on token:       {len(common)}   (missing from drift: {len(missing)})")

tokens  = np.array(common)
drift_l2 = np.array([drift_map[t][0] for t in common])
cos_sim  = np.array([drift_map[t][1] for t in common])
freq     = np.array([lns_map[t][1] for t in common])   # corpus frequency from LNS file
lns      = np.array([lns_map[t][0] for t in common])

# ---------- correlations ---------------------------------------------------- #

print("\nCorrelations (n = {:,}):".format(len(common)))
print(f"  LNS vs drift_l2 :  Pearson = {pearson(lns, drift_l2):+.3f}   "
      f"Spearman = {spearman(lns, drift_l2):+.3f}")
print(f"  LNS vs cos_sim  :  Pearson = {pearson(lns, cos_sim):+.3f}    "
      f"Spearman = {spearman(lns, cos_sim):+.3f}")
print(f"  LNS vs log(freq+1):Pearson = {pearson(lns, np.log10(freq+1)):+.3f}    "
      f"Spearman = {spearman(lns, freq.astype(float)):+.3f}")
print(f"  drift_l2 vs log(freq+1): Spearman = "
      f"{spearman(drift_l2, freq.astype(float)):+.3f}")

rho_l2 = spearman(lns, drift_l2)
rho_cs = spearman(lns, cos_sim)


# ---------- plot ------------------------------------------------------------ #

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

# Colour scale: log frequency.
cval = np.log10(np.maximum(freq, 1))
cmap = plt.cm.viridis

# --- Panel 1: LNS vs drift_l2 --- #
ax = axes[0]
sc = ax.scatter(drift_l2, lns, c=cval, cmap=cmap, s=9, alpha=0.55,
                edgecolor="none")
# Binned median trend line.
c, m, p25, p75 = binned_stats(drift_l2, lns, n_bins=16)
ax.plot(c, m, color="black", lw=2, label="binned median")
ax.fill_between(c, p25, p75, color="black", alpha=0.15, label="IQR")
ax.set_xlabel("drift L2  (||e2e - frozen||)")
ax.set_ylabel("LNS(w)")
ax.set_title(f"LNS vs L2 drift\n"
             f"Spearman rho = {rho_l2:+.3f}  "
             f"(further drift -> smaller LNS)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9, frameon=False)
cb = fig.colorbar(sc, ax=ax, shrink=0.9)
cb.set_label("log10(corpus frequency)")

# Annotate a handful of interesting tokens.
order_hi_lns = np.argsort(-lns)[:6]
order_lo_lns = np.argsort(lns)[:6]
# Tokens with high drift yet high LNS are the most surprising -> annotate too.
interest = set(order_hi_lns.tolist() + order_lo_lns.tolist())
# Also pick the 4 tokens with the largest drift_l2.
interest |= set(np.argsort(-drift_l2)[:4].tolist())
for i in interest:
    ax.annotate(tokens[i], (drift_l2[i], lns[i]),
                fontsize=8, alpha=0.85,
                xytext=(3, 3), textcoords="offset points")

# --- Panel 2: LNS vs cos_sim --- #
ax = axes[1]
sc = ax.scatter(cos_sim, lns, c=cval, cmap=cmap, s=9, alpha=0.55,
                edgecolor="none")
c, m, p25, p75 = binned_stats(cos_sim, lns, n_bins=16)
ax.plot(c, m, color="black", lw=2, label="binned median")
ax.fill_between(c, p25, p75, color="black", alpha=0.15, label="IQR")
ax.axvline(0, color="red", lw=0.8, ls=":", alpha=0.7)
ax.set_xlabel("cosine(frozen, e2e)")
ax.set_ylabel("LNS(w)")
ax.set_title(f"LNS vs cosine(frozen, e2e)\n"
             f"Spearman rho = {rho_cs:+.3f}  "
             f"(direction preserved -> larger LNS)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9, frameon=False)
cb = fig.colorbar(sc, ax=ax, shrink=0.9)
cb.set_label("log10(corpus frequency)")

for i in interest:
    ax.annotate(tokens[i], (cos_sim[i], lns[i]),
                fontsize=8, alpha=0.85,
                xytext=(3, 3), textcoords="offset points")

fig.suptitle(
    "Per-token LNS vs drift (bert_frozen_v2 -> bert_e2e, k=50 Euclidean)",
    y=1.02, fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print(f"\nsaved plot -> {OUT_PNG}")


# ---------- compact tables -------------------------------------------------- #

def show(order: np.ndarray, title: str, n: int = 12) -> None:
    print(f"\n{title}")
    print(f"  {'token':<14s} {'freq':>7s} {'drift_l2':>9s} {'cos_sim':>8s} "
          f"{'LNS':>6s}")
    for i in order[:n]:
        print(f"  {tokens[i]:<14s} {int(freq[i]):>7d} {drift_l2[i]:>9.3f} "
              f"{cos_sim[i]:>+8.3f} {lns[i]:>6.3f}")


show(np.argsort(-drift_l2),
     "Top-12 by drift_l2 (moved furthest):")
show(np.argsort(drift_l2),
     "Bottom-12 by drift_l2 (moved least):")
show(np.argsort(-cos_sim),
     "Top-12 by cos_sim (direction preserved):")
show(np.argsort(cos_sim),
     "Bottom-12 by cos_sim (direction flipped):")
