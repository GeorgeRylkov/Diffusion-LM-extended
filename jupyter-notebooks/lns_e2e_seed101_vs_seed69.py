"""
LNS between two end-to-end ROCStories embeddings trained with different RNG seeds.

    embeddings/bert_e2e_sd101_e2e_sd69_filtered/e2e_vectors_sd_101.tsv
    embeddings/bert_e2e_sd101_e2e_sd69_filtered/e2e_vectors_sd_69.tsv

Same filtered vocab (V = 5180, d = 128). Cosine k-NN, Jaccard LNS as in the
Embedding Comparator paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/grylkov/git/hse/Diffusion-LM/embeddings/bert_e2e_sd101_e2e_sd69_filtered")
E101 = ROOT / "e2e_vectors_sd_101.tsv"
E69 = ROOT / "e2e_vectors_sd_69.tsv"
META_A = ROOT / "e2e_sd_101_metadata.tsv"
META_B = ROOT / "e2e_sd_69_metadata.tsv"
OUT_JSON = ROOT.parent.parent / "charts" / "lns_bert_e2e_sd101_vs_sd69.json"

K_VALUES = (10, 50)


def load_vectors(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32, delimiter="\t")


def load_tokens(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text().splitlines()]


def knn_sets(e: np.ndarray, k: int) -> np.ndarray:
    v = e.shape[0]
    en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    sim = en @ en.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(sim, kth=v - 1 - k, axis=1)[:, v - k:]
    return np.sort(idx.astype(np.int64), axis=1)


def jaccard_batch(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    v = a.shape[0]
    out = np.empty(v, dtype=np.float64)
    for i in range(v):
        inter = np.intersect1d(a[i], b[i], assume_unique=True).size
        out[i] = inter / (2 * k - inter)
    return out


def main() -> None:
    ta = load_tokens(META_A)
    tb = load_tokens(META_B)
    assert ta == tb, "Vocab order must match for LNS."
    e101 = load_vectors(E101)
    e69 = load_vectors(E69)
    assert e101.shape == e69.shape
    v, d = e101.shape

    summary: dict[str, object] = {}
    for k in K_VALUES:
        lns = jaccard_batch(knn_sets(e101, k), knn_sets(e69, k), k)
        rnd = (k * k / (v - 1)) / (2 * k - k * k / (v - 1))
        summary[str(k)] = dict(
            k=k,
            metric="cosine",
            vocab=v,
            dim=d,
            mean=float(lns.mean()),
            median=float(np.median(lns)),
            std=float(lns.std()),
            quantiles={
                "p05": float(np.quantile(lns, 0.05)),
                "p25": float(np.quantile(lns, 0.25)),
                "p50": float(np.quantile(lns, 0.50)),
                "p75": float(np.quantile(lns, 0.75)),
                "p95": float(np.quantile(lns, 0.95)),
            },
            min=float(lns.min()),
            max=float(lns.max()),
            random_baseline=float(rnd),
            mean_over_random=float(lns.mean() / rnd),
        )
        print(f"k={k}: mean={lns.mean():.4f}  median={np.median(lns):.4f}  "
              f"mean/random={lns.mean()/rnd:.1f}x")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
