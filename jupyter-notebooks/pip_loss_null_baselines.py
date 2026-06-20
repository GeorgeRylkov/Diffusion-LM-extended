"""
Random reference baselines for PIP loss.

For each observed pair (e2e vs frozen), we construct several null pairs that
share some statistics but have unrelated geometry, and compute the same PIP
metrics.  This gives us a meaningful scale: if the observed pair's PIP is close
to a null PIP, the two geometries are effectively unrelated; if it's much
smaller, they still share structure.

Nulls tested, for the pair (E1, E2) with row-norm vectors n1, n2:

  (A) E1 vs random-Gaussian-with-matching-row-norms(E2):
        take random unit directions for each of V rows, scale to |E2|'s norms.
        -> "what if the other matrix were a random embedding with the same
        row-norm spectrum?"

  (B) E1_shuf vs E2:
        randomly permute the vocabulary of E1 (break the token alignment).
        Gram(E1_shuf) has the same eigenvalues as Gram(E1), same ||G||_F; only
        the row-to-row correspondence is destroyed.
        -> "what if E1's geometry were identical but we scrambled the token
        identities?"  This is the purest test of whether PIP measures
        alignment rather than global scale.

  (C) Random-Gaussian with matching row-norms, twice:
        two completely independent random embeddings of the same norm spectrum.
        -> "what PIP looks like for two unrelated embedding spaces".

We print:  observed PIP, relPIP vs ||G2||_F, Gram cosine distance, and the
same for each null, all with the same block-size implementation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.parent.name}  {arr.shape}  in {time.time()-t0:.1f}s",
          flush=True)
    return arr


def compute_pip(e1: np.ndarray, e2: np.ndarray, block: int = 1024) -> dict:
    v = e1.shape[0]
    diff = g1 = g2 = inner = 0.0
    for s in range(0, v, block):
        r1 = e1[s:s + block] @ e1.T
        r2 = e2[s:s + block] @ e2.T
        d = r1 - r2
        diff += float(np.einsum("ij,ij->", d, d, dtype=np.float64))
        g1 += float(np.einsum("ij,ij->", r1, r1, dtype=np.float64))
        g2 += float(np.einsum("ij,ij->", r2, r2, dtype=np.float64))
        inner += float(np.einsum("ij,ij->", r1, r2, dtype=np.float64))
    pip = float(np.sqrt(diff))
    g1 = float(np.sqrt(g1))
    g2 = float(np.sqrt(g2))
    cos = inner / (g1 * g2) if g1 and g2 else 0.0
    return dict(pip=pip, g1=g1, g2=g2, cos=cos, dist=1.0 - cos,
                rel_g2=pip / g2 if g2 else float("nan"),
                rel_sym=pip / np.sqrt(g1 * g2) if g1 and g2 else float("nan"))


def random_like(e_like: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random Gaussian embedding with the same row-norm distribution as `e_like`."""
    v, d = e_like.shape
    z = rng.standard_normal((v, d)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    target_norms = np.linalg.norm(e_like, axis=1, keepdims=True)
    return (z * target_norms).astype(np.float32)


def shuffled(e: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-shuffle copy of e (destroys token alignment, preserves geometry)."""
    perm = rng.permutation(e.shape[0])
    return e[perm]


def fmt(res: dict) -> str:
    return (f"PIP={res['pip']:.1f}  ||G1||={res['g1']:.1f}  ||G2||={res['g2']:.1f}  "
            f"relPIP(g2)={res['rel_g2']:.3f}  relPIP(sym)={res['rel_sym']:.3f}  "
            f"gram-cos-dist={res['dist']:.3f}")


def compare_pair(name1: str, name2: str, e1: np.ndarray, e2: np.ndarray,
                 n_seeds: int = 5) -> None:
    print(f"\n####### {name1}  vs  {name2}  (V={e1.shape[0]}, d={e1.shape[1]}) #######")

    # Observed.
    obs = compute_pip(e1, e2)
    print(f"  OBSERVED                          : {fmt(obs)}")

    # (A) e1 vs random-Gaussian with e2's row-norms.
    # (B) shuffled-e1 vs e2.
    # (C) random vs random, both with respective row-norm spectra.
    def run(null_name: str, maker, seeds):
        stats = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            e1p, e2p = maker(rng)
            stats.append(compute_pip(e1p, e2p))
        pip = np.mean([s["pip"] for s in stats])
        g1 = np.mean([s["g1"] for s in stats])
        g2 = np.mean([s["g2"] for s in stats])
        dist = np.mean([s["dist"] for s in stats])
        rg2 = np.mean([s["rel_g2"] for s in stats])
        rsym = np.mean([s["rel_sym"] for s in stats])
        pip_std = np.std([s["pip"] for s in stats])
        dist_std = np.std([s["dist"] for s in stats])
        print(f"  {null_name:34s}: PIP={pip:.1f}±{pip_std:.1f}  "
              f"||G1||={g1:.1f}  ||G2||={g2:.1f}  "
              f"relPIP(g2)={rg2:.3f}  relPIP(sym)={rsym:.3f}  "
              f"gram-cos-dist={dist:.3f}±{dist_std:.3f}")
        return dict(pip=pip, dist=dist, rel_sym=rsym)

    seeds = list(range(n_seeds))

    null_A = run("(A) E1 vs RandGauss(E2-norms)",
                 lambda rng: (e1, random_like(e2, rng)),
                 seeds)
    null_B = run("(B) shuffled(E1) vs E2       ",
                 lambda rng: (shuffled(e1, rng), e2),
                 seeds)
    null_C = run("(C) RandGauss(E1) vs RandGauss(E2)",
                 lambda rng: (random_like(e1, rng), random_like(e2, rng)),
                 seeds)

    # Position observed relative to nulls.
    print("  ------------------------------------------------------------------")
    if null_C["pip"] > 0:
        frac_to_C = obs["pip"] / null_C["pip"]
        print(f"  observed PIP is {frac_to_C:.2%} of null-(C) 'two random '"
              f"embeddings' PIP  (1.0 = just as different as random)")
    if null_B["pip"] > 0:
        frac_to_B = obs["pip"] / null_B["pip"]
        print(f"  observed PIP is {frac_to_B:.2%} of null-(B) 'token-scrambled' PIP "
              f"(1.0 = no more aligned than a random token permutation)")


def main() -> None:
    root = Path("/Users/grylkov/git/hse/Diffusion-LM/embeddings")
    pairs = [
        ("bert_e2e_128d",
         root / "bert_128d_filtered/e2e_vectors.tsv",
         "bert_frozen_v2_128d",
         root / "bert_frozen_v2_128d_filtered/bert_frozen_v2_vectors.tsv"),
        ("gpt2_e2e_128d",
         root / "gpt2_e2e_128d_filtered/gpt2_e2e_vectors.tsv",
         "gpt2_pca_frozen_128d",
         root / "gpt2_pca_frozen_128d_filtered/gpt2_pca_frozen_vectors.tsv"),
    ]

    print("Loading:")
    for (n1, p1, n2, p2) in pairs:
        e1 = load_vectors(p1)
        e2 = load_vectors(p2)
        compare_pair(n1, n2, e1, e2)


if __name__ == "__main__":
    main()
