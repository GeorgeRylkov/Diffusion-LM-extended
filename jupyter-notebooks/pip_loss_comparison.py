"""
PIP loss comparison between two sets of word embeddings.

PIP loss (Yin & Shen, 2018, "On the Dimensionality of Word Embedding"):
    PIP(E1, E2) = || E1 @ E1.T  -  E2 @ E2.T ||_F

Since our vocab V = 30522 makes the V x V Gram matrix ~3.7B entries (~14 GB in
fp32), we never materialize it. Instead we compute the Frobenius norm of the
difference block by block over rows.

Key identity used per block of rows R:
    || G1[R]  -  G2[R] ||_F^2
        = tr( (E1[R] E1^T) (E1 E1[R]^T) )
        + tr( (E2[R] E2^T) (E2 E2[R]^T) )
        - 2 * tr( (E1[R] E1^T) (E2 E2[R]^T) )
        = || E1[R] @ E1.T ||_F^2
        + || E2[R] @ E2.T ||_F^2
        - 2 * < E1[R] @ E1.T , E2[R] @ E2.T >_F

All three terms are ~V * block_size in memory, easy to handle.

We also report a few normalized variants so the raw magnitude of the norms
doesn't dominate the interpretation:
  * PIP / ||G1||_F                  -- relative to "frozen" reference
  * PIP / sqrt(||G1||_F * ||G2||_F) -- symmetric normalization
  * 1 - <G1, G2>_F / (||G1||_F ||G2||_F)  -- 1 minus cosine of Gram matrices
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


def load_embeddings(vectors_path: Path) -> np.ndarray:
    """Load a TSV of shape (V, d) as float32."""
    print(f"  loading {vectors_path} ...", flush=True)
    t0 = time.time()
    arr = np.loadtxt(vectors_path, dtype=np.float32, delimiter="\t")
    print(f"    -> shape={arr.shape}, dtype={arr.dtype}, {time.time() - t0:.1f}s",
          flush=True)
    return arr


def compute_pip_loss(
    e1: np.ndarray,
    e2: np.ndarray,
    block_size: int = 512,
) -> dict:
    """Compute PIP loss and related stats between two embedding matrices.

    Parameters
    ----------
    e1, e2 : np.ndarray, shape (V, d1), (V, d2)
    block_size : rows processed per iteration

    Returns
    -------
    dict with keys: pip_loss, g1_frob, g2_frob, gram_inner,
                    pip_rel_g1, pip_sym, gram_cosine_distance
    """
    assert e1.shape[0] == e2.shape[0], "Vocabularies must match."
    vocab = e1.shape[0]

    # 1st-pass streaming Frobenius norms and Frobenius inner product of Grams.
    # G_i = E_i @ E_i.T.  We accumulate in float64 for numerical safety.
    diff_sq_sum = 0.0  # || G1 - G2 ||_F^2
    g1_sq_sum = 0.0  # || G1 ||_F^2
    g2_sq_sum = 0.0  # || G2 ||_F^2
    inner_sum = 0.0  # < G1, G2 >_F

    n_blocks = (vocab + block_size - 1) // block_size
    t0 = time.time()
    for bi, start in enumerate(range(0, vocab, block_size)):
        end = min(start + block_size, vocab)
        # Row slabs of the two Gram matrices, shape (block, V).
        g1_rows = e1[start:end] @ e1.T  # (b, V)
        g2_rows = e2[start:end] @ e2.T  # (b, V)

        diff = g1_rows - g2_rows
        diff_sq_sum += float(np.einsum("ij,ij->", diff, diff, dtype=np.float64))
        g1_sq_sum += float(np.einsum("ij,ij->", g1_rows, g1_rows, dtype=np.float64))
        g2_sq_sum += float(np.einsum("ij,ij->", g2_rows, g2_rows, dtype=np.float64))
        inner_sum += float(np.einsum("ij,ij->", g1_rows, g2_rows, dtype=np.float64))

        if (bi + 1) % max(1, n_blocks // 20) == 0 or bi == n_blocks - 1:
            elapsed = time.time() - t0
            eta = elapsed / (bi + 1) * (n_blocks - bi - 1)
            print(
                f"    block {bi + 1}/{n_blocks}  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                flush=True,
            )

    pip_loss = float(np.sqrt(diff_sq_sum))
    g1_frob = float(np.sqrt(g1_sq_sum))
    g2_frob = float(np.sqrt(g2_sq_sum))
    cos = inner_sum / (g1_frob * g2_frob) if g1_frob > 0 and g2_frob > 0 else 0.0

    return {
        "pip_loss": pip_loss,
        "g1_frob": g1_frob,
        "g2_frob": g2_frob,
        "gram_inner": float(inner_sum),
        "pip_rel_g1": pip_loss / g1_frob if g1_frob > 0 else float("nan"),
        "pip_rel_g2": pip_loss / g2_frob if g2_frob > 0 else float("nan"),
        "pip_sym": pip_loss / float(np.sqrt(g1_frob * g2_frob))
        if g1_frob > 0 and g2_frob > 0
        else float("nan"),
        "gram_cosine": cos,
        "gram_cosine_distance": 1.0 - cos,
    }


def norm_stats(e: np.ndarray, name: str) -> None:
    norms = np.linalg.norm(e, axis=1)
    print(
        f"  {name}: rows={e.shape[0]} dim={e.shape[1]} "
        f"||row||: mean={norms.mean():.4f} std={norms.std():.4f} "
        f"min={norms.min():.4f} max={norms.max():.4f}"
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_a = root / "embeddings" / "bert_e2e_128d" / "e2e_vectors.tsv"
    default_b = (
        root
        / "embeddings"
        / "bert_frozen_v2_128d"
        / "bert_frozen_v2_vectors.tsv"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=default_a,
                        help="Path to first embedding TSV (E1).")
    parser.add_argument("--b", type=Path, default=default_b,
                        help="Path to second embedding TSV (E2).")
    parser.add_argument("--name-a", type=str, default="bert_e2e_128d")
    parser.add_argument("--name-b", type=str, default="bert_frozen_v2_128d")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument(
        "--also-normalized",
        action="store_true",
        help="Also report PIP loss after L2-normalizing every row of both "
             "embedding matrices (removes norm scale differences).",
    )
    args = parser.parse_args()

    print("Loading embeddings:")
    e1 = load_embeddings(args.a)
    e2 = load_embeddings(args.b)

    if e1.shape[0] != e2.shape[0]:
        print(
            f"ERROR: vocab size mismatch: {e1.shape[0]} vs {e2.shape[0]}",
            file=sys.stderr,
        )
        return 1

    print("\nRow-norm stats:")
    norm_stats(e1, args.name_a)
    norm_stats(e2, args.name_b)

    print("\nComputing PIP loss (raw embeddings)...")
    raw = compute_pip_loss(e1, e2, block_size=args.block_size)
    print_results(args.name_a, args.name_b, raw, tag="raw")

    if args.also_normalized:
        print("\nComputing PIP loss (L2-normalized rows)...")
        e1n = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
        e2n = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)
        norm = compute_pip_loss(e1n, e2n, block_size=args.block_size)
        print_results(args.name_a, args.name_b, norm, tag="L2-normalized")

    return 0


def print_results(name_a: str, name_b: str, res: dict, tag: str) -> None:
    print(f"\n=== PIP loss results [{tag}]  ({name_a}  vs  {name_b}) ===")
    print(f"  ||E1 E1^T||_F           = {res['g1_frob']:.4f}")
    print(f"  ||E2 E2^T||_F           = {res['g2_frob']:.4f}")
    print(f"  <E1 E1^T, E2 E2^T>_F    = {res['gram_inner']:.4f}")
    print(f"  PIP loss  = ||G1-G2||_F = {res['pip_loss']:.4f}")
    print(f"  PIP / ||G1||_F          = {res['pip_rel_g1']:.4f}")
    print(f"  PIP / ||G2||_F          = {res['pip_rel_g2']:.4f}")
    print(f"  PIP / sqrt(|G1||G2|)    = {res['pip_sym']:.4f}")
    print(f"  Gram cosine similarity  = {res['gram_cosine']:.4f}")
    print(f"  Gram cosine distance    = {res['gram_cosine_distance']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
