"""
Cross-vocabulary PIP loss comparison.

Given two embedding matrices with different tokenizers, intersect their
vocabularies on whole-word matches (stripping GPT-2's leading U+0120 byte-level
space marker and BERT's '##' subword marker, then case-folding), keep only
unambiguous one-to-one pairs, and compute the standard PIP loss on the
resulting aligned sub-vocabulary.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np


def load_meta(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in Path(path).read_text().splitlines()]


def load_vectors(path: Path) -> np.ndarray:
    t0 = time.time()
    arr = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(f"  loaded {path.name} -> {arr.shape} in {time.time() - t0:.1f}s",
          flush=True)
    return arr


def normalize_bert(tok: str) -> str | None:
    """Return a normalized form for a BERT-uncased token, or None for subpieces."""
    if tok.startswith("##"):
        return None
    return tok.lower()


def normalize_gpt2(tok: str) -> str:
    """Strip the GPT-2 byte-level space marker and case-fold."""
    if tok.startswith("\u0120"):
        tok = tok[1:]
    return tok.lower()


def align_vocabularies(bert_meta: list[str], gpt2_meta: list[str]) -> tuple[list[int], list[int], list[str]]:
    """Return indices (i_bert, i_gpt2) and token strings for one-to-one matches."""
    bert_map: dict[str, list[int]] = {}
    for i, t in enumerate(bert_meta):
        n = normalize_bert(t)
        if n is None:
            continue
        bert_map.setdefault(n, []).append(i)
    gpt2_map: dict[str, list[int]] = {}
    for i, t in enumerate(gpt2_meta):
        n = normalize_gpt2(t)
        gpt2_map.setdefault(n, []).append(i)

    shared = sorted(set(bert_map) & set(gpt2_map))
    pairs = [
        (bert_map[w][0], gpt2_map[w][0], w)
        for w in shared
        if len(bert_map[w]) == 1 and len(gpt2_map[w]) == 1
    ]
    i_bert = [p[0] for p in pairs]
    i_gpt2 = [p[1] for p in pairs]
    words = [p[2] for p in pairs]
    return i_bert, i_gpt2, words


def compute_pip(e1: np.ndarray, e2: np.ndarray, block_size: int = 1024) -> dict:
    assert e1.shape[0] == e2.shape[0]
    vocab = e1.shape[0]
    diff_sq = g1_sq = g2_sq = inner = 0.0
    for start in range(0, vocab, block_size):
        end = min(start + block_size, vocab)
        r1 = e1[start:end] @ e1.T
        r2 = e2[start:end] @ e2.T
        d = r1 - r2
        diff_sq += float(np.einsum("ij,ij->", d, d, dtype=np.float64))
        g1_sq += float(np.einsum("ij,ij->", r1, r1, dtype=np.float64))
        g2_sq += float(np.einsum("ij,ij->", r2, r2, dtype=np.float64))
        inner += float(np.einsum("ij,ij->", r1, r2, dtype=np.float64))
    pip = float(np.sqrt(diff_sq))
    g1 = float(np.sqrt(g1_sq))
    g2 = float(np.sqrt(g2_sq))
    cos = inner / (g1 * g2) if g1 > 0 and g2 > 0 else 0.0
    return dict(
        pip=pip, g1=g1, g2=g2, inner=inner, cos=cos, dist=1.0 - cos,
        pip_rel_g1=pip / g1 if g1 else float("nan"),
        pip_rel_g2=pip / g2 if g2 else float("nan"),
        pip_sym=pip / float(np.sqrt(g1 * g2)) if g1 and g2 else float("nan"),
    )


def print_block(tag: str, a_name: str, b_name: str, vocab: int, res: dict) -> None:
    print(f"\n=== PIP [{tag}]  ({a_name}  vs  {b_name})  V_shared={vocab} ===")
    print(f"  ||G1||_F={res['g1']:.4f}  ||G2||_F={res['g2']:.4f}")
    print(f"  <G1,G2>_F={res['inner']:.4f}")
    print(f"  PIP=||G1-G2||_F={res['pip']:.4f}")
    print(f"  PIP/||G1||={res['pip_rel_g1']:.4f}  PIP/||G2||={res['pip_rel_g2']:.4f}"
          f"  PIP/sqrt(||G1||||G2||)={res['pip_sym']:.4f}")
    print(f"  Gram cosine sim={res['cos']:.4f}  distance={res['dist']:.4f}")


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "embeddings"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bert-a-meta",
                    default=root / "bert_128d_filtered" / "e2e_metadata.tsv",
                    type=Path)
    ap.add_argument("--bert-a-vecs",
                    default=root / "bert_128d_filtered" / "e2e_vectors.tsv",
                    type=Path)
    ap.add_argument("--bert-a-name", default="bert_e2e_128d_filtered")

    ap.add_argument("--gpt2-b-meta",
                    default=root / "gpt2_e2e_128d_filtered" / "gpt2_e2e_metadata.tsv",
                    type=Path)
    ap.add_argument("--gpt2-b-vecs",
                    default=root / "gpt2_e2e_128d_filtered" / "gpt2_e2e_vectors.tsv",
                    type=Path)
    ap.add_argument("--gpt2-b-name", default="gpt2_e2e_128d_filtered")
    ap.add_argument("--normalize", action="store_true",
                    help="Also report numbers after L2-normalizing every row.")
    args = ap.parse_args()

    print("Loading metadata + vectors...")
    bert_meta = load_meta(args.bert_a_meta)
    gpt2_meta = load_meta(args.gpt2_b_meta)
    bert_vec = load_vectors(args.bert_a_vecs)
    gpt2_vec = load_vectors(args.gpt2_b_vecs)

    i_b, i_g, words = align_vocabularies(bert_meta, gpt2_meta)
    print(f"\nAligned (one-to-one) whole-word intersection: {len(words)} tokens")
    print(f"  examples: {words[:15]}  ...  {words[-5:]}")

    e1 = bert_vec[i_b]
    e2 = gpt2_vec[i_g]
    print(f"  e1 shape={e1.shape}  e2 shape={e2.shape}")

    for name, col in ((args.bert_a_name, e1), (args.gpt2_b_name, e2)):
        norms = np.linalg.norm(col, axis=1)
        print(f"  {name}: row-norm mean={norms.mean():.4f} std={norms.std():.4f} "
              f"range=({norms.min():.4f},{norms.max():.4f})")

    res_raw = compute_pip(e1, e2)
    print_block("raw", args.bert_a_name, args.gpt2_b_name, len(words), res_raw)

    if args.normalize:
        e1n = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
        e2n = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12)
        res_n = compute_pip(e1n, e2n)
        print_block("L2-normalized", args.bert_a_name, args.gpt2_b_name,
                    len(words), res_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
