"""Extract and compare embeddings from e2e and emb trained models.

e2e model: embeddings are inside the diffusion model checkpoint (word_embedding.weight),
           jointly trained with the diffusion model.
emb model: embeddings are in random_emb.torch, frozen during diffusion training.

Both models share the same vocab.json and the same random_emb.torch initialization.
"""

import argparse
import json
import torch
import numpy as np

E2E_DIR = "trained_models/diff_e2e-tgt_pad_rand64_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_e2e_e2e_training"
EMB_DIR = "trained_models/diff_e2e-tgt_pad_rand64_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_e2e_emb_training"


def load_vocab(model_dir):
    with open(f"{model_dir}/vocab.json", "r") as f:
        vocab = json.load(f)
    id2word = {v: k for k, v in vocab.items()}
    return vocab, id2word


def load_e2e_embeddings(model_dir, checkpoint="ema_0.9999_200000.pt"):
    state = torch.load(f"{model_dir}/{checkpoint}", map_location="cpu")
    return state["word_embedding.weight"]


def load_emb_embeddings(model_dir):
    state = torch.load(f"{model_dir}/random_emb.torch", map_location="cpu")
    return state["weight"]


def load_initial_embeddings(model_dir):
    """Load the frozen embeddings (random_emb.torch) which are the initialization for both models."""
    return load_emb_embeddings(model_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from e2e and emb models")
    parser.add_argument("--e2e_dir", default=E2E_DIR)
    parser.add_argument("--emb_dir", default=EMB_DIR)
    parser.add_argument("--output", default="embeddings_comparison.npz",
                        help="Output file for embeddings (.npz)")
    parser.add_argument("--show_top_k", type=int, default=20,
                        help="Number of most-changed words to show")
    args = parser.parse_args()

    vocab, id2word = load_vocab(args.e2e_dir)
    print(f"Vocab size: {len(vocab)}")

    emb_e2e = load_e2e_embeddings(args.e2e_dir)
    emb_frozen = load_emb_embeddings(args.emb_dir)
    emb_init = load_initial_embeddings(args.e2e_dir)

    print(f"e2e embeddings:    {emb_e2e.shape} (from model checkpoint)")
    print(f"emb embeddings:    {emb_frozen.shape} (frozen random_emb.torch)")
    print(f"e2e init:          {emb_init.shape} (random_emb.torch in e2e dir)")

    print(f"\n--- Norms ---")
    print(f"e2e mean norm:     {torch.norm(emb_e2e, dim=-1).mean():.4f}")
    print(f"emb mean norm:     {torch.norm(emb_frozen, dim=-1).mean():.4f}")
    print(f"e2e init mean norm:{torch.norm(emb_init, dim=-1).mean():.4f}")

    init_match = torch.allclose(emb_frozen, emb_init, atol=1e-6)
    print(f"\nemb == e2e_init (same initialization): {init_match}")

    diff = emb_e2e - emb_init
    per_word_drift = torch.norm(diff, dim=-1)
    print(f"\n--- e2e drift from initialization ---")
    print(f"Mean L2 drift:     {per_word_drift.mean():.4f}")
    print(f"Max L2 drift:      {per_word_drift.max():.4f}")
    print(f"Min L2 drift:      {per_word_drift.min():.4f}")

    cos_sim = torch.nn.functional.cosine_similarity(emb_e2e, emb_init, dim=-1)
    print(f"Mean cosine sim:   {cos_sim.mean():.4f}")

    topk = torch.topk(per_word_drift, k=min(args.show_top_k, len(vocab)))
    print(f"\nTop {args.show_top_k} most changed words (e2e vs init):")
    for i, (idx, drift) in enumerate(zip(topk.indices, topk.values)):
        word = id2word[idx.item()]
        sim = cos_sim[idx].item()
        print(f"  {i+1:3d}. {word:20s}  drift={drift:.4f}  cos_sim={sim:.4f}")

    bottomk = torch.topk(per_word_drift, k=min(args.show_top_k, len(vocab)), largest=False)
    print(f"\nTop {args.show_top_k} least changed words (e2e vs init):")
    for i, (idx, drift) in enumerate(zip(bottomk.indices, bottomk.values)):
        word = id2word[idx.item()]
        sim = cos_sim[idx].item()
        print(f"  {i+1:3d}. {word:20s}  drift={drift:.4f}  cos_sim={sim:.4f}")

    np.savez(
        args.output,
        e2e=emb_e2e.detach().numpy(),
        emb=emb_frozen.detach().numpy(),
        init=emb_init.detach().numpy(),
        vocab=json.dumps(vocab),
    )
    print(f"\nSaved embeddings to {args.output}")


if __name__ == "__main__":
    main()
