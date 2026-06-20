"""Check per-dimension variance of embeddings across models."""

import os
import sys
import torch
import numpy as np

models = {
    "BERT frozen 768d": {
        "emb_path": "scripts/diffusion_models/diff_roc_pad_bert768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_v2/random_emb.torch",
        "ckpt_path": "scripts/diffusion_models/diff_roc_pad_bert768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_v2/ema_0.9999_400000.pt",
        "e2e": False,
    },
    "BERT random e2e 768d": {
        "emb_path": "scripts/diffusion_models/diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_v2/random_emb.torch",
        "ckpt_path": "scripts/diffusion_models/diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_v2/ema_0.9999_400000.pt",
        "e2e": True,
    },
    "BERT-uncased random e2e 128d": {
        "emb_path": "scripts/diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased/random_emb.torch",
        "ckpt_path": "scripts/diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased/ema_0.9999_400000.pt",
        "e2e": True,
    },
}


def analyze_embeddings(name, weight):
    """Print statistics about an embedding matrix [vocab_size, emb_dim]."""
    per_dim_var = weight.var(dim=0)
    per_dim_mean = weight.mean(dim=0)
    per_token_norm = weight.norm(dim=-1)

    print(f"  Shape: {list(weight.shape)}")
    print(f"  Per-dim variance:  mean={per_dim_var.mean():.4f}, std={per_dim_var.std():.4f}, "
          f"min={per_dim_var.min():.4f}, max={per_dim_var.max():.4f}")
    print(f"  Per-dim mean:      mean={per_dim_mean.mean():.4f}, std={per_dim_mean.std():.4f}")
    print(f"  Per-token L2 norm: mean={per_token_norm.mean():.4f}, std={per_token_norm.std():.4f}, "
          f"min={per_token_norm.min():.4f}, max={per_token_norm.max():.4f}")
    print(f"  Overall mean: {weight.mean():.6f}")
    print(f"  Overall std:  {weight.std():.6f}")
    return per_dim_var.mean().item(), per_token_norm.mean().item()


print("=" * 70)
print("EMBEDDING ANALYSIS")
print("=" * 70)

summary = {}

for name, info in models.items():
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    # Initial / preprocessing embeddings
    print(f"\n  --- Initial embeddings (random_emb.torch) ---")
    state = torch.load(info["emb_path"], map_location="cpu", weights_only=True)
    init_weight = state["weight"]
    init_var, init_norm = analyze_embeddings(name, init_weight)

    # For e2e models, also load the learned embeddings from the checkpoint
    if info["e2e"] and os.path.exists(info["ckpt_path"]):
        print(f"\n  --- Learned embeddings (from checkpoint word_embedding.weight) ---")
        ckpt = torch.load(info["ckpt_path"], map_location="cpu", weights_only=True)
        if "word_embedding.weight" in ckpt:
            learned_weight = ckpt["word_embedding.weight"]
            learned_var, learned_norm = analyze_embeddings(name, learned_weight)

            delta = learned_weight - init_weight
            print(f"\n  --- Drift from initial ---")
            print(f"  Mean absolute change per param: {delta.abs().mean():.6f}")
            print(f"  Norm of change: {delta.norm():.4f}")
            print(f"  Relative change: {delta.norm() / init_weight.norm():.4f}")
            summary[name] = (learned_var, learned_norm)
        else:
            print("  word_embedding.weight not found in checkpoint keys")
            print(f"  Available keys with 'emb' or 'word': "
                  f"{[k for k in ckpt.keys() if 'emb' in k.lower() or 'word' in k.lower()]}")
            summary[name] = (init_var, init_norm)
    else:
        summary[name] = (init_var, init_norm)

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"{'Model':<35} {'Per-dim Var':>12} {'Mean L2 Norm':>13}")
print("-" * 62)
for name, (var, norm) in summary.items():
    print(f"{name:<35} {var:>12.4f} {norm:>13.4f}")
