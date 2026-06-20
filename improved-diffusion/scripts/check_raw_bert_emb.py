"""Check raw BERT embedding stats before and after z-normalization."""
import torch
from transformers import BertModel

BERT_MODEL = "bert-base-cased"
print(f"Loading {BERT_MODEL}...")
bert = BertModel.from_pretrained(BERT_MODEL)
raw = bert.embeddings.word_embeddings.weight.data.clone()
del bert

print(f"\n--- Raw BERT embeddings (before z-norm) ---")
print(f"Shape: {list(raw.shape)}")
per_dim_var = raw.var(dim=0)
per_dim_mean = raw.mean(dim=0)
per_token_norm = raw.norm(dim=-1)
print(f"Per-dim variance:  mean={per_dim_var.mean():.4f}, std={per_dim_var.std():.4f}")
print(f"Per-dim mean:      mean={per_dim_mean.mean():.4f}, std={per_dim_mean.std():.4f}")
print(f"Per-token L2 norm: mean={per_token_norm.mean():.4f}, std={per_token_norm.std():.4f}")
print(f"Overall mean: {raw.mean():.6f}")
print(f"Overall std:  {raw.std():.6f}")

# Simulate the z-norm as in load_bert_embeddings (full-vocab stats)
norm_mean = raw.mean(dim=0)
norm_std = raw.std(dim=0).clamp(min=1e-6)
normalized = (raw - norm_mean) / norm_std

print(f"\n--- After z-norm (full vocab, for reference) ---")
per_dim_var_n = normalized.var(dim=0)
per_token_norm_n = normalized.norm(dim=-1)
print(f"Per-dim variance:  mean={per_dim_var_n.mean():.4f}")
print(f"Per-token L2 norm: mean={per_token_norm_n.mean():.4f}")
print(f"Overall mean: {normalized.mean():.6f}")
print(f"Overall std:  {normalized.std():.6f}")

print(f"\n--- Scale factor from z-norm ---")
print(f"Per-dim std used for normalization: mean={norm_std.mean():.4f}, min={norm_std.min():.4f}, max={norm_std.max():.4f}")
print(f"Norm shrinkage ratio: {per_token_norm_n.mean() / per_token_norm.mean():.4f}")
