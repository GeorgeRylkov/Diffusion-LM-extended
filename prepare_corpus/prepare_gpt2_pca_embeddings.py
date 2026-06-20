"""Extract GPT-2 wte embeddings, center, and reduce via PCA for Diffusion-LM."""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2Model, GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GPT-2 PCA embeddings for Diffusion-LM")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save embeddings and tokenizer")
    parser.add_argument("--target_dim", type=int, default=128,
                        choices=[64, 128, 256, 384, 512],
                        help="Target embedding dimensionality after PCA")
    parser.add_argument("--gpt2_model", type=str, default="gpt2",
                        help="GPT-2 model name (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply naive per-dim normalization (uniform over vocab). "
                             "Skip this if you plan to normalize using e2e-learned norms.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load GPT-2 and extract token embedding matrix ---
    print(f"Loading {args.gpt2_model}...")
    model = GPT2Model.from_pretrained(args.gpt2_model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model)
    W = model.wte.weight.detach().numpy()  # [50257, 768]
    del model
    print(f"Extracted wte.weight: {W.shape}")

    # --- 2. PCA to target dimensionality (sklearn PCA centers internally) ---
    print(f"Fitting PCA: {W.shape[1]}d -> {args.target_dim}d...")
    pca = PCA(n_components=args.target_dim, random_state=args.seed)
    W_reduced = pca.fit_transform(W)
    variance_retained = pca.explained_variance_ratio_.sum()
    print(f"Variance retained: {variance_retained * 100:.1f}%")

    # --- 3. Optionally normalize ---
    W_tensor = torch.tensor(W_reduced, dtype=torch.float32)
    if args.normalize:
        mean = W_tensor.mean(dim=0)
        std = W_tensor.std(dim=0).clamp(min=1e-6)
        W_final = (W_tensor - mean) / std
        norm_label = "per-dimension zero-mean unit-variance (uniform over vocab)"
        print(f"Applied normalization. Mean norm: {torch.norm(W_final, dim=-1).mean():.4f}")
    else:
        W_final = W_tensor
        norm_label = "none (raw PCA output)"
        print(f"Skipping normalization (raw PCA output). Mean norm: {torch.norm(W_final, dim=-1).mean():.4f}")
        print("  -> Normalize at training time using corpus stats or e2e-learned norms.")

    # --- 4. Save nn.Embedding as random_emb.torch ---
    vocab_size = W_final.shape[0]
    emb = torch.nn.Embedding(vocab_size, args.target_dim)
    emb.weight.data = W_final
    emb_path = os.path.join(args.output_dir, "random_emb.torch")
    torch.save(emb.state_dict(), emb_path)
    print(f"Saved embedding: {emb_path}  ({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    # --- 5. Save GPT-2 tokenizer ---
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved tokenizer to {args.output_dir}")

    # --- 6. Save metadata for reproducibility ---
    metadata = {
        "source_model": args.gpt2_model,
        "source_dim": int(W.shape[1]),
        "target_dim": args.target_dim,
        "vocab_size": vocab_size,
        "pca_variance_retained": float(variance_retained),
        "normalization": norm_label,
        "seed": args.seed,
    }
    meta_path = os.path.join(args.output_dir, "pca_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    # --- 7. Summary ---
    print(f"\n{'='*60}")
    print(f"Done! Output in: {args.output_dir}")
    print(f"  Embedding:  [{vocab_size}, {args.target_dim}]")
    print(f"  Variance:   {variance_retained * 100:.1f}%")
    print(f"  Files:")
    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        size = os.path.getsize(fpath)
        unit = "KB" if size < 1e6 else "MB"
        val = size / 1e3 if size < 1e6 else size / 1e6
        print(f"    {fname:30s}  {val:6.1f} {unit}")
    print(f"\nTransfer the entire '{args.output_dir}' directory to your remote host.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
