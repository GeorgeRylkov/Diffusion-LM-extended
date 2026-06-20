"""
Plot reconstruction loss comparison for wiki-trained BERT 128d models:
  - Frozen: bert-tiny embeddings
  - E2E: bert-uncased random embeddings

Run from repo root:
  python improved-diffusion/scripts/plot_recon_wiki_bert128.py
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPTS_DIR, "diffusion_models")

FROZEN_DIR = os.path.join(
    MODELS_DIR,
    "diff_wiki_pad_bert128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen_bsz256x3gpu_wu2000",
)
E2E_DIR = os.path.join(
    MODELS_DIR,
    "diff_wiki_pad_rand128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased_bsz256x3gpu_wu2000",
)

FROZEN_NPZ = os.path.join(FROZEN_DIR, "reconstruction_loss_ema_0.9999_166667.npz")
E2E_NPZ = os.path.join(E2E_DIR, "reconstruction_loss_ema_0.9999_150003.npz")

COMPARISON_OUT = os.path.join(MODELS_DIR, "recon_loss_comparison_wiki_bert128.png")
CHARTS_DIR = os.path.join(MODELS_DIR, "charts")
PAPER_OUT = os.path.join(CHARTS_DIR, "recon_loss_comparison_bert128d_wiki.png")


def load_npz(path):
    data = np.load(path)
    return data["t_indices"], data["mse_per_t"]


def plot_comparison(t_frozen, mse_frozen, t_e2e, mse_e2e, out_path, dpi=150):
    plt.figure(figsize=(8, 5))
    plt.plot(
        t_frozen,
        mse_frozen,
        linewidth=2,
        color="tab:blue",
        label="bert-tiny frozen embeddings",
    )
    plt.plot(
        t_e2e,
        mse_e2e,
        linewidth=2,
        color="tab:red",
        label="e2e trained embeddings",
    )
    plt.xlabel("Diffusion timestep t", fontsize=13)
    plt.ylabel("Reconstruction MSE", fontsize=13)
    plt.title("Reconstruction loss: BERT 128d Wikipedia", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for label, path in [("Frozen", FROZEN_NPZ), ("E2E", E2E_NPZ)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} NPZ not found: {path}")
            sys.exit(1)

    t_frozen, mse_frozen = load_npz(FROZEN_NPZ)
    t_e2e, mse_e2e = load_npz(E2E_NPZ)

    plot_comparison(t_frozen, mse_frozen, t_e2e, mse_e2e, COMPARISON_OUT, dpi=150)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plot_comparison(t_frozen, mse_frozen, t_e2e, mse_e2e, PAPER_OUT, dpi=300)

    print("\nSummary:")
    print(f"  Frozen — mean MSE: {mse_frozen.mean():.6f}, max: {mse_frozen.max():.6f}")
    print(f"  E2E    — mean MSE: {mse_e2e.mean():.6f}, max: {mse_e2e.max():.6f}")


if __name__ == "__main__":
    main()
