"""
Plot reconstruction loss graphs for GPT2 models:
  1. Individual per-timestep reconstruction loss for GPT2 e2e
  2. Individual per-timestep reconstruction loss for GPT2-PCA frozen
  3. Comparison: GPT2 E2E vs GPT2-PCA Frozen

Run from the repo root after both eval jobs have completed:
  python improved-diffusion/scripts/plot_recon_gpt2.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__))
MODELS_DIR = os.path.join(SCRIPTS_DIR, "diffusion_models")

GPT2_E2E_DIR = os.path.join(
    MODELS_DIR,
    "diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_gpt2",
)
GPT2_PCA_DIR = os.path.join(
    MODELS_DIR,
    "diff_roc_pad_gpt2pca128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_gpt2_pca_frozen",
)

GPT2_E2E_NPZ = os.path.join(GPT2_E2E_DIR, "reconstruction_loss_ema_0.9999_400000.npz")
GPT2_PCA_NPZ = os.path.join(GPT2_PCA_DIR, "reconstruction_loss_ema_0.9999_400000.npz")

CHARTS_DIR = os.path.join(MODELS_DIR, "charts")
PAPER_GPT2_OUT = os.path.join(CHARTS_DIR, "recon_loss_comparison_gpt2_128d.png")


def load_npz(path):
    data = np.load(path)
    return data["t_indices"], data["mse_per_t"]


def plot_individual(t_indices, mse, model_name, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(t_indices, mse, linewidth=2)
    plt.xlabel("Diffusion timestep t", fontsize=13)
    plt.ylabel("Reconstruction MSE", fontsize=13)
    plt.title(f"Per-timestep Reconstruction Loss\n{model_name}", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison(t_frozen, mse_frozen, t_e2e, mse_e2e, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(t_frozen, mse_frozen, linewidth=2, color="tab:blue", label="GPT2-PCA frozen embeddings")
    plt.plot(t_e2e, mse_e2e, linewidth=2, color="tab:red", label="GPT2 E2E embeddings")
    plt.xlabel("Diffusion timestep t", fontsize=13)
    plt.ylabel("Reconstruction MSE", fontsize=13)
    plt.title("Reconstruction loss: GPT2 e2e vs GPT2-PCA frozen embeddings", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison_paper(t_frozen, mse_frozen, t_e2e, mse_e2e, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(
        t_frozen,
        mse_frozen,
        linewidth=2,
        color="tab:blue",
        label="GPT-2 frozen embeddings",
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
    plt.title("Reconstruction loss: GPT-2 128d ROCStories", fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for label, path in [("GPT2 E2E", GPT2_E2E_NPZ), ("GPT2-PCA frozen", GPT2_PCA_NPZ)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} npz not found: {path}")
            sys.exit(1)

    t_e2e, mse_e2e = load_npz(GPT2_E2E_NPZ)
    t_frozen, mse_frozen = load_npz(GPT2_PCA_NPZ)

    # Graph 1: individual plot for GPT2 e2e
    plot_individual(
        t_e2e,
        mse_e2e,
        "diff_roc_pad_rand128_..._sd101_gpt2 (E2E)",
        os.path.join(GPT2_E2E_DIR, "reconstruction_loss_ema_0.9999_400000.png"),
    )

    # Graph 2: individual plot for GPT2-PCA frozen
    plot_individual(
        t_frozen,
        mse_frozen,
        "diff_roc_pad_gpt2pca128_..._sd101_gpt2_pca_frozen",
        os.path.join(GPT2_PCA_DIR, "reconstruction_loss_ema_0.9999_400000.png"),
    )

    # Graph 3: comparison plot
    plot_comparison(
        t_frozen,
        mse_frozen,
        t_e2e,
        mse_e2e,
        os.path.join(MODELS_DIR, "recon_loss_comparison_gpt2.png"),
    )

    plot_comparison_paper(
        t_frozen,
        mse_frozen,
        t_e2e,
        mse_e2e,
        PAPER_GPT2_OUT,
    )

    print("\nSummary:")
    print(f"  GPT2 E2E       — mean MSE: {mse_e2e.mean():.6f}, max: {mse_e2e.max():.6f}")
    print(f"  GPT2-PCA frozen — mean MSE: {mse_frozen.mean():.6f}, max: {mse_frozen.max():.6f}")


if __name__ == "__main__":
    main()
