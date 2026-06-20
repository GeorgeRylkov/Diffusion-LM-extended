"""
Plot reconstruction loss graphs for E2E-tgt models (BERT-based):
  1. Individual per-timestep reconstruction loss for E2E-tgt frozen
  2. Individual per-timestep reconstruction loss for E2E-tgt e2e
  3. Comparison: E2E embeddings vs Frozen embeddings on E2E-tgt dataset

Run from the repo root after both eval jobs have completed:
  python improved-diffusion/scripts/plot_recon_e2e_tgt.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__))
MODELS_DIR = os.path.join(SCRIPTS_DIR, "diffusion_models")

FROZEN_DIR = os.path.join(
    MODELS_DIR,
    "diff_e2e-tgt_pad_bert128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen",
)
E2E_DIR = os.path.join(
    MODELS_DIR,
    "diff_e2e-tgt_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased",
)

FROZEN_NPZ = os.path.join(FROZEN_DIR, "reconstruction_loss_ema_0.9999_200000.npz")
E2E_NPZ = os.path.join(E2E_DIR, "reconstruction_loss_ema_0.9999_200000.npz")

CHARTS_DIR = os.path.join(MODELS_DIR, "charts")
PAPER_E2E_TGT_OUT = os.path.join(CHARTS_DIR, "recon_loss_comparison_bert128d_e2e_dataset.png")


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
    plt.plot(t_frozen, mse_frozen, linewidth=2, color="tab:blue", label="Frozen embeddings")
    plt.plot(t_e2e, mse_e2e, linewidth=2, color="tab:red", label="E2E embeddings")
    plt.xlabel("Diffusion timestep t", fontsize=13)
    plt.ylabel("Reconstruction MSE", fontsize=13)
    plt.title("Reconstruction loss: e2e vs frozen embeddings (E2E-tgt dataset)", fontsize=12)
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
    plt.title("Reconstruction loss: BERT 128d E2E dataset", fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for label, path in [("Frozen", FROZEN_NPZ), ("E2E", E2E_NPZ)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} npz not found: {path}")
            sys.exit(1)

    t_frozen, mse_frozen = load_npz(FROZEN_NPZ)
    t_e2e, mse_e2e = load_npz(E2E_NPZ)

    # Graph 1: individual plot for frozen
    plot_individual(
        t_frozen,
        mse_frozen,
        "diff_e2e-tgt_pad_bert128_..._bert_tiny_frozen",
        os.path.join(FROZEN_DIR, "reconstruction_loss_ema_0.9999_200000.png"),
    )

    # Graph 2: individual plot for e2e
    plot_individual(
        t_e2e,
        mse_e2e,
        "diff_e2e-tgt_pad_rand128_..._bert_uncased (E2E)",
        os.path.join(E2E_DIR, "reconstruction_loss_ema_0.9999_200000.png"),
    )

    # Graph 3: comparison plot
    plot_comparison(
        t_frozen,
        mse_frozen,
        t_e2e,
        mse_e2e,
        os.path.join(MODELS_DIR, "recon_loss_comparison_e2e_tgt.png"),
    )

    plot_comparison_paper(
        t_frozen,
        mse_frozen,
        t_e2e,
        mse_e2e,
        PAPER_E2E_TGT_OUT,
    )

    print("\nSummary:")
    print(f"  Frozen     — mean MSE: {mse_frozen.mean():.6f}, max: {mse_frozen.max():.6f}")
    print(f"  E2E        — mean MSE: {mse_e2e.mean():.6f}, max: {mse_e2e.max():.6f}")


if __name__ == "__main__":
    main()
