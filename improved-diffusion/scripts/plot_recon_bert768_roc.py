"""
BERT 768d ROCStories: frozen vs e2e reconstruction comparison.

Requires:
  - diff_roc_pad_bert768_..._sd101_v2/reconstruction_loss_ema_0.9999_400000.npz
  - diff_roc_pad_rand768_..._sd101_bert_v2/reconstruction_loss_ema_0.9999_400000.npz

Run from repo root:
  python improved-diffusion/scripts/plot_recon_bert768_roc.py
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
    "diff_roc_pad_bert768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_v2",
)
E2E_DIR = os.path.join(
    MODELS_DIR,
    "diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_v2",
)

FROZEN_NPZ = os.path.join(FROZEN_DIR, "reconstruction_loss_ema_0.9999_400000.npz")
E2E_NPZ = os.path.join(E2E_DIR, "reconstruction_loss_ema_0.9999_400000.npz")

CHARTS_DIR = os.path.join(MODELS_DIR, "charts")
PAPER_OUT = os.path.join(CHARTS_DIR, "recon_loss_comparison_bert768d.png")
COMPARISON_OUT = os.path.join(MODELS_DIR, "recon_loss_comparison_bert768.png")


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


def plot_comparison(t_frozen, mse_frozen, t_e2e, mse_e2e, out_path, dpi=150):
    plt.figure(figsize=(8, 5))
    plt.plot(
        t_frozen,
        mse_frozen,
        linewidth=2,
        color="tab:blue",
        label="BERT frozen embeddings",
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
    plt.title("Reconstruction loss: BERT 768d frozen vs e2e (ROCStories)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved: {out_path}")


def plot_paper(t_frozen, mse_frozen, t_e2e, mse_e2e, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(
        t_frozen,
        mse_frozen,
        linewidth=2,
        color="tab:blue",
        label="BERT frozen embeddings",
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
    plt.title("Reconstruction loss: BERT 768d ROCStories", fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    for label, path in [("BERT 768 frozen", FROZEN_NPZ), ("BERT 768 e2e", E2E_NPZ)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} npz not found: {path}")
            sys.exit(1)

    t_frozen, mse_frozen = load_npz(FROZEN_NPZ)
    t_e2e, mse_e2e = load_npz(E2E_NPZ)

    plot_individual(
        t_frozen,
        mse_frozen,
        "diff_roc_pad_bert768_..._sd101_v2 (frozen)",
        os.path.join(FROZEN_DIR, "reconstruction_loss_ema_0.9999_400000.png"),
    )
    plot_individual(
        t_e2e,
        mse_e2e,
        "diff_roc_pad_rand768_..._sd101_bert_v2 (e2e)",
        os.path.join(E2E_DIR, "reconstruction_loss_ema_0.9999_400000.png"),
    )

    plot_comparison(t_frozen, mse_frozen, t_e2e, mse_e2e, COMPARISON_OUT, dpi=150)
    plot_paper(t_frozen, mse_frozen, t_e2e, mse_e2e, PAPER_OUT)

    print("\nSummary:")
    print(f"  Frozen — mean MSE: {mse_frozen.mean():.6f}, max: {mse_frozen.max():.6f}")
    print(f"  E2E    — mean MSE: {mse_e2e.mean():.6f}, max: {mse_e2e.max():.6f}")


if __name__ == "__main__":
    main()
