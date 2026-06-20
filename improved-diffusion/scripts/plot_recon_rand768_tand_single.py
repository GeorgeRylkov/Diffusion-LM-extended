"""
Remake the single-model reconstruction plot for rand768 BERT e2e with tan-d noise (ROCStories).

Reads NPZ written by eval_reconstruction.py and writes a labeled figure matching paper style.

Run from repo root:
  python improved-diffusion/scripts/plot_recon_rand768_tand_single.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPTS_DIR, "diffusion_models")

MODEL_DIR = os.path.join(
    MODELS_DIR,
    "diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_tand_Lsimple_h128_s2_d0.1_sd101_bert_v2",
)
NPZ_PATH = os.path.join(MODEL_DIR, "reconstruction_loss_ema_0.9999_400000.npz")
OUT_MODEL_DIR = os.path.join(MODEL_DIR, "reconstruction_loss_ema_0.9999_400000.png")
CHARTS_DIR = os.path.join(MODELS_DIR, "charts")
OUT_CHART = os.path.join(CHARTS_DIR, "recon_loss_rand768_tand_e2e_rocstories.png")


def main():
    data = np.load(NPZ_PATH)
    t = data["t_indices"]
    mse = data["mse_per_t"]

    # Match comparison-chart styling: red = e2e trained
    plt.figure(figsize=(8, 5))
    plt.plot(t, mse, linewidth=2, color="tab:red", label="e2e trained embeddings")
    plt.xlabel("Diffusion timestep t", fontsize=13)
    plt.ylabel("Reconstruction MSE", fontsize=13)
    plt.title(
        "Reconstruction loss: BERT 768d ROCStories\n"
        "tan-d noise schedule",
        fontsize=13,
    )
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.savefig(OUT_MODEL_DIR, dpi=150)
    plt.savefig(OUT_CHART, dpi=300)
    plt.close()

    print(f"Saved: {OUT_MODEL_DIR}")
    print(f"Saved: {OUT_CHART}")


if __name__ == "__main__":
    main()
