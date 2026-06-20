"""
Paper-style single-line reconstruction plots for rand768 BERT e2e (ROCStories),
tan-2 and tan-3 noise schedules (same layout as plot_recon_rand768_tand_single.py).

Run from repo root:
  python improved-diffusion/scripts/plot_recon_rand768_tan2_tan3.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPTS_DIR, "diffusion_models")
CHARTS_DIR = os.path.join(MODELS_DIR, "charts")

CHECKPOINT_STEM = "reconstruction_loss_ema_0.9999_400000"

RUNS = [
    (
        "diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_tan_2_Lsimple_h128_s2_d0.1_sd101_bert",
        "tan-2 noise schedule",
        "recon_loss_rand768_tan_2_e2e_rocstories.png",
    ),
    (
        "diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_tan_3_Lsimple_h128_s2_d0.1_sd101_bert",
        "tan-3 noise schedule",
        "recon_loss_rand768_tan_3_e2e_rocstories.png",
    ),
]


def plot_one(model_subdir: str, schedule_subtitle: str, chart_basename: str) -> None:
    model_dir = os.path.join(MODELS_DIR, model_subdir)
    npz_path = os.path.join(model_dir, f"{CHECKPOINT_STEM}.npz")
    out_model_png = os.path.join(model_dir, f"{CHECKPOINT_STEM}.png")
    out_chart = os.path.join(CHARTS_DIR, chart_basename)

    data = np.load(npz_path)
    t = data["t_indices"]
    mse = data["mse_per_t"]

    plt.figure(figsize=(8, 5))
    plt.plot(t, mse, linewidth=2, color="tab:red", label="e2e trained embeddings")
    plt.xlabel("Diffusion timestep t", fontsize=13)
    plt.ylabel("Reconstruction MSE", fontsize=13)
    plt.title(
        f"Reconstruction loss: BERT 768d ROCStories\n{schedule_subtitle}",
        fontsize=13,
    )
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.savefig(out_model_png, dpi=150)
    plt.savefig(out_chart, dpi=300)
    plt.close()

    print(f"Saved: {out_model_png}")
    print(f"Saved: {out_chart}")


def main():
    for subdir, subtitle, chart_name in RUNS:
        plot_one(subdir, subtitle, chart_name)


if __name__ == "__main__":
    main()
