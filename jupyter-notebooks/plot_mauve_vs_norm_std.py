"""
Plot MAUVE and PPL as functions of embedding row-norm std, one dot per setup
(BERT-tiny / GPT-2 x frozen / e2e) on the filtered ROCStories vocabulary.

Row-norm std is taken from the PIP-comparison runs we already performed on the
filtered vocabularies; MAUVE and PPL come from the user's results table.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (name, backbone, frozen/e2e, row_norm_std, ppl, mauve, color, marker)
points = [
    ("e2e BERT-tiny",       "BERT-tiny", "e2e",    0.168, 3.26, 0.198, "#1f77b4", "o"),
    ("Frozen BERT-tiny",    "BERT-tiny", "frozen", 0.459, 3.04, 0.503, "#1f77b4", "s"),
    ("e2e GPT-2",           "GPT-2",     "e2e",    0.127, 3.29, 0.201, "#d62728", "o"),
    ("Frozen GPT-2 (PCA)",  "GPT-2",     "frozen", 1.406, 4.11, 0.111, "#d62728", "s"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

for ax, (metric_name, metric_idx, better) in zip(
    axes,
    [("MAUVE (higher is better)", 5, "higher"),
     ("PPL  (lower is better)",   4, "lower")],
):
    xs = np.array([p[3] for p in points])
    ys = np.array([p[metric_idx] for p in points])

    for p in points:
        ax.scatter(p[3], p[metric_idx], s=180, c=p[6], marker=p[7],
                   edgecolors="black", linewidths=1.2, zorder=3)

    for p in points:
        dx, dy = 0.04, 0.0
        if p[0] == "Frozen BERT-tiny":
            dy = 0.015 if metric_idx == 5 else -0.05
        if p[0] == "e2e BERT-tiny":
            dy = -0.03 if metric_idx == 5 else 0.05
        if p[0] == "e2e GPT-2":
            dy = 0.018 if metric_idx == 5 else -0.05
        if p[0] == "Frozen GPT-2 (PCA)":
            dx = -0.09
            dy = 0.02 if metric_idx == 5 else 0.06
            ha = "right"
        else:
            ha = "left"
        ax.annotate(p[0], xy=(p[3], p[metric_idx]),
                    xytext=(p[3] + dx, p[metric_idx] + dy),
                    fontsize=10, ha=ha,
                    color=p[6])

    # Light guides
    ax.axvspan(0.25, 0.65, color="#2ca02c", alpha=0.08, zorder=0)
    ax.text(0.45, ax.get_ylim()[1] if False else 0,
            "diffusion-friendly\nrow-norm band",
            color="#2ca02c", fontsize=9, ha="center", va="bottom",
            transform=ax.get_xaxis_transform())

    ax.set_xlabel("Embedding row-norm std  (filtered ROCStories vocab)")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name)
    ax.set_xlim(0.0, 1.55)
    ax.grid(True, alpha=0.3)

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markeredgecolor="black", markersize=12, label="e2e learned"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
           markeredgecolor="black", markersize=12, label="Frozen"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
           markeredgecolor="black", markersize=12, label="BERT-tiny (WordPiece)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
           markeredgecolor="black", markersize=12, label="GPT-2 (BPE)"),
]
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=10)

fig.suptitle(
    "Generation quality vs. embedding row-norm anisotropy  (Diffusion-LM on ROCStories, 128d)",
    y=1.06, fontsize=12, fontweight="bold",
)

out = Path(__file__).resolve().parents[1] / "charts" / "mauve_ppl_vs_row_norm_std.png"
fig.tight_layout()
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"saved -> {out}")
