"""
Plot per-quartile reconstruction loss from training progress.csv files.

Produces a bar chart (or line plot over training) of the final MSE per
quartile, approximating the TENCDM-style per-timestep reconstruction plot.

Each quartile corresponds to a range of diffusion timesteps:
  q0: t in [0, T/4)     — low noise
  q1: t in [T/4, T/2)   — moderate noise
  q2: t in [T/2, 3T/4)  — high noise
  q3: t in [3T/4, T)    — very high noise
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_progress(csv_path):
    """Read progress.csv, skipping rows where 'step' is empty (eval-only rows)."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("step", "").strip():
                rows.append(row)
    return rows


def extract_final_quartile_mse(rows, window=100):
    """Extract the average MSE per quartile over the last `window` training steps."""
    recent = rows[-window:]
    quartiles = {}
    for q in range(4):
        key = f"mse_q{q}"
        vals = [float(r[key]) for r in recent if r.get(key, "").strip()]
        quartiles[q] = np.mean(vals) if vals else 0.0
    overall = [float(r["mse"]) for r in recent if r.get("mse", "").strip()]
    return quartiles, np.mean(overall) if overall else 0.0


def extract_training_curve(rows, col, smoothing=50):
    """Extract a smoothed training curve for a given column."""
    steps, vals = [], []
    for r in rows:
        if r.get("step", "").strip() and r.get(col, "").strip():
            steps.append(int(float(r["step"])))
            vals.append(float(r[col]))
    steps, vals = np.array(steps), np.array(vals)
    if smoothing > 1 and len(vals) > smoothing:
        kernel = np.ones(smoothing) / smoothing
        vals = np.convolve(vals, kernel, mode="valid")
        steps = steps[smoothing - 1:]
    return steps, vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", help="progress.csv files to compare")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each file")
    parser.add_argument("--window", type=int, default=200, help="Averaging window for final values")
    parser.add_argument("--smoothing", type=int, default=100, help="Smoothing for training curves")
    parser.add_argument("--output", type=str, default="quartile_reconstruction.png")
    parser.add_argument("--training_curves", action="store_true", help="Also plot training curves")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = []
        for f in args.csv_files:
            parts = os.path.dirname(f).split("/")
            name = parts[-1] if parts[-1] else os.path.basename(f)
            if "bert768" in name:
                args.labels.append("BERT frozen")
            elif "rand768" in name and "bert" in name:
                args.labels.append("BERT random e2e")
            elif "rand" in name:
                args.labels.append("Random")
            else:
                args.labels.append(name[:40])

    n_models = len(args.csv_files)
    quartile_labels = ["q0\n[0, T/4)", "q1\n[T/4, T/2)", "q2\n[T/2, 3T/4)", "q3\n[3T/4, T)"]
    quartile_midpoints = [0.125, 0.375, 0.625, 0.875]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    all_data = []
    for csv_path in args.csv_files:
        rows = read_progress(csv_path)
        quartiles, overall = extract_final_quartile_mse(rows, window=args.window)
        all_data.append((rows, quartiles, overall))
        print(f"{csv_path}:")
        print(f"  Steps in log: {len(rows)}, final step: {rows[-1].get('step', '?')}")
        print(f"  Overall MSE (last {args.window}): {overall:.6f}")
        for q in range(4):
            print(f"  mse_q{q}: {quartiles[q]:.6f}")

    if args.training_curves:
        n_plots = 2
    else:
        n_plots = 1

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    bar_width = 0.8 / n_models
    x = np.array(quartile_midpoints)
    for i, (rows, quartiles, overall) in enumerate(all_data):
        offsets = np.arange(4) * (0.8 / 4) + i * bar_width - (n_models - 1) * bar_width / 2
        q_vals = [quartiles[q] for q in range(4)]
        bars = ax.bar(
            np.arange(4) + i * bar_width - (n_models - 1) * bar_width / 2,
            q_vals, bar_width, label=args.labels[i], color=colors[i % len(colors)], alpha=0.85,
        )
        for bar, val in zip(bars, q_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(4))
    ax.set_xticklabels(quartile_labels, fontsize=10)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Reconstruction Loss by Timestep Quartile\n(averaged over last training steps)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    if args.training_curves:
        ax2 = axes[1]
        for i, (rows, quartiles, overall) in enumerate(all_data):
            for q in range(4):
                steps, vals = extract_training_curve(rows, f"mse_q{q}", smoothing=args.smoothing)
                linestyle = ["-", "--", "-.", ":"][q]
                ax2.plot(steps, vals, linestyle=linestyle, color=colors[i % len(colors)],
                         alpha=0.7, label=f"{args.labels[i]} q{q}" if i == 0 or q == 0 else None)
        ax2.set_xlabel("Step", fontsize=12)
        ax2.set_ylabel("MSE", fontsize=12)
        ax2.set_title("Quartile MSE Over Training", fontsize=12)
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(alpha=0.3)
        ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
