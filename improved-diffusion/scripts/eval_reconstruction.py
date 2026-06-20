"""
Evaluate per-timestep reconstruction loss for a trained diffusion model,
reproducing the style of Figure 4 in the TENCDM paper.

For each timestep t in [0, T) we:
  1. Take clean embeddings x_0 from the validation set
  2. Noise them to x_t via q(x_t | x_0)
  3. Run the model to predict x̂_0
  4. Compute MSE(x_0, x̂_0) averaged over samples and dimensions
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from improved_diffusion import dist_util
from improved_diffusion.rounding import load_models
from improved_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.gaussian_diffusion import _extract_into_tensor
from transformers import set_seed, AutoTokenizer


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)

    dist_util.setup_dist()

    model_dir = os.path.split(args.model_path)[0]
    config_path = os.path.join(model_dir, "training_args.json")
    print(f"Loading config from {config_path}")
    with open(config_path, "rb") as f:
        training_args = json.load(f)

    cli_only = {}
    for k in (
        "batch_size",
        "num_batches",
        "num_timesteps",
        "model_path",
        "roc_train",
        "e2e_train",
        "wiki_corpus_train",
        "wiki_tokenized_dir",
        "tokenizer_name_or_path",
        "seed",
        "t_start",
        "t_end",
        "override_diffusion_steps",
    ):
        v = getattr(args, k, None)
        if v is not None:
            cli_only[k] = v

    args.__dict__.update(training_args)
    args.__dict__.update(cli_only)
    args.sigma_small = True
    args.checkpoint_path = model_dir

    if args.experiment == "random1":
        args.experiment = "random"

    if args.override_diffusion_steps is not None:
        orig_steps = args.diffusion_steps
        args.diffusion_steps = int(args.override_diffusion_steps)
        print(f"*** Overriding diffusion_steps: {orig_steps} -> {args.diffusion_steps} ***")

    print(f"\n{'='*50}")
    print("FINAL ARGS (after all overrides):")
    print(f"{'='*50}")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print(f"{'='*50}\n")

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {pytorch_total_params}")

    # Load tokenizer + embedding model from the checkpoint directory (offline-safe)
    model2, tokenizer = load_models(
        args.modality, args.experiment, args.model_name_or_path,
        args.in_channel, model_dir, extra_args=args,
    )

    if args.use_bert_tokenizer == "yes":
        rev_tokenizer = tokenizer
    else:
        rev_tokenizer = None

    # For e2e models, update model2 weights from the checkpoint
    if args.training_mode.startswith("e2e"):
        print("e2e mode: loading word_embedding weights from checkpoint")
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    print("Loading validation data...")
    data = load_data_text(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        data_args=args,
        task_mode=args.modality,
        padding_mode=args.padding_mode,
        split="valid",
        load_vocab=rev_tokenizer,
        model=model2,
    )

    print(f"Collecting {args.num_batches} validation batches...")
    all_x0 = []
    for i in range(args.num_batches):
        batch, cond = next(data)
        all_x0.append(batch)
    all_x0 = th.cat(all_x0, dim=0).to(dist_util.dev())
    N = all_x0.shape[0]
    print(f"Collected {N} samples, shape {all_x0.shape}")

    T = diffusion.num_timesteps
    t_start = int(args.t_start) if args.t_start is not None else None
    t_end = int(args.t_end) if args.t_end is not None else None
    if t_start is not None and t_end is not None:
        t_indices = np.arange(t_start, min(t_end + 1, T))
        print(f"Custom range: evaluating every timestep in [{t_start}, {t_end}]")
    else:
        num_t = min(args.num_timesteps, T)
        t_indices = np.linspace(0, T - 1, num_t, dtype=int)
    t_indices = np.unique(t_indices)

    print(f"Evaluating reconstruction at {len(t_indices)} timesteps (T={T})...")

    if args.no_rescale:
        original_rescale = diffusion.rescale_timesteps
        diffusion.rescale_timesteps = False
        print(f"*** no_rescale: disabled rescale_timesteps (was {original_rescale}) ***")
        print(f"    Model will receive raw timesteps [0, {T-1}] instead of scaled [0, {1000*(T-1)/T:.1f}]")

    mse_per_t = []
    wrapped_model = diffusion._wrap_model(model)

    with th.no_grad():
        for idx, t_val in enumerate(t_indices):
            t_tensor = th.full((N,), t_val, device=dist_util.dev(), dtype=th.long)
            noise = th.randn_like(all_x0)
            x_t = diffusion.q_sample(all_x0, t_tensor, noise=noise)

            model_output = wrapped_model(x_t, t_tensor)

            if diffusion.model_mean_type.name == "START_X":
                pred_x0 = model_output
            else:
                pred_x0 = (
                    _extract_into_tensor(diffusion.sqrt_recip_alphas_cumprod, t_tensor, x_t.shape) * x_t
                    - _extract_into_tensor(diffusion.sqrt_recipm1_alphas_cumprod, t_tensor, x_t.shape) * model_output
                )

            mse = ((all_x0 - pred_x0) ** 2).mean().item()
            mse_per_t.append(mse)

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  t={t_val:5d} ({t_val/T:.3f}): MSE = {mse:.6f}")

    t_normalized = t_indices / T
    mse_array = np.array(mse_per_t)

    # Save all results
    checkpoint_name = os.path.splitext(os.path.basename(args.model_path))[0]
    suffix = f"_seed{args.seed}" if args.seed != 101 else ""
    if getattr(args, 't_start', None) is not None:
        suffix += f"_t{args.t_start}-{args.t_end}"
    if args.no_rescale:
        suffix += "_norescale"
    if args.override_diffusion_steps is not None:
        suffix += f"_T{args.diffusion_steps}"
    npz_path = os.path.join(model_dir, f"reconstruction_loss_{checkpoint_name}{suffix}.npz")
    np.savez(
        npz_path,
        t_indices=t_indices,
        t_normalized=t_normalized,
        mse_per_t=mse_array,
        num_samples=N,
        num_timesteps_total=T,
        seed=args.seed,
        model_path=args.model_path,
        training_mode=args.training_mode,
        experiment=args.experiment,
        noise_schedule=args.noise_schedule,
        in_channel=args.in_channel,
    )

    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"  Model: {model_dir}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Num validation samples: {N}")
    print(f"  Num timesteps evaluated: {len(t_indices)}")
    print(f"  Overall mean MSE: {mse_array.mean():.6f}")
    print(f"  MSE at t=0:       {mse_array[0]:.6f}")
    print(f"  MSE at t=T/4:     {mse_array[len(mse_array)//4]:.6f}")
    print(f"  MSE at t=T/2:     {mse_array[len(mse_array)//2]:.6f}")
    print(f"  MSE at t=3T/4:    {mse_array[3*len(mse_array)//4]:.6f}")
    print(f"  MSE at t=T-1:     {mse_array[-1]:.6f}")
    print(f"  MSE ratio (max/min): {mse_array.max()/max(mse_array.min(), 1e-10):.1f}x")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(t_indices, mse_per_t, linewidth=2)
    plt.xlabel("Diffusion timestep t", fontsize=13)
    plt.ylabel("Reconstruction MSE", fontsize=13)
    plt.title(f"Per-timestep Reconstruction Loss\n{os.path.basename(model_dir)}", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(model_dir, f"reconstruction_loss_{checkpoint_name}{suffix}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved figure to {fig_path}")
    print(f"Saved data to {npz_path}")


def create_argparser():
    defaults = dict(
        model_path="",
        batch_size=64,
        num_batches=8,
        num_timesteps=200,
        seed=101,
        t_start=None,
        t_end=None,
        no_rescale=False,
        override_diffusion_steps=None,
        roc_train=None,
        e2e_train=None,
        wiki_corpus_train=None,
        wiki_tokenized_dir=None,
        tokenizer_name_or_path=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
