"""
Diagnostic #2: Multiple noise samples at t=1000.

Runs 20 independent noise draws at t=1000 with T=2000 schedule,
using the same x_0 each time. Reports MSE for each draw.
If the peak is real: all draws should give ~0.073.
If it's a noise artifact: draws should scatter around ~0.038-0.073.
"""

import argparse
import json
import os
import sys

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
from transformers import set_seed


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    dist_util.setup_dist()

    model_dir = os.path.split(args.model_path)[0]
    with open(os.path.join(model_dir, "training_args.json"), "rb") as f:
        training_args = json.load(f)

    cli_only = {}
    for k in ("batch_size", "num_batches", "model_path", "roc_train", "seed"):
        v = getattr(args, k, None)
        if v is not None:
            cli_only[k] = v

    args.__dict__.update(training_args)
    args.__dict__.update(cli_only)
    args.sigma_small = True
    args.checkpoint_path = model_dir
    if args.experiment == "random1":
        args.experiment = "random"

    print("Creating model (T=2000)...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    model2, tokenizer = load_models(
        args.modality, args.experiment, args.model_name_or_path,
        args.in_channel, model_dir, extra_args=args,
    )
    rev_tokenizer = tokenizer if args.use_bert_tokenizer == "yes" else None
    if args.training_mode.startswith("e2e"):
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    print("Loading validation data...")
    data = load_data_text(
        data_dir=args.data_dir, batch_size=args.batch_size,
        image_size=args.image_size, class_cond=args.class_cond,
        data_args=args, task_mode=args.modality,
        padding_mode=args.padding_mode, split="valid",
        load_vocab=rev_tokenizer, model=model2,
    )

    all_x0 = []
    for i in range(args.num_batches):
        batch, cond = next(data)
        all_x0.append(batch)
    all_x0 = th.cat(all_x0, dim=0).to(dist_util.dev())
    N = all_x0.shape[0]
    print(f"Collected {N} samples, shape {all_x0.shape}")

    T = diffusion.num_timesteps
    NUM_DRAWS = 20

    mid = T // 2
    test_timesteps = sorted(set([
        T // 4, mid - 50, mid - 25, mid - 10, mid - 5,
        mid,
        mid + 5, mid + 10, mid + 25, mid + 50, mid + 75,
        3 * T // 4,
    ]))
    test_timesteps = [t for t in test_timesteps if 0 <= t < T]

    print(f"\n{'='*70}")
    print(f"MULTI-NOISE TEST: {NUM_DRAWS} independent noise draws per timestep")
    print(f"{'='*70}\n")

    with th.no_grad():
        for t_val in test_timesteps:
            mse_values = []
            t_tensor = th.full((N,), t_val, device=dist_util.dev(), dtype=th.long)
            scaled_t = diffusion._scale_timesteps(t_tensor)

            for draw in range(NUM_DRAWS):
                noise = th.randn_like(all_x0)
                x_t = diffusion.q_sample(all_x0, t_tensor, noise=noise)
                pred = model(x_t, scaled_t)
                mse = ((all_x0 - pred) ** 2).mean().item()
                mse_values.append(mse)

            arr = np.array(mse_values)
            print(f"t={t_val:5d} (scaled={t_val*1000/T:.1f}):  "
                  f"mean={arr.mean():.6f}  std={arr.std():.6f}  "
                  f"min={arr.min():.6f}  max={arr.max():.6f}  "
                  f"cv={arr.std()/arr.mean()*100:.1f}%")

    print(f"\n{'='*70}")
    print("If the peak is real, t=1000 mean should be >> t=1050/t=1100 mean.")
    print("If it's noise variance, t=1000 mean should be similar to neighbors.")
    print(f"{'='*70}")


def create_argparser():
    defaults = dict(
        model_path="",
        batch_size=64,
        num_batches=8,
        seed=101,
        roc_train=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
