"""
Diagnostic: coefficient swap test.

Uses the T=2000 model and validation data. At t=1000, constructs x_t two ways:
  A) With T=2000's coefficients (normal)
  B) With T=1000's equivalent coefficients (from t=502)
Same x_0, same noise, same scaled_t=500.0. Only the mixing coefficients differ by ~0.3%.

If MSE differs dramatically, the model is sensitive to exact coefficients.
If MSE is similar, the T=1000 eval's lower MSE must be caused by something else (e.g. different x_0).
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


def betas_for_alpha_bar(T, alpha_bar_fn, max_beta=0.999):
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return np.array(betas)


def build_sqrt_schedule(T):
    betas = betas_for_alpha_bar(T, lambda t: 1 - np.sqrt(t + 0.0001))
    alphas = 1.0 - betas
    ac = np.cumprod(alphas)
    return ac


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    dist_util.setup_dist()

    model_dir = os.path.split(args.model_path)[0]
    config_path = os.path.join(model_dir, "training_args.json")
    with open(config_path, "rb") as f:
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

    print("Creating model (T=2000 diffusion)...")
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

    # Build both schedules
    ac_2000 = build_sqrt_schedule(2000)
    ac_1000 = build_sqrt_schedule(1000)

    T = diffusion.num_timesteps
    assert T == 2000, f"Expected T=2000 from training, got T={T}"

    # Test points: T=2000 t=1000 vs T=1000 t=502 (closest ac match)
    t_2000 = 1000
    t_1000 = 502  # ac_1000[502] ≈ 0.2936 vs ac_2000[1000] ≈ 0.2954

    sqrt_ac_A = float(np.sqrt(ac_2000[t_2000]))
    sqrt_1mac_A = float(np.sqrt(1 - ac_2000[t_2000]))
    sqrt_ac_B = float(np.sqrt(ac_1000[t_1000]))
    sqrt_1mac_B = float(np.sqrt(1 - ac_1000[t_1000]))

    # Also use T=2000's _scale_timesteps for both
    scaled_t_A = t_2000 * (1000.0 / 2000)  # = 500.0
    scaled_t_B = t_1000 * (1000.0 / 1000)  # = 502.0

    print(f"\n{'='*70}")
    print(f"COEFFICIENT SWAP TEST")
    print(f"{'='*70}")
    print(f"Pass A (T=2000 native):  t={t_2000}, ac={ac_2000[t_2000]:.8f}")
    print(f"  sqrt_ac={sqrt_ac_A:.8f}, sqrt_1mac={sqrt_1mac_A:.8f}, scaled_t={scaled_t_A:.1f}")
    print(f"Pass B (T=1000 coeffs):  t_equiv={t_1000}, ac={ac_1000[t_1000]:.8f}")
    print(f"  sqrt_ac={sqrt_ac_B:.8f}, sqrt_1mac={sqrt_1mac_B:.8f}, scaled_t={scaled_t_B:.1f}")
    print(f"Coefficient difference:  sqrt_ac {abs(sqrt_ac_A-sqrt_ac_B)/sqrt_ac_A*100:.3f}%, "
          f"sqrt_1mac {abs(sqrt_1mac_A-sqrt_1mac_B)/sqrt_1mac_A*100:.3f}%")
    print(f"{'='*70}\n")

    # Generate ONE noise tensor, shared across all passes
    noise = th.randn_like(all_x0)

    with th.no_grad():
        # === Pass A: T=2000 coefficients, scaled_t=500.0 ===
        x_t_A = sqrt_ac_A * all_x0 + sqrt_1mac_A * noise
        scaled_t_tensor_A = th.full((N,), scaled_t_A, device=dist_util.dev(), dtype=th.float)
        pred_A = model(x_t_A, scaled_t_tensor_A)
        mse_A = ((all_x0 - pred_A) ** 2).mean().item()

        # === Pass B: T=1000 coefficients, scaled_t=502.0 ===
        x_t_B = sqrt_ac_B * all_x0 + sqrt_1mac_B * noise
        scaled_t_tensor_B = th.full((N,), scaled_t_B, device=dist_util.dev(), dtype=th.float)
        pred_B = model(x_t_B, scaled_t_tensor_B)
        mse_B = ((all_x0 - pred_B) ** 2).mean().item()

        # === Pass C: T=1000 coefficients but scaled_t=500.0 (isolate coeff effect) ===
        pred_C = model(x_t_B, scaled_t_tensor_A)
        mse_C = ((all_x0 - pred_C) ** 2).mean().item()

        # === Pass D: T=2000 coefficients but scaled_t=502.0 (isolate scaled_t effect) ===
        pred_D = model(x_t_A, scaled_t_tensor_B)
        mse_D = ((all_x0 - pred_D) ** 2).mean().item()

        # === Passes at neighboring timesteps for context ===
        context_results = []
        for t_raw in [950, 980, 990, 999, 1000, 1001, 1010, 1020, 1050, 1100, 1150]:
            t_tensor = th.full((N,), t_raw, device=dist_util.dev(), dtype=th.long)
            noise_ctx = th.randn_like(all_x0)
            x_t_ctx = diffusion.q_sample(all_x0, t_tensor, noise=noise_ctx)
            pred_ctx = model(x_t_ctx, diffusion._scale_timesteps(t_tensor))
            mse_ctx = ((all_x0 - pred_ctx) ** 2).mean().item()
            context_results.append((t_raw, mse_ctx))

    print(f"{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Pass A: T=2000 coeffs, scaled_t=500.0  → MSE = {mse_A:.6f}")
    print(f"Pass B: T=1000 coeffs, scaled_t=502.0  → MSE = {mse_B:.6f}")
    print(f"Pass C: T=1000 coeffs, scaled_t=500.0  → MSE = {mse_C:.6f}  (isolate coeff effect)")
    print(f"Pass D: T=2000 coeffs, scaled_t=502.0  → MSE = {mse_D:.6f}  (isolate scaled_t effect)")
    print()
    print(f"A vs B (both differ): {mse_A:.6f} vs {mse_B:.6f} (ratio {mse_A/mse_B:.3f})")
    print(f"A vs C (only coeffs differ): {mse_A:.6f} vs {mse_C:.6f} (ratio {mse_A/mse_C:.3f})")
    print(f"A vs D (only scaled_t differs): {mse_A:.6f} vs {mse_D:.6f} (ratio {mse_A/mse_D:.3f})")
    print()
    print(f"Context: normal T=2000 eval at nearby timesteps (independent noise):")
    for t_raw, mse_ctx in context_results:
        marker = " <<<" if t_raw == 1000 else ""
        print(f"  t={t_raw:5d}  MSE={mse_ctx:.6f}{marker}")

    # Prediction statistics
    print(f"\n{'='*70}")
    print(f"PREDICTION DIAGNOSTICS")
    print(f"{'='*70}")
    diff_pred = ((pred_A - pred_B) ** 2).mean().item()
    print(f"MSE between pred_A and pred_B: {diff_pred:.8f}")
    print(f"x_t difference: {((x_t_A - x_t_B) ** 2).mean().item():.8f}")
    print(f"pred_A stats: mean={pred_A.mean().item():.6f}, std={pred_A.std().item():.6f}")
    print(f"pred_B stats: mean={pred_B.mean().item():.6f}, std={pred_B.std().item():.6f}")
    print(f"x_0 stats:    mean={all_x0.mean().item():.6f}, std={all_x0.std().item():.6f}")


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
