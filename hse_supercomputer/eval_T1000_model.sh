#!/bin/bash
#SBATCH --job-name=eval_T1000_model
#SBATCH --partition=normal
#SBATCH --output=logs/eval_T1000_model_%j.out
#SBATCH --error=logs/eval_T1000_model_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Eval T=1000 trained model (50k checkpoint)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_hse/bin/activate

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd improved-diffusion

MODEL_DIR=scripts/diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_1000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased
CKPT=${MODEL_DIR}/ema_0.9999_050000.pt

echo ""
echo "=== Part 1: Full 1000-point reconstruction eval ==="
python -u scripts/eval_reconstruction.py \
    --model_path "${CKPT}" \
    --batch_size 64 \
    --num_batches 8 \
    --num_timesteps 1000 \
    --roc_train ../datasets/ROCstory

echo ""
echo "=== Part 2: Multi-noise diagnostic around t=500 (T/2) ==="
python -u scripts/diag_multi_noise.py \
    --model_path "${CKPT}" \
    --batch_size 64 \
    --num_batches 8 \
    --roc_train ../datasets/ROCstory

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
