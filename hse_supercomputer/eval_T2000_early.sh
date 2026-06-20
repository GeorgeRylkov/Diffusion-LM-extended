#!/bin/bash
#SBATCH --job-name=eval_T2000_early
#SBATCH --partition=normal
#SBATCH --output=logs/eval_T2000_early_%j.out
#SBATCH --error=logs/eval_T2000_early_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Eval T=2000 e2e model (50k checkpoint) — for comparison with T=1000"
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

MODEL_DIR=scripts/diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased
CKPT=${MODEL_DIR}/ema_0.9999_050000.pt

echo ""
echo "=== Full 2000-point reconstruction eval ==="
python -u scripts/eval_reconstruction.py \
    --model_path "${CKPT}" \
    --batch_size 64 \
    --num_batches 8 \
    --num_timesteps 2000 \
    --roc_train ../datasets/ROCstory

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
