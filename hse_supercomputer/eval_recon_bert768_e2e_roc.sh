#!/bin/bash
#SBATCH --job-name=eval_recon_b768_e2e
#SBATCH --partition=normal
#SBATCH --output=logs/eval_recon_bert768_e2e_rand_%j.out
#SBATCH --error=logs/eval_recon_bert768_e2e_rand_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Per-timestep reconstruction loss — BERT 768d E2E (ROCStories)"
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

MODEL_DIR=scripts/diffusion_models/diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_v2
CHECKPOINT=${MODEL_DIR}/ema_0.9999_400000.pt

python -u scripts/eval_reconstruction.py \
    --model_path "${CHECKPOINT}" \
    --batch_size 32 \
    --num_batches 8 \
    --num_timesteps 2000 \
    --roc_train ../datasets/ROCstory

echo "Completed at: $(date)"
