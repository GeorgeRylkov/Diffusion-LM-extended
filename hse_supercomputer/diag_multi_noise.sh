#!/bin/bash
#SBATCH --job-name=diag_multi_noise
#SBATCH --partition=normal
#SBATCH --output=logs/diag_multi_noise_%j.out
#SBATCH --error=logs/diag_multi_noise_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Diagnostic: Multiple noise draws per timestep"
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

DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_roc_pad_bert128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen
CHECKPOINT=${DIFFUSION_MODEL_DIR}/ema_0.9999_400000.pt

python -u scripts/diag_multi_noise.py \
    --model_path "${CHECKPOINT}" \
    --batch_size 64 \
    --num_batches 8 \
    --roc_train ../datasets/ROCstory

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
