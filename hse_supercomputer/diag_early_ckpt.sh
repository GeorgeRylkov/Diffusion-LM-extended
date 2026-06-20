#!/bin/bash
#SBATCH --job-name=diag_early_ckpt
#SBATCH --partition=normal
#SBATCH --output=logs/diag_early_ckpt_%j.out
#SBATCH --error=logs/diag_early_ckpt_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Diagnostic: Peak at early checkpoints (e2e 128d)"
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

for STEP in 050000 100000 200000; do
    CKPT=${MODEL_DIR}/ema_0.9999_${STEP}.pt
    echo ""
    echo "========================================"
    echo "Checkpoint: ema_0.9999_${STEP}.pt"
    echo "========================================"
    python -u scripts/diag_multi_noise.py \
        --model_path "${CKPT}" \
        --batch_size 64 \
        --num_batches 8 \
        --roc_train ../datasets/ROCstory
done

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
