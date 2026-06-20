#!/bin/bash
#SBATCH --job-name=eval_spike_e2e
#SBATCH --partition=normal
#SBATCH --output=logs/eval_spike_e2e_%j.out
#SBATCH --error=logs/eval_spike_e2e_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Spike diagnostic: non-BERT e2e-tgt model"
echo "Model: e2e-tgt random 64d (SpaCy tokenizer)"
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

DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_e2e-tgt_pad_rand64_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_e2e_emb_training
CHECKPOINT=${DIFFUSION_MODEL_DIR}/ema_0.9999_200000.pt

echo ""
echo "=== Full 200-point eval ==="
python -u scripts/eval_reconstruction.py \
    --model_path "${CHECKPOINT}" \
    --batch_size 64 \
    --num_batches 8 \
    --num_timesteps 200

echo ""
echo "=== Spike zoom: t=[900..1100] ==="
python -u scripts/eval_reconstruction.py \
    --model_path "${CHECKPOINT}" \
    --batch_size 64 \
    --num_batches 8 \
    --t_start 900 \
    --t_end 1100

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
