#!/bin/bash
#SBATCH --job-name=eval_norescale
#SBATCH --partition=normal
#SBATCH --output=logs/eval_norescale_%j.out
#SBATCH --error=logs/eval_norescale_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Diagnostic: rescale_timesteps disabled"
echo "Model: BERT-tiny frozen 128d"
echo "If spike moves to t=500 -> model's fault at scaled_t=500"
echo "If spike stays at t=1000 -> noise schedule position"
echo "If spike disappears -> interaction effect"
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

echo ""
echo "Model path: ${CHECKPOINT}"
echo "Running full eval with --no_rescale True..."
python -u scripts/eval_reconstruction.py \
    --model_path "${CHECKPOINT}" \
    --batch_size 64 \
    --num_batches 8 \
    --num_timesteps 200 \
    --no_rescale True \
    --roc_train ../datasets/ROCstory

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
