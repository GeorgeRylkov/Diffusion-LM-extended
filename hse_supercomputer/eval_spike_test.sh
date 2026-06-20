#!/bin/bash
#SBATCH --job-name=eval_spike_test
#SBATCH --partition=normal
#SBATCH --output=logs/eval_spike_test_%j.out
#SBATCH --error=logs/eval_spike_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Spike diagnostic: per-timestep eval around t=1000"
echo "Evaluating every timestep in [900, 1100]"
echo "Model: BERT-tiny frozen 128d"
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
echo "Evaluating t=[900..1100] at every single timestep..."
python -u scripts/eval_reconstruction.py \
    --model_path "${CHECKPOINT}" \
    --batch_size 64 \
    --num_batches 8 \
    --t_start 900 \
    --t_end 1100 \
    --roc_train ../datasets/ROCstory

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
