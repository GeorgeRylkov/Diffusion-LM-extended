#!/bin/bash
#SBATCH --job-name=eval_recon_bert_rand
#SBATCH --partition=normal
#SBATCH --output=logs/eval_recon_bert_rand_%j.out
#SBATCH --error=logs/eval_recon_bert_rand_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Per-timestep reconstruction loss evaluation"
echo "Model: BERT random e2e"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_hse/bin/activate

echo ""
echo "Environment information:"
which python
python -V
nvidia-smi

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd improved-diffusion

DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_v2
CHECKPOINT=${DIFFUSION_MODEL_DIR}/ema_0.9999_400000.pt

echo ""
echo "Model path: ${CHECKPOINT}"
echo "Starting per-timestep reconstruction evaluation..."
python -u scripts/eval_reconstruction.py \
    --model_path "${CHECKPOINT}" \
    --batch_size 64 \
    --num_batches 8 \
    --num_timesteps 200 \
    --roc_train ../datasets/ROCstory

echo ""
echo "========================================"
echo "Evaluation completed at: $(date)"
echo "========================================"
