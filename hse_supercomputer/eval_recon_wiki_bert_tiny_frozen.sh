#!/bin/bash
#SBATCH --job-name=eval_recon_wiki_frz
#SBATCH --partition=normal
#SBATCH --output=logs/eval_recon_wiki_bert_tiny_%j.out
#SBATCH --error=logs/eval_recon_wiki_bert_tiny_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Per-timestep reconstruction loss — Wiki BERT-tiny frozen 128d (FULL 2000 t)"
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

DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_wiki_pad_bert128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen_bsz256x3gpu_wu2000
CHECKPOINT=${DIFFUSION_MODEL_DIR}/ema_0.9999_166667.pt

echo ""
echo "Model path: ${CHECKPOINT}"
echo "Wiki validation JSON lives under wiki_corpus_train (relative to improved-diffusion/)"
echo "Starting per-timestep reconstruction evaluation..."
python -u scripts/eval_reconstruction.py \
    --model_path "${CHECKPOINT}" \
    --batch_size 64 \
    --num_batches 8 \
    --num_timesteps 2000 \
    --wiki_corpus_train ../datasets/roots_en_wikipedia

echo ""
echo "========================================"
echo "Evaluation completed at: $(date)"
echo "========================================"
