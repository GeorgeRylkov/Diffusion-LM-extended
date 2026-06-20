#!/bin/bash
#SBATCH --job-name=precompute_emb_stats
#SBATCH --partition=normal
#SBATCH --output=logs/precompute_emb_stats_%j.out
#SBATCH --error=logs/precompute_emb_stats_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0

echo "========================================"
echo "Precompute embedding normalization stats"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_hse/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

E2E_CHECKPOINT=improved-diffusion/scripts/diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased/ema_0.9999_400000.pt
OUTPUT=improved-diffusion/scripts/emb_norm_stats.pt

cd improved-diffusion/scripts

python -u precompute_emb_stats.py \
    --e2e_checkpoint ../../${E2E_CHECKPOINT} \
    --roc_train ../../datasets/ROCstory \
    --output ../../${OUTPUT}

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
