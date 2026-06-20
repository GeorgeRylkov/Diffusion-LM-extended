#!/bin/bash
#SBATCH --job-name=ppl_bert_uncased
#SBATCH --partition=normal
#SBATCH --output=logs/ppl_bert_uncased_%j.out
#SBATCH --error=logs/ppl_bert_uncased_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "PPL calculation: bert-base-uncased 128d e2e"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_hse/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd improved-diffusion

python -u scripts/ppl_under_ar.py \
    --model_path scripts/diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased/ema_0.9999_400000.pt \
    --modality roc \
    --experiment random \
    --model_name_or_path ../classifier_models/roc_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_bert_uncased \
    --input_text generation_outputs/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased.ema_0.9999_400000.pt.samples_-1.0.txt \
    --mode eval

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
