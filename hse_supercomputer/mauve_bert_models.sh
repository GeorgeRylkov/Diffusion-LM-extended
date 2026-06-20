#!/bin/bash
#SBATCH --job-name=mauve_bert_models
#SBATCH --partition=normal
#SBATCH --output=logs/mauve_bert_models_%j.out
#SBATCH --error=logs/mauve_bert_models_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "MAUVE evaluation: BERT models (old repo)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_hse/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd improved-diffusion

REFERENCES=generation_outputs/roc_references.txt

# BERT models: tokenizer.decode() produces clean natural text,
# so raw references are already in the same space. No vocab preprocessing needed.
# Just --tokens_only to strip [PAD] from generated text.

# --- BERT frozen (frozen BERT embeddings, BERT tokenizer) ---
echo ""
echo "========================================"
echo "  BERT frozen"
echo "========================================"
python -u scripts/calculate_mauve.py \
    --generated "generation_outputs/diff_roc_pad_bert768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_v2.ema_0.9999_400000.pt.samples_-1.0.txt" \
    --references "$REFERENCES" \
    --tokens_only

# --- BERT random e2e (random 768d embeddings, BERT tokenizer) ---
echo ""
echo "========================================"
echo "  BERT random e2e"
echo "========================================"
python -u scripts/calculate_mauve.py \
    --generated "generation_outputs/diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_v2.ema_0.9999_400000.pt.samples_-1.0.txt" \
    --references "$REFERENCES" \
    --tokens_only

echo ""
echo "========================================"
echo "All done at: $(date)"
echo "========================================"
