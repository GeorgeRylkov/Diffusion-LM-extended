#!/bin/bash
#SBATCH --job-name=mauve_roc_fasttext
#SBATCH --partition=test
#SBATCH --output=logs/mauve_roc_fasttext_%j.out
#SBATCH --error=logs/mauve_roc_fasttext_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# MAUVE-only evaluation for ROCStories + FastText
# Reuses generated samples from job 3692546

echo "========================================"
echo "MAUVE evaluation: ROCStories + FastText"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

GENERATED=generation_outputs/diff_roc_pad_fasttext128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101.ema_0.9999_400000.pt.samples_-1.0.txt
REFERENCES=generation_outputs/cleaned_texts_mauve/roc_references_preprocessed.txt

echo ""
echo "Generated: $GENERATED"
echo "References: $REFERENCES"
echo ""

if [ ! -f "$GENERATED" ] || [ ! -f "$REFERENCES" ]; then
    echo "ERROR: Missing input files"
    ls -l "$GENERATED" "$REFERENCES" 2>&1
    exit 1
fi

echo "Starting MAUVE calculation..."
python -u scripts/calculate_mauve.py \
    --generated "$GENERATED" \
    --references "$REFERENCES" \
    --tokens_only

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
