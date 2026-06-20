#!/bin/bash
#SBATCH --job-name=inference_e2e_bert_tiny_frozen
#SBATCH --partition=normal
#SBATCH --output=logs/inference_e2e_bert_tiny_frozen_%j.out
#SBATCH --error=logs/inference_e2e_bert_tiny_frozen_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Inference (batch decode) for E2E-tgt with frozen bert-tiny 128d embeddings

echo "========================================"
echo "Inference: E2E-tgt + frozen bert-tiny 128d"
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

DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_e2e-tgt_pad_bert128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen
MODEL_NAME=$(basename ${DIFFUSION_MODEL_DIR})

cd improved-diffusion
mkdir -p generation_outputs

echo ""
echo "Diffusion model dir: ${DIFFUSION_MODEL_DIR}"
echo "Starting batch decode..."
python -u scripts/batch_decode.py \
    "${DIFFUSION_MODEL_DIR}" \
    --top_p -1.0 \
    --pattern ema \
    --num_samples 1000 \
    --out_dir generation_outputs

echo ""
echo "=== MAUVE ==="
GENERATED=$(ls -t generation_outputs/${MODEL_NAME}.ema_*.samples_-1.0.txt 2>/dev/null | head -1)
if [ -n "$GENERATED" ]; then
    echo "Generated file: $GENERATED"
    python -u scripts/calculate_mauve.py \
        --generated "$GENERATED" \
        --references generation_outputs/e2e_references.txt \
        --bert_preprocess bert-base-uncased
else
    echo "ERROR: No generated samples file found for ${MODEL_NAME}"
fi

echo ""
echo "========================================"
echo "Inference completed at: $(date)"
echo "========================================"
