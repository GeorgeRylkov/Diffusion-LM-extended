#!/bin/bash
#SBATCH --job-name=inference_roc_bert_rand_tan_2
#SBATCH --partition=normal
#SBATCH --output=logs/inference_roc_bert_rand_tan_2_%j.out
#SBATCH --error=logs/inference_roc_bert_rand_tan_2_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Inference (batch decode) for ROCStories with BERT random embeddings
# HSE Supercomputer

echo "========================================"
echo "Inference: ROCStories + BERT random, tan-d noise schedule, d=2"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Purge all modules and load PyTorch module
module purge
module load Python/PyTorch_GPU_v2.4 openmpi

# Activate venv
source hse_supercomputer/.venv_hse/bin/activate

# Show environment info
echo ""
echo "Environment information:"
which python
python -V
nvidia-smi

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_tan_2_Lsimple_h128_s2_d0.1_sd101_bert
MODEL_NAME=$(basename ${DIFFUSION_MODEL_DIR})

# batch_decode.py runs from improved-diffusion/
cd improved-diffusion
mkdir -p generation_outputs

echo ""
echo "Diffusion model dir: ${DIFFUSION_MODEL_DIR}"
echo "Starting batch decode + reference generation..."
python -u scripts/batch_decode.py \
    "${DIFFUSION_MODEL_DIR}" \
    --top_p -1.0 \
    --pattern ema \
    --num_samples 1000 \
    --gen_refs \
    --ref_file ../datasets/ROCstory/roc_valid.json \
    --n_refs 1000

echo ""
echo "Starting MAUVE calculation..."
GENERATED=$(ls -t generation_outputs/${MODEL_NAME}.ema_*.samples_-1.0.txt 2>/dev/null | head -1)
if [ -n "$GENERATED" ]; then
    python -u scripts/calculate_mauve.py \
        --generated "$GENERATED" \
        --references generation_outputs/roc_references.txt \
        --tokens_only \
        --bert_preprocess bert-base-cased
else
    echo "ERROR: No generated samples file found for ${MODEL_NAME}"
fi

echo ""
echo "========================================"
echo "Inference completed at: $(date)"
echo "========================================"
