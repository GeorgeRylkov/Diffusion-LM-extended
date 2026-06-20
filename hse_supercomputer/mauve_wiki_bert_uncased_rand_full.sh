#!/bin/bash
#SBATCH --job-name=mauve_wiki_bert_uncased_rand_full
#SBATCH --partition=normal
#SBATCH --output=logs/mauve_wiki_bert_uncased_rand_full_%j.out
#SBATCH --error=logs/mauve_wiki_bert_uncased_rand_full_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# MAUVE only for full Wikipedia + bert-base-uncased (no sampling / PPL / ref generation).
# Same pairing as inference_wiki_bert_uncased_rand_full.sh: newest
#   generation_outputs/<MODEL_NAME>.ema_*.samples_-1.0.txt
# vs generation_outputs/wiki_full_valid_refs.txt
#
# Submit from repo root (Diffusion-LM).

echo "========================================"
echo "MAUVE only: wiki full + bert-base-uncased"
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
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
# Optional: set TRANSFORMERS_OFFLINE=1 to force strictly local cache usage.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

echo "HF_HOME: ${HF_HOME}"
echo "HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE}"
echo "TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"

# DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_wiki_pad_rand128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased_bsz256x3gpu_wu2000
DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_wiki_pad_rand128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased_bsz256x3gpu_wu2000_2
MODEL_NAME=$(basename "${DIFFUSION_MODEL_DIR}")

cd improved-diffusion
mkdir -p generation_outputs

WIKI_REF_OUT="generation_outputs/wiki_full_valid_refs.txt"
GENERATED=$(ls -t generation_outputs/"${MODEL_NAME}".ema_*.samples_-1.0.txt 2>/dev/null | head -1)

echo ""
echo "MODEL_NAME: ${MODEL_NAME}"
echo "Generated:  ${GENERATED:-<not found>}"
echo "References: ${WIKI_REF_OUT}"

if [ -n "${GENERATED}" ] && [ -f "${GENERATED}" ] && [ -f "${WIKI_REF_OUT}" ]; then
    python -u scripts/calculate_mauve.py \
        --generated "${GENERATED}" \
        --references "${WIKI_REF_OUT}" \
        --n_refs 1000 \
        --tokens_only \
        --bert_preprocess bert-base-uncased
else
    echo "ERROR: Missing generated samples and/or reference file."
    echo "  expected generated: generation_outputs/${MODEL_NAME}.ema_*.samples_-1.0.txt"
    echo "  expected refs:      ${WIKI_REF_OUT}"
    exit 1
fi

echo ""
echo "========================================"
echo "MAUVE job finished at: $(date)"
echo "========================================"
