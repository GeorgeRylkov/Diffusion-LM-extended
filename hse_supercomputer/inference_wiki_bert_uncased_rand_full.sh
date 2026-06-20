#!/bin/bash
#SBATCH --job-name=inference_wiki_bert_uncased_rand_full
#SBATCH --partition=normal
#SBATCH --output=logs/inference_wiki_bert_uncased_rand_full_%j.out
#SBATCH --error=logs/inference_wiki_bert_uncased_rand_full_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Inference (batch decode + PPL under AR + MAUVE) for full Wikipedia diffusion
# (bert-base-uncased, random 128d e2e, 3-GPU training run).
#
# Diffusion: scripts/diffusion_models/diff_wiki_pad_rand128_..._bert_uncased_bsz256x3gpu_wu2000
# Classifier for PPL: same as partial wiki for now (batch_decode.py):
#   ../classifier_models/wiki_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_bert_uncased_wiki_partial
#
# MAUVE references: sampled from FULL corpus validation JSON (roots_en_wikipedia_full).
#
# Submit from repo root (Diffusion-LM). Ensure datasets/roots_en_wikipedia_full/wiki_valid.json exists.

echo "========================================"
echo "Inference: full Wikipedia + bert-uncased (random 128d e2e) + classifier"
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

# DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_wiki_pad_rand128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased_bsz256x3gpu_wu2000
DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_wiki_pad_rand128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased_bsz256x3gpu_wu2000_50h
MODEL_NAME=$(basename "${DIFFUSION_MODEL_DIR}")

cd improved-diffusion
mkdir -p generation_outputs

echo ""
echo "Diffusion model dir: ${DIFFUSION_MODEL_DIR}"

WIKI_REF_OUT="generation_outputs/wiki_full_valid_refs.txt"
WIKI_VALID_JSON="../datasets/roots_en_wikipedia_full/wiki_valid.json"

if [ ! -f "${WIKI_VALID_JSON}" ]; then
    echo "ERROR: Full-wiki validation JSON not found: ${WIKI_VALID_JSON}"
    echo "Prepare corpus with prepare_wikipedia_corpus (full) so roots_en_wikipedia_full/wiki_valid.json exists."
    exit 1
fi

echo ""
echo "Starting batch decode + PPL + ref sampling (full wiki valid -> ${WIKI_REF_OUT})..."
python -u scripts/batch_decode.py \
    "${DIFFUSION_MODEL_DIR}" \
    --top_p -1.0 \
    --pattern ema \
    --num_samples 1000 \
    --gen_refs \
    --ref_file "${WIKI_VALID_JSON}" \
    --ref_out "${WIKI_REF_OUT}" \
    --n_refs 1000 \
    --ref_seed 42

echo ""
echo "Starting MAUVE calculation..."
GENERATED=$(ls -t generation_outputs/"${MODEL_NAME}".ema_*.samples_-1.0.txt 2>/dev/null | head -1)
if [ -n "${GENERATED}" ] && [ -f "${WIKI_REF_OUT}" ]; then
    echo "Generated file: $GENERATED"
    echo "Reference file: ${WIKI_REF_OUT}"
    python -u scripts/calculate_mauve.py \
        --generated "$GENERATED" \
        --references "${WIKI_REF_OUT}" \
        --tokens_only \
        --bert_preprocess bert-base-uncased
else
    echo "ERROR: Missing generated samples and/or reference file."
    echo "  expected generated: generation_outputs/${MODEL_NAME}.ema_*.samples_-1.0.txt"
    echo "  expected refs:      ${WIKI_REF_OUT}"
fi

echo ""
echo "========================================"
echo "Inference completed at: $(date)"
echo "========================================"
