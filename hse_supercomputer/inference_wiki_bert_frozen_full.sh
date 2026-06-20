#!/bin/bash
#SBATCH --job-name=inference_wiki_bert_frozen_full
#SBATCH --partition=normal
#SBATCH --output=logs/inference_wiki_bert_frozen_full_%j.out
#SBATCH --error=logs/inference_wiki_bert_frozen_full_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Inference (batch decode + PPL under AR + MAUVE) for full Wikipedia diffusion with
# frozen bert-tiny 128d embeddings (training_mode emb), 3-GPU run:
#   train_wiki_bert_frozen_3gpu.sh -> diffusion_models/diff_wiki_pad_bert128_...
#
# Classifier for PPL (batch_decode.py wiki branch): same as random full wiki —
#   wiki_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_bert_uncased_wiki_partial
#
# MAUVE references: same file as the random full-wiki job:
#   improved-diffusion/generation_outputs/wiki_full_valid_refs.txt
# By default we do NOT resample if that file already exists (fair comparison to the
# e2e run; avoids overwriting). To resample from wiki_valid.json (seed 42, n=1000,
# same recipe as inference_wiki_bert_uncased_rand_full.sh), delete the refs file or set
#   FORCE_REGEN_WIKI_REFS=1
#
# MAUVE step mirrors mauve_wiki_bert_uncased_rand_full.sh (HF cache env + --n_refs 1000).
#
# Submit from repo root (Diffusion-LM).

echo "========================================"
echo "Inference: full Wikipedia + bert-tiny frozen (128d emb) + classifier + MAUVE"
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
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

echo "HF_HOME: ${HF_HOME}"
echo "HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE}"
echo "TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"

DIFFUSION_MODEL_DIR=scripts/diffusion_models/diff_wiki_pad_bert128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen_bsz256x3gpu_wu2000
MODEL_NAME=$(basename "${DIFFUSION_MODEL_DIR}")

cd improved-diffusion
mkdir -p generation_outputs

echo ""
echo "Diffusion model dir: ${DIFFUSION_MODEL_DIR}"

WIKI_REF_OUT="generation_outputs/wiki_full_valid_refs.txt"
WIKI_VALID_JSON="../datasets/roots_en_wikipedia_full/wiki_valid.json"

GEN_REF_ARGS=()
if [[ -f "${WIKI_REF_OUT}" && "${FORCE_REGEN_WIKI_REFS:-0}" != "1" ]]; then
    echo ""
    echo "Reusing existing MAUVE refs (not resampling): ${WIKI_REF_OUT}"
    echo "  Delete this file or FORCE_REGEN_WIKI_REFS=1 to regenerate from ${WIKI_VALID_JSON}"
else
    if [ ! -f "${WIKI_VALID_JSON}" ]; then
        echo "ERROR: Full-wiki validation JSON not found: ${WIKI_VALID_JSON}"
        echo "Prepare corpus with prepare_wikipedia_corpus (full) so roots_en_wikipedia_full/wiki_valid.json exists."
        exit 1
    fi
    GEN_REF_ARGS=(--gen_refs --ref_file "${WIKI_VALID_JSON}" --ref_out "${WIKI_REF_OUT}" --n_refs 1000 --ref_seed 42)
    echo ""
    echo "Writing MAUVE refs -> ${WIKI_REF_OUT} (same seed/n as inference_wiki_bert_uncased_rand_full.sh)"
fi

echo ""
echo "Starting batch decode + PPL..."
python -u scripts/batch_decode.py \
    "${DIFFUSION_MODEL_DIR}" \
    --top_p -1.0 \
    --pattern ema \
    --num_samples 1000 \
    "${GEN_REF_ARGS[@]}"

echo ""
echo "Starting MAUVE calculation..."
GENERATED=$(ls -t generation_outputs/"${MODEL_NAME}".ema_*.samples_-1.0.txt 2>/dev/null | head -1)
if [ -n "${GENERATED}" ] && [ -f "${WIKI_REF_OUT}" ]; then
    echo "Generated file: $GENERATED"
    echo "Reference file: ${WIKI_REF_OUT}"
    python -u scripts/calculate_mauve.py \
        --generated "$GENERATED" \
        --references "${WIKI_REF_OUT}" \
        --n_refs 1000 \
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
