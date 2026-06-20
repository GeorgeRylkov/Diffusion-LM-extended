#!/bin/bash
#SBATCH --job-name=precompute_wiki_emb_stats
#SBATCH --partition=normal
#SBATCH --output=logs/precompute_wiki_emb_stats_%j.out
#SBATCH --error=logs/precompute_wiki_emb_stats_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

# Precompute corpus-weighted embedding statistics for the Wikipedia e2e model.
#
# What this produces
# ------------------
#   emb_norm_stats_wiki.pt
#       With default (full) run: src_mean, src_std, tgt_mean, tgt_std, n_tokens, ...
#       With SKIP_FROZEN_SOURCE=1: tgt_* only (e2e-only; lower RAM on small nodes).
#
# The single number that matters
# ------------------------------
# At the end of the log, look for:
#       "E2E target -> mean L2 norm" (streaming) or Option B verification lines.
# That value replaces BERT_TARGET_CORPUS_NORMS['wiki'] in
#   improved-diffusion/improved_diffusion/text_datasets.py
#
# Why Arrow + precompute_emb_stats.py
# ------------------------------------------
# Mmap'd DatasetDict from tokenize_wiki_corpus.sh; remote script streams rows
# (no flat ~1.7B-token tensor), mmap checkpoint load, optional tiny
# E2E_WORD_EMBEDDING_PT to avoid unpickling the full EMA on low-RAM nodes.
#
# Submit from the Diffusion-LM repo root (same as other hse_supercomputer jobs).
#
# Optional env overrides
# ----------------------
#   SKIP_FROZEN_SOURCE=1     e2e stats only (no BERT-tiny / no verification).
#   E2E_WORD_EMBEDDING_PT=path/from/repo/root.pt   small .pt with word_embedding.weight only.
#   TOKENIZER_MODEL=...      default: models/hf/bert-base-uncased (offline).
#   BERT_TINY_MODEL=...      default: models/hf/prajjwal1-bert-tiny (if not skipping frozen).
#   NO_MMAP_CHECKPOINT=1     pass --no_mmap_checkpoint (NFS / loader issues).
#   CORPUS_LOG_EVERY_CHUNKS  default 200; 0 disables progress lines.
#   GC_EVERY_CHUNKS          default 500; 0 disables periodic gc.collect.
#   MAX_TOKENS               default 700_000_000 (~41% of wiki bert_uncased_seq64;
#                            stays before the observed late-run slowdown). Set 0 for full corpus.

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT" || {
    echo "ERROR: cannot cd to REPO_ROOT=$REPO_ROOT"
    exit 1
}

echo "========================================"
echo "Precompute Wikipedia embedding norm stats"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "REPO_ROOT: $REPO_ROOT"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_hse/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Optional: reduce glibc holding freed RAM on long CPU loops (Linux).
export MALLOC_TRIM_THRESHOLD_=100000

# --- Point at your trained wiki e2e run -------------------------------------
WIKI_RUN=diff_wiki_pad_rand128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased_bsz256x3gpu_wu2000
EMA_STEP=150003
E2E_CHECKPOINT=improved-diffusion/scripts/diffusion_models/${WIKI_RUN}/ema_0.9999_${EMA_STEP}.pt
# Optional: path from repo root to a tiny .pt (only word_embedding.weight).
# Create once on the login node, e.g.:
#   python -c "import torch; d=torch.load('.../ema_....pt',map_location='cpu',weights_only=True);\
#     torch.save({'word_embedding.weight': d['word_embedding.weight'].float().cpu()}, 'we.pt')"
E2E_WORD_EMBEDDING_PT="${E2E_WORD_EMBEDDING_PT:-}"
# ----------------------------------------------------------------------------

WIKI_TOKENIZED_DIR=datasets/roots_en_wikipedia/bert_uncased_seq64
OUTPUT=improved-diffusion/scripts/emb_norm_stats_wiki.pt

TOKENIZER_MODEL="${TOKENIZER_MODEL:-models/hf/bert-base-uncased}"
BERT_TINY_MODEL="${BERT_TINY_MODEL:-models/hf/prajjwal1-bert-tiny}"
SKIP_FROZEN_SOURCE="${SKIP_FROZEN_SOURCE:-0}"
CORPUS_LOG_EVERY_CHUNKS="${CORPUS_LOG_EVERY_CHUNKS:-200}"
GC_EVERY_CHUNKS="${GC_EVERY_CHUNKS:-500}"
MAX_TOKENS="${MAX_TOKENS:-700000000}"

if [ -n "$E2E_WORD_EMBEDDING_PT" ]; then
    if [ -f "$E2E_WORD_EMBEDDING_PT" ]; then
        WE_PATH="$E2E_WORD_EMBEDDING_PT"
    elif [ -f "${REPO_ROOT}/${E2E_WORD_EMBEDDING_PT}" ]; then
        WE_PATH="${REPO_ROOT}/${E2E_WORD_EMBEDDING_PT}"
    else
        echo "ERROR: E2E_WORD_EMBEDDING_PT set but file not found: $E2E_WORD_EMBEDDING_PT"
        exit 1
    fi
    if [[ "$WE_PATH" = /* ]]; then
        EMB_ARG=(--e2e_word_embedding_pt "$WE_PATH")
    else
        EMB_ARG=(--e2e_word_embedding_pt "../../${E2E_WORD_EMBEDDING_PT}")
    fi
else
    if [ ! -f "${REPO_ROOT}/${E2E_CHECKPOINT}" ]; then
        echo "ERROR: e2e checkpoint not found at ${E2E_CHECKPOINT}"
        echo "Adjust WIKI_RUN and EMA_STEP, or set E2E_WORD_EMBEDDING_PT to a small we.pt."
        exit 1
    fi
    EMB_ARG=(--e2e_checkpoint "../../${E2E_CHECKPOINT}")
fi

if [ ! -d "${REPO_ROOT}/${WIKI_TOKENIZED_DIR}" ]; then
    echo "ERROR: pre-tokenized wiki dataset not found at ${WIKI_TOKENIZED_DIR}"
    echo "Run hse_supercomputer/tokenize_wiki_corpus_bert_uncased.sh first."
    exit 1
fi

if [ ! -e "${REPO_ROOT}/${TOKENIZER_MODEL}" ] && [[ "$TOKENIZER_MODEL" != /* ]]; then
    echo "WARNING: tokenizer path may be missing: ${TOKENIZER_MODEL} (Hub download may fail offline)."
fi

echo ""
if [ -n "$E2E_WORD_EMBEDDING_PT" ]; then
    echo "Word embedding only:  ${E2E_WORD_EMBEDDING_PT}"
else
    echo "Checkpoint (full):    ${E2E_CHECKPOINT}"
fi
echo "Pre-tokenized Arrow:  ${WIKI_TOKENIZED_DIR}"
echo "Output stats:         ${OUTPUT}"
echo "Tokenizer (local):    ${TOKENIZER_MODEL}"
echo "SKIP_FROZEN_SOURCE:   ${SKIP_FROZEN_SOURCE}"
echo "CORPUS_LOG_EVERY_CHUNKS: ${CORPUS_LOG_EVERY_CHUNKS}"
echo "GC_EVERY_CHUNKS:      ${GC_EVERY_CHUNKS}"
echo "MAX_TOKENS:           ${MAX_TOKENS} (0 = full corpus)"

cd improved-diffusion/scripts || exit 1

PY_ARGS=(
    -u precompute_emb_stats.py
    "${EMB_ARG[@]}"
    --wiki_tokenized_dir "../../${WIKI_TOKENIZED_DIR}"
    --output "../../${OUTPUT}"
    --tokenizer_model "../../${TOKENIZER_MODEL}"
    --bert_tiny_model "../../${BERT_TINY_MODEL}"
    --corpus_log_every_chunks "${CORPUS_LOG_EVERY_CHUNKS}"
    --gc_every_chunks "${GC_EVERY_CHUNKS}"
    --max_tokens "${MAX_TOKENS}"
)

if [ "$SKIP_FROZEN_SOURCE" = "1" ]; then
    PY_ARGS+=(--skip_frozen_source)
fi
if [ "${NO_MMAP_CHECKPOINT:-0}" = "1" ]; then
    PY_ARGS+=(--no_mmap_checkpoint)
fi

python "${PY_ARGS[@]}"
exit_code=$?

echo ""
echo "========================================"
if [ "$exit_code" -eq 0 ]; then
    echo "Finished at: $(date)"
else
    echo "FAILED: precompute_emb_stats.py exited with ${exit_code} at $(date)"
fi
echo ""
echo "ACTION ITEM: take the 'E2E target -> mean L2 norm' value printed above"
echo "and paste it into BERT_TARGET_CORPUS_NORMS['wiki'] in"
echo "  improved-diffusion/improved_diffusion/text_datasets.py"
echo "========================================"

exit "$exit_code"
