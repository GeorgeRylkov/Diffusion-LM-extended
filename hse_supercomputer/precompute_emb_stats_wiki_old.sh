#!/bin/bash
#SBATCH --job-name=precompute_wiki_emb_stats
#SBATCH --partition=normal
#SBATCH --output=logs/precompute_wiki_emb_stats_%j.out
#SBATCH --error=logs/precompute_wiki_emb_stats_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0

# Precompute corpus-weighted embedding statistics for the Wikipedia e2e model.
#
# What this produces
# ------------------
#   emb_norm_stats_wiki.pt
#       Always: tgt_mean, tgt_std, n_tokens, frozen_source_skipped, ...
#       If SKIP_FROZEN_SOURCE=0: also src_mean, src_std from BERT-tiny; Option
#       A/B verification lines appear in the log.
#       If SKIP_FROZEN_SOURCE=1 (default below): e2e-only stats — less RAM
#       (no BERT-tiny matrix resident during checkpoint load + e2e scan).
#
# The single number that matters
# ------------------------------
# In the log, use the measured e2e corpus mean L2 norm, e.g. the line:
#       "  Corpus mean L2 norm: X.XXXX"
# after "[e2e] Stage: corpus scan...". If SKIP_FROZEN_SOURCE=0, you can also
# read it from verification:
#       "Verification — Option B: z-score + uniform rescale (full corpus):"
#       "  E2E target        -> mean L2 norm: X.XXXX"
# That X.XXXX value replaces the placeholder
#       BERT_TARGET_CORPUS_NORMS['wiki'] = 4.055
# in improved-diffusion/improved_diffusion/text_datasets.py. Do that edit and
# commit before launching any run with --experiment bert --modality wiki;
# otherwise the frozen BERT-tiny embeddings will be rescaled to the ROC
# target norm (4.055) by accident.
#
# Why this script uses the Arrow dataset
# --------------------------------------
# collect_corpus_ids_arrow mmaps the pre-tokenized DatasetDict produced by
# tokenize_wiki_corpus.sh instead of re-tokenizing wiki_train.json on the
# fly. Same end result, much faster (~1 min vs ~30 min), and avoids holding
# ~100 GB of Python objects in RAM on the login-node-sized partition.
#
# Prerequisites
# -------------
#   1) datasets/roots_en_wikipedia/bert_uncased_seq64/ exists (Arrow).
#   2) diffusion_models/<WIKI_RUN>/ema_0.9999_<STEP>.pt exists — a trained
#      wiki e2e EMA checkpoint to extract word_embedding.weight from.
#   3) models/hf/bert-base-uncased exists (tokenizer snapshot).
#   4) If SKIP_FROZEN_SOURCE=0: models/hf/prajjwal1-bert-tiny for the frozen pass.
#
# Tunables (export before sbatch or edit below)
# ----------------------------------------------
#   SKIP_FROZEN_SOURCE=1   Skip BERT-tiny + src stats (default in this script).
#   SKIP_FROZEN_SOURCE=0   Full run with src_* and verification.
#   CORPUS_LOG_EVERY_CHUNKS=200   Progress every N token chunks (0 = off).
#
# Submit from the repo root (Slurm copies this script to /var/spool/... so we
# cannot locate the repo from the script path alone), e.g.:
#   cd /path/to/Diffusion-LM && sbatch hse_supercomputer/precompute_emb_stats_wiki.sh
# Or set DIFFUSION_LM_ROOT=/path/to/Diffusion-LM before sbatch.

echo "========================================"
echo "Precompute Wikipedia embedding norm stats"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

# Slurm copies this script to /var/spool/slurm/.../slurm_script, so
# dirname(BASH_SOURCE)/.. is NOT the repo. Resolve repo by venv location:
#   1) DIFFUSION_LM_ROOT if set (for odd sbatch cwd layouts)
#   2) initial cwd (normally SLURM submit directory when you `cd repo && sbatch`)
#   3) SLURM_SUBMIT_DIR if it contains the venv
#   4) parent of this script's directory, only when the script still lives under the repo
_venv_rel="hse_supercomputer/.venv_hse/bin/activate"
if [[ -n "${DIFFUSION_LM_ROOT:-}" && -f "${DIFFUSION_LM_ROOT}/${_venv_rel}" ]]; then
  REPO_ROOT="$(cd "${DIFFUSION_LM_ROOT}" && pwd)"
elif [[ -f "$(pwd -P)/${_venv_rel}" ]]; then
  REPO_ROOT="$(pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/${_venv_rel}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "${_script_dir}" != /var/spool/slurm/* && -f "${_script_dir}/../${_venv_rel}" ]]; then
    REPO_ROOT="$(cd "${_script_dir}/.." && pwd)"
  else
    echo "ERROR: cannot find Diffusion-LM repo (no ${_venv_rel})."
    echo "Fix: run from repo root, e.g.  cd /path/to/Diffusion-LM && sbatch hse_supercomputer/$(basename "${BASH_SOURCE[0]}")"
    echo "Or set:  export DIFFUSION_LM_ROOT=/path/to/Diffusion-LM  before sbatch"
    exit 1
  fi
fi
cd "${REPO_ROOT}" || exit 1
echo "REPO_ROOT: ${REPO_ROOT}"
TOKENIZER_MODEL="${REPO_ROOT}/models/hf/bert-base-uncased"
BERT_TINY_MODEL="${REPO_ROOT}/models/hf/prajjwal1-bert-tiny"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

set -e
source hse_supercomputer/.venv_hse/bin/activate
set +e

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Compute nodes may not reach huggingface.co; use repo snapshots only.
export TRANSFORMERS_OFFLINE=1

# --- Point at your trained wiki e2e run -------------------------------------
# With save_interval=16667 the last saved EMA step for the 3-GPU run was
# either 150003 or whatever `last_completed_save_step` is for your run.
# Adjust ${EMA_STEP} below if you resume training or pick a different step.
WIKI_RUN=diff_wiki_pad_rand128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased_bsz256x3gpu_wu2000
EMA_STEP=150003
E2E_CHECKPOINT=improved-diffusion/scripts/diffusion_models/${WIKI_RUN}/ema_0.9999_${EMA_STEP}.pt
# ----------------------------------------------------------------------------

# precompute_emb_stats.py: skip frozen pass (RAM); corpus progress in log.
SKIP_FROZEN_SOURCE="${SKIP_FROZEN_SOURCE:-1}"
CORPUS_LOG_EVERY_CHUNKS="${CORPUS_LOG_EVERY_CHUNKS:-200}"

WIKI_TOKENIZED_DIR=datasets/roots_en_wikipedia/bert_uncased_seq64
OUTPUT=improved-diffusion/scripts/emb_norm_stats_wiki.pt

if [ ! -f "${E2E_CHECKPOINT}" ]; then
    echo "ERROR: e2e checkpoint not found at ${REPO_ROOT}/${E2E_CHECKPOINT}"
    echo "Adjust WIKI_RUN and EMA_STEP at the top of this script."
    exit 1
fi
if [ ! -d "${WIKI_TOKENIZED_DIR}" ]; then
    echo "ERROR: pre-tokenized wiki dataset not found at ${WIKI_TOKENIZED_DIR}"
    echo "Run hse_supercomputer/tokenize_wiki_corpus_bert_uncased.sh first."
    exit 1
fi
if [ ! -d "${TOKENIZER_MODEL}" ]; then
    echo "ERROR: offline tokenizer snapshot not found at ${TOKENIZER_MODEL}"
    echo "On the login node, run once (from repo root):"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('google-bert/bert-base-uncased', local_dir='models/hf/bert-base-uncased')\""
    exit 1
fi
if [ "${SKIP_FROZEN_SOURCE}" != "1" ]; then
    if [ ! -d "${BERT_TINY_MODEL}" ]; then
        echo "ERROR: offline bert-tiny snapshot not found at ${BERT_TINY_MODEL}"
        echo "On the login node, run once:"
        echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('prajjwal1/bert-tiny', local_dir='models/hf/prajjwal1-bert-tiny')\""
        exit 1
    fi
fi

echo ""
echo "Checkpoint:           ${E2E_CHECKPOINT}"
echo "Pre-tokenized Arrow:  ${WIKI_TOKENIZED_DIR}"
echo "Output stats:         ${OUTPUT}"
echo "Tokenizer (local):    ${TOKENIZER_MODEL}"
echo "SKIP_FROZEN_SOURCE:   ${SKIP_FROZEN_SOURCE}"
echo "CORPUS_LOG_EVERY_CHUNKS: ${CORPUS_LOG_EVERY_CHUNKS}"
if [ "${SKIP_FROZEN_SOURCE}" != "1" ]; then
    echo "BERT-tiny (local):    ${BERT_TINY_MODEL}"
fi

cd "${REPO_ROOT}/improved-diffusion/scripts" || exit 1

set -- python -u precompute_emb_stats.py \
    --e2e_checkpoint "../../${E2E_CHECKPOINT}" \
    --wiki_tokenized_dir "../../${WIKI_TOKENIZED_DIR}" \
    --output "../../${OUTPUT}" \
    --tokenizer_model "${TOKENIZER_MODEL}" \
    --bert_tiny_model "${BERT_TINY_MODEL}"
if [ "${SKIP_FROZEN_SOURCE}" = "1" ]; then
    set -- "$@" --skip_frozen_source
fi
if [ "${CORPUS_LOG_EVERY_CHUNKS}" != "0" ]; then
    set -- "$@" --corpus_log_every_chunks "${CORPUS_LOG_EVERY_CHUNKS}"
fi
"$@"
rc=$?

if [ "${rc}" -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "FAILED: precompute_emb_stats.py exited with ${rc} at $(date)"
    echo "========================================"
    exit "${rc}"
fi

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "ACTION ITEM: take the e2e corpus mean L2 norm from the log (the line"
echo "  '  Corpus mean L2 norm: X.XXXX' after the e2e corpus scan) and paste"
echo "it into BERT_TARGET_CORPUS_NORMS['wiki'] in"
echo "  improved-diffusion/improved_diffusion/text_datasets.py"
echo "If you ran with SKIP_FROZEN_SOURCE=0, the Option B block prints the same number."
echo "========================================"
