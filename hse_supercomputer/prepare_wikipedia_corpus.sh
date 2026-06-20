#!/bin/bash
#SBATCH --job-name=prep_wiki_corpus
#SBATCH --partition=normal
#SBATCH --output=logs/prepare_wikipedia_corpus_%j.out
#SBATCH --error=logs/prepare_wikipedia_corpus_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#
# CPU-only: build wiki_train.json / wiki_valid.json from local ROOTS parquet.
# Venv lives next to .venv_hse (see eval_recon_frozen_v2.sh).
#
# One-time on login node (repo root) — upgrade pip first or pyarrow may pull an sdist
# (needs Cython). Prefer wheels: see hse_supercomputer/requirements_wikipedia_tools.txt
#   module purge && module load Python/PyTorch_GPU_v2.4 openmpi
#   python3 -m venv hse_supercomputer/.venv_wikipedia
#   source hse_supercomputer/.venv_wikipedia/bin/activate
#   pip install -U pip setuptools wheel
#   pip install --only-binary=:all: -r hse_supercomputer/requirements_wikipedia_tools.txt
#
# Submit from repository root:
#   cd /path/to/Diffusion-LM && sbatch hse_supercomputer/prepare_wikipedia_corpus.sh
# Full corpus (all articles, longer wall time): prepare_wikipedia_corpus_full.sh

echo "========================================"
echo "prepare_wikipedia_corpus (Slurm)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cwd: $(pwd)"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

if [[ ! -f hse_supercomputer/.venv_wikipedia/bin/activate ]]; then
  echo "ERROR: missing hse_supercomputer/.venv_wikipedia (see comments at top of this script)."
  exit 1
fi

# shellcheck source=/dev/null
source hse_supercomputer/.venv_wikipedia/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo ""
echo "Environment information:"
which python
python -V

MAX_ARTICLES="${MAX_ARTICLES:-50000}"

python -u prepare_wikipedia_corpus.py \
  --input_dir datasets/roots_en_wikipedia/data \
  --output_dir datasets/roots_en_wikipedia \
  --max_articles "${MAX_ARTICLES}"

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
