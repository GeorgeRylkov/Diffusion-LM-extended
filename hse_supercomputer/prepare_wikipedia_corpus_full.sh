#!/bin/bash
#SBATCH --job-name=prep_wiki_full
#SBATCH --partition=normal
#SBATCH --output=logs/prepare_wikipedia_corpus_full_%j.out
#SBATCH --error=logs/prepare_wikipedia_corpus_full_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#
# Full ROOTS English Wikipedia → wiki_train.json / wiki_valid.json (no --max_articles).
# Loads all Parquet shards into RAM first; docstring warns ~60–80 GB RAM peak possible.
#
# Default output dir avoids overwriting the 50k smoke test in datasets/roots_en_wikipedia/:
#   OUTPUT_DIR=datasets/roots_en_wikipedia_full
# Override to replace in place, e.g.:
#   OUTPUT_DIR=datasets/roots_en_wikipedia sbatch hse_supercomputer/prepare_wikipedia_corpus_full.sh
#
# Same venv as prepare_wikipedia_corpus.sh — see requirements_wikipedia_tools.txt
#
# Submit from repository root:
#   cd /path/to/Diffusion-LM && sbatch hse_supercomputer/prepare_wikipedia_corpus_full.sh

echo "========================================"
echo "prepare_wikipedia_corpus FULL (Slurm)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cwd: $(pwd)"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

if [[ ! -f hse_supercomputer/.venv_wikipedia/bin/activate ]]; then
  echo "ERROR: missing hse_supercomputer/.venv_wikipedia (see prepare_wikipedia_corpus.sh)."
  exit 1
fi

# shellcheck source=/dev/null
source hse_supercomputer/.venv_wikipedia/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="${OUTPUT_DIR:-datasets/roots_en_wikipedia_full}"

echo ""
echo "Environment information:"
which python
python -V
echo "Output directory: ${OUTPUT_DIR}"

python -u prepare_wikipedia_corpus.py \
  --input_dir datasets/roots_en_wikipedia/data \
  --output_dir "${OUTPUT_DIR}"

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
