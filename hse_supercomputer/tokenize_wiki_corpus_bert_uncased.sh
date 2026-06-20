#!/bin/bash
#SBATCH --job-name=tokenize_wiki_bert_uncased
#SBATCH --partition=normal
#SBATCH --output=logs/tokenize_wiki_bert_uncased_%j.out
#SBATCH --error=logs/tokenize_wiki_bert_uncased_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0

# One-shot preprocessing: convert datasets/roots_en_wikipedia/wiki_{train,valid}.json
# (text-level JSONL produced by prepare_wikipedia_corpus.py) into a memory-mappable
# HuggingFace Arrow DatasetDict with pre-tokenized input_ids using BERT base uncased.
#
# After this runs, multi-GPU training jobs can mmap the same Arrow file across
# all ranks on a node, keeping per-rank RAM in the low single-digit GB range
# instead of ~100 GB (which would OOM any HSE V100 node above 1 GPU).
#
# NOTE on seqlen vs max_words: seqlen below is the WordPiece-token count the
# model operates on (= image_size**2 in the training script). max_words in
# prepare_wikipedia_corpus.py is the whitespace-word budget per passage.
# For bert-base-uncased, seqlen roughly needs to be >= 1.5*max_words to avoid
# truncating most passages. max_words=64 + seqlen=64 keeps the model shape
# compatible with image_size=8 but silently truncates ~50% of passages at the
# tail; that matches what the pre-existing single-GPU pipeline did.
#
# Expected runtime: ~2-4 h for the full 27M-passage corpus on 16 CPUs.
# Expected output size: ~7-8 GB for seqlen=64, bert-base-uncased.

# --- Tokenizer-specific parameters ----------------------------------------
TOKENIZER=bert-base-uncased
SEQLEN=64
INPUT_DIR=datasets/roots_en_wikipedia_full
# Output dir deliberately encodes tokenizer + seqlen so different tokenizer
# choices don't clobber each other.
OUTPUT_DIR=datasets/roots_en_wikipedia/bert_uncased_seq${SEQLEN}
# --------------------------------------------------------------------------

echo "========================================"
echo "Pre-tokenize Wikipedia corpus"
echo "Tokenizer: ${TOKENIZER}"
echo "Seqlen:    ${SEQLEN}"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4

source hse_supercomputer/.venv_hse/bin/activate

echo ""
echo "Environment information:"
which python
python -V

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo ""
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

python -u tokenize_wiki_corpus.py \
    --input_dir  ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --tokenizer  ${TOKENIZER} \
    --seqlen     ${SEQLEN} \
    --num_proc   16 \
    --writer_batch_size 10000

echo ""
echo "========================================"
echo "Tokenization completed at: $(date)"
echo "========================================"
