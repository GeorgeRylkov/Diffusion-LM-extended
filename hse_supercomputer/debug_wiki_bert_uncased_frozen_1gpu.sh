#!/bin/bash
#SBATCH --job-name=debug_wiki_bert_uncased_frozen_1gpu
#SBATCH --partition=test
#SBATCH --output=logs/debug_wiki_bert_uncased_frozen_1gpu_%j.out
#SBATCH --error=logs/debug_wiki_bert_uncased_frozen_1gpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

# Training: Diffusion-LM on FULL Wikipedia (~27M passages), bert-base-uncased
# tokenizer, FROZEN bert-tiny (128d) pretrained embeddings (corpus-normalized),
# 1 GPU DDP.
#
# This is the frozen-embeddings sibling of train_wiki_bert_uncased_rand_3gpu.sh.
# Key differences from the random-embedding variant:
#   - --experiment bert          (not random)
#   - --training_mode emb        (not e2e; emb-space objective for frozen emb)
#   - Embedding source: prajjwal1/bert-tiny 128d → z-score normalize over
#     corpus tokens → rescale to TARGET_CORPUS_NORM['wiki']=4.055.
#
# Prerequisites:
#   1) prepare_wikipedia_corpus.py has produced datasets/roots_en_wikipedia/wiki_{train,valid}.json
#   2) hse_supercomputer/tokenize_wiki_corpus_bert_uncased.sh has produced
#      datasets/roots_en_wikipedia/bert_uncased_seq64/ (Arrow DatasetDict)
#   3) prajjwal1/bert-tiny weights are in the HF cache at
#      ~/.cache/huggingface/hub/models--prajjwal1--bert-tiny/ (load_bert_embeddings
#      points at that path). If missing, run once on a login node:
#        python -c "from transformers import BertModel; BertModel.from_pretrained('prajjwal1/bert-tiny')"
#
# How the fast path handles --experiment bert:
#   Normally load_bert_embeddings needs a Python sentence_lst to compute the
#   corpus-weighted z-score stats (would OOM on 3 ranks × ~100 GB each).
#   Instead, rank 0 derives corpus_ids directly from the Arrow dataset
#   (flatten input_ids, strip pad), normalizes, and writes random_emb.torch
#   into ${OPENAI_LOGDIR}. Other ranks wait on a dist.barrier() and then
#   load the file. Only the first run pays this cost (~1 min on rank 0);
#   subsequent restarts of the same ${OPENAI_LOGDIR} skip normalization.
#
# Hyperparameters: baseline is the 1-GPU frozen run (lr=1e-4, batch=64,
# lr_anneal_steps=400000). For 3 GPUs we bump batch to 128 per GPU (global
# 384, 6x the 1-GPU 64), square-root scale lr, and scale steps so total
# samples seen stays the same as the 1-GPU run.
#   - per-GPU batch_size: 64   -> 128       (2x; frozen emb gradient is cheap)
#   - global batch:       64   -> 384       (6x)
#   - lr:                 1e-4 -> 2.5e-4    (sqrt(6) scaling ≈ 2.45)
#   - lr_anneal_steps:    400K -> 66666     (same 25.6M samples: 66666*384 ≈ 25.6M)
#   - lr_warmup_steps:    0    -> 2000
#   - save_interval:      50K  -> 8333
#
# Sanity check: loss_q0..loss_q3 must stay separated (q0 < q1 < q2 < q3). If
# they collapse to a single value, training has degenerated; drop lr by half
# and re-run (same procedure as the random-emb runs).

echo "========================================"
echo "Diffusion-LM Training: FULL Wikipedia + bert-tiny (128d) FROZEN embeddings, 3 GPUs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_hse/bin/activate

echo ""
echo "Environment information:"
which python
python -V
which mpirun
nvidia-smi

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "WandB mode: $WANDB_MODE"

# One MPI rank per GPU; dist_util.dev() uses GPUS_PER_NODE for assignment.
export GPUS_PER_NODE=3

# MPI knobs that match debug_quick_2gpu.sh and avoid PML/CUDA-IPC hangs on HSE.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mpi_cuda_support=0

export OPENAI_LOGDIR=diffusion_models/diff_wiki_pad_bert128_transformer_lr0.00025_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen_bsz128x3gpu_wu2000

WIKI_TOKENIZED_DIR=../../datasets/roots_en_wikipedia/bert_uncased_seq64

cd improved-diffusion/scripts

if [ ! -d "${WIKI_TOKENIZED_DIR}" ]; then
    echo "ERROR: pre-tokenized wiki dataset not found at ${WIKI_TOKENIZED_DIR}"
    echo "Run hse_supercomputer/tokenize_wiki_corpus_bert_uncased.sh first."
    exit 1
fi

echo ""
echo "Starting 3-GPU frozen-BERT training..."
echo "Pre-tokenized wiki:  ${WIKI_TOKENIZED_DIR}"
echo "Output directory:    ${OPENAI_LOGDIR}"

srun --mpi=pmix python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality wiki \
    --save_interval 8333 \
    --lr 0.00025 \
    --batch_size 128 \
    --diffusion_steps 2000 \
    --noise_schedule sqrt \
    --use_kl False \
    --learn_sigma False \
    --image_size 8 \
    --num_channels 128 \
    --seed 101 \
    --dropout 0.1 \
    --in_channel 128 \
    --out_channel 128 \
    --padding_mode pad \
    --experiment bert \
    --lr_anneal_steps 66666 \
    --lr_warmup_steps 2000 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode emb \
    --vocab_size 30522 \
    --cache_mode no \
    --use_bert_tokenizer yes \
    --wiki_corpus_train ../../datasets/roots_en_wikipedia \
    --wiki_tokenized_dir ${WIKI_TOKENIZED_DIR}
    # Note: --wiki_corpus_train (raw JSONL) is currently unused — the
    # --wiki_tokenized_dir Arrow fast path in get_corpus_rocstory short-circuits
    # before JSON is touched. Kept as a fallback so the job does not fail if
    # someone changes --experiment, --use_bert_tokenizer, or --modality and
    # drops out of the fast path.

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
