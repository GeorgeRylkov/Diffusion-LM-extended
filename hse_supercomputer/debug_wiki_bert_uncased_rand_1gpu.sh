#!/bin/bash
#SBATCH --job-name=debug_wiki_bert_uncased_rand_1gpu
#SBATCH --partition=test
#SBATCH --output=logs/debug_wiki_bert_uncased_rand_1gpu_%j.out
#SBATCH --error=logs/debug_wiki_bert_uncased_rand_1gpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# DEBUG variant of train_wiki_bert_uncased_rand_4gpu.sh on 1 GPU.
# Purpose: surface bugs (dataloader, OOM, tokenizer, Arrow path, model shapes,
# logging, checkpointing) in ~minutes before burning the multi-day 4-GPU queue.
#
# What's reduced vs the production 4-GPU script:
#   - Time limit:      36h      -> 30 min
#   - Ranks/GPUs:      4/4      -> 1/1
#   - lr_anneal_steps: 125000   -> 200
#   - lr_warmup_steps: 1000     -> 50
#   - save_interval:  12500     -> 100   (exercise the checkpoint path once)
#   - lr:             8e-4      -> 2e-4  (unscaled 1-GPU baseline)
#   - log_interval (env): 10    (see train.py LOG_INTERVAL handling)
#   - OPENAI_LOGDIR    separate "debug_runs/..." dir, job-id suffixed
#
# Unchanged (so bugs that would hit the real run still hit this one):
#   per-GPU batch_size, diffusion_steps, noise_schedule, model dims,
#   dataset path, tokenizer, vocab_size, cache_mode.

echo "========================================"
echo "DEBUG: Diffusion-LM Wikipedia + bert-base-uncased (random 128d e2e), 1 GPU"
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

export GPUS_PER_NODE=1

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mpi_cuda_support=0

export OPENAI_LOGDIR=debug_runs/wiki_bert_uncased_rand_1gpu_${SLURM_JOB_ID}

WIKI_TOKENIZED_DIR=../../datasets/roots_en_wikipedia/bert_uncased_seq64

mkdir -p ${OPENAI_LOGDIR}

cd improved-diffusion/scripts

if [ ! -d "${WIKI_TOKENIZED_DIR}" ]; then
    echo "ERROR: pre-tokenized wiki dataset not found at ${WIKI_TOKENIZED_DIR}"
    echo "Run hse_supercomputer/tokenize_wiki_corpus_bert_uncased.sh first."
    exit 1
fi

echo ""
echo "Starting 1-GPU DEBUG run..."
echo "Pre-tokenized wiki:  ${WIKI_TOKENIZED_DIR}"
echo "Output directory:    ${OPENAI_LOGDIR}"

srun --mpi=pmix python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality wiki \
    --save_interval 100 \
    --lr 0.0002 \
    --batch_size 256 \
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
    --experiment random \
    --lr_anneal_steps 200 \
    --lr_warmup_steps 50 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode e2e \
    --vocab_size 30522 \
    --cache_mode no \
    --use_bert_tokenizer yes \
    --wiki_corpus_train ../../datasets/roots_en_wikipedia \
    --wiki_tokenized_dir ${WIKI_TOKENIZED_DIR}

echo ""
echo "========================================"
echo "DEBUG (1 GPU) run completed at: $(date)"
echo "========================================"
