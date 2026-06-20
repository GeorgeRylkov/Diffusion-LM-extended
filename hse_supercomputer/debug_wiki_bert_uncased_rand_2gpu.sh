#!/bin/bash
#SBATCH --job-name=debug_wiki_bert_uncased_rand_2gpu
#SBATCH --partition=normal
#SBATCH --output=logs/debug_wiki_bert_uncased_rand_2gpu_%j.out
#SBATCH --error=logs/debug_wiki_bert_uncased_rand_2gpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2

# DEBUG variant of train_wiki_bert_uncased_rand_4gpu.sh on 2 GPUs.
# Purpose: verify DDP/MPI scaling (gradient sync, dataset sharding, per-rank
# device assignment, multi-process checkpoint writer) before the 4-GPU run.
#
# Mirrors debug_wiki_bert_uncased_rand_1gpu.sh but with 2 ranks and a global
# batch 2x, so if DDP has a rank-hang / collective / all_reduce issue it shows
# up here instead of in a 2-day 4-GPU slot.

echo "========================================"
echo "DEBUG: Diffusion-LM Wikipedia + bert-base-uncased (random 128d e2e), 2 GPUs"
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

export GPUS_PER_NODE=2

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mpi_cuda_support=0

export OPENAI_LOGDIR=debug_runs/wiki_bert_uncased_rand_2gpu_${SLURM_JOB_ID}

WIKI_TOKENIZED_DIR=../../datasets/roots_en_wikipedia/bert_uncased_seq64

mkdir -p ${OPENAI_LOGDIR}

cd improved-diffusion/scripts

if [ ! -d "${WIKI_TOKENIZED_DIR}" ]; then
    echo "ERROR: pre-tokenized wiki dataset not found at ${WIKI_TOKENIZED_DIR}"
    echo "Run hse_supercomputer/tokenize_wiki_corpus_bert_uncased.sh first."
    exit 1
fi

echo ""
echo "Starting 2-GPU DEBUG run..."
echo "Pre-tokenized wiki:  ${WIKI_TOKENIZED_DIR}"
echo "Output directory:    ${OPENAI_LOGDIR}"

srun --mpi=pmix python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality wiki \
    --save_interval 100 \
    --lr 0.0004 \
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
echo "DEBUG (2 GPU) run completed at: $(date)"
echo "========================================"
