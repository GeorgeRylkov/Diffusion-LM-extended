#!/bin/bash
#SBATCH --job-name=diffusion_wiki_bert_uncased_rand_3gpu_50h
#SBATCH --partition=normal
#SBATCH --output=logs/train_wiki_bert_uncased_rand_3gpu_50h_%j.out
#SBATCH --error=logs/train_wiki_bert_uncased_rand_3gpu_50h_%j.err
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --gpus=3

# Training: Diffusion-LM on FULL Wikipedia (~27M passages), bert-base-uncased tokenizer,
# random 128d embeddings (e2e), 3 GPUs DDP.
#
# Middle ground between the 2-GPU and 4-GPU configs. Nothing in the code
# assumes world_size is a power of two: DistributedSampler splits the dataset
# into 3 shards, dist_util.dev() assigns cuda:0/1/2 via rank % GPUS_PER_NODE,
# and the MPI-aware logger averages metrics across any number of ranks.
#
# Prerequisites:
#   1) prepare_wikipedia_corpus.py has produced datasets/roots_en_wikipedia/wiki_{train,valid}.json
#   2) hse_supercomputer/tokenize_wiki_corpus_bert_uncased.sh has produced
#      datasets/roots_en_wikipedia/bert_uncased_seq64/ (Arrow DatasetDict)
#
# Without the pre-tokenized Arrow dir, each DDP rank would load ~100 GB of
# Python objects at startup and OOM any HSE V100 node above 1 GPU.
#
# Hyperparameters, scaled from the 1-GPU baseline (batch 256, lr 2e-4, 500K steps):
#   - per-GPU batch_size: 256  -> global batch 768 (3x)
#   - lr:                2e-4  -> 3.5e-4        (sqrt(3) scaling; linear was unstable at 4 GPUs)
#   - lr_anneal_steps:   500K  -> 166667        (same 128M samples seen: 166667 * 768 ≈ 128M)
#   - lr_warmup_steps:   0     -> 2000          (AdamW second-moment warmup)
#   - save_interval:     50K   -> 16667         (same samples between checkpoints)
#
# Sanity check after launch: watch for HEALTHY quartile-loss separation in the
# first ~2000 steps, e.g.
#       loss_q0=0.05, loss_q1=0.08, loss_q2=0.11, loss_q3=0.13
# If you see loss_q0 ≈ loss_q1 ≈ loss_q2 ≈ loss_q3, the model has collapsed
# to a constant and running longer will not recover. Kill the job and retry
# with lr=2e-4 (no scaling at all).

echo "========================================"
echo "Diffusion-LM Training: FULL Wikipedia + bert-base-uncased (random 128d e2e), 3 GPUs"
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

export OPENAI_LOGDIR=diffusion_models/diff_wiki_pad_rand128_transformer_lr0.00035_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased_bsz256x3gpu_wu2000_50h

WIKI_TOKENIZED_DIR=../../datasets/roots_en_wikipedia/bert_uncased_seq64

cd improved-diffusion/scripts

if [ ! -d "${WIKI_TOKENIZED_DIR}" ]; then
    echo "ERROR: pre-tokenized wiki dataset not found at ${WIKI_TOKENIZED_DIR}"
    echo "Run hse_supercomputer/tokenize_wiki_corpus_bert_uncased.sh first."
    exit 1
fi

echo ""
echo "Starting 3-GPU training..."
echo "Pre-tokenized wiki:  ${WIKI_TOKENIZED_DIR}"
echo "Output directory:    ${OPENAI_LOGDIR}"

srun --mpi=pmix python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality wiki \
    --save_interval 16667 \
    --lr 0.00035 \
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
    --lr_anneal_steps 166667 \
    --lr_warmup_steps 2000 \
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
echo "Training completed at: $(date)"
echo "========================================"
