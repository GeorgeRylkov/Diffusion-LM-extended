#!/bin/bash
#SBATCH --job-name=diffusion_roc_bert_tiny_frozen_v2
#SBATCH --partition=normal
#SBATCH --output=logs/train_roc_bert_tiny_frozen_v2_%j.out
#SBATCH --error=logs/train_roc_bert_tiny_frozen_v2_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Diffusion-LM Training: ROCStories + frozen bert-tiny (128d) embeddings (v2: fixed corpus-weighted normalization)"
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

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "WandB mode: $WANDB_MODE"

export OPENAI_LOGDIR=diffusion_models/diff_roc_pad_bert128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen_v2

cd improved-diffusion/scripts

echo ""
echo "Starting ROCStories + frozen bert-tiny (128d) training (v2: corpus-weighted norm)..."
python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality roc \
    --save_interval 50000 \
    --lr 0.0001 \
    --batch_size 64 \
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
    --lr_anneal_steps 400000 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode emb \
    --vocab_size 30522 \
    --cache_mode no \
    --use_bert_tokenizer yes \
    --roc_train ../../datasets/ROCstory

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
