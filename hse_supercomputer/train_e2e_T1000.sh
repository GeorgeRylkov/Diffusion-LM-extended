#!/bin/bash
#SBATCH --job-name=train_e2e_T1000
#SBATCH --partition=normal
#SBATCH --output=logs/train_e2e_T1000_%j.out
#SBATCH --error=logs/train_e2e_T1000_%j.err
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Training: Same as e2e 128d bert-uncased but with diffusion_steps=1000 (instead of 2000)
# Goal: check if the t=T/2 peak appears at t=500 with T=1000
# Plan: stop early after 50k steps and evaluate

echo "========================================"
echo "Diffusion-LM Training: e2e 128d, T=1000 sqrt schedule"
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

export OPENAI_LOGDIR=diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_1000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased

cd improved-diffusion/scripts

echo ""
echo "Starting training (diffusion_steps=1000)..."
python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality roc \
    --save_interval 50000 \
    --lr 0.0001 \
    --batch_size 64 \
    --diffusion_steps 1000 \
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
    --lr_anneal_steps 400000 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode e2e \
    --vocab_size 30522 \
    --cache_mode no \
    --use_bert_tokenizer yes \
    --roc_train ../../datasets/ROCstory

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
