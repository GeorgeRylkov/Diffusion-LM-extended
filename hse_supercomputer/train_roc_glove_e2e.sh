#!/bin/bash
#SBATCH --job-name=diffusion_roc_glove_e2e
#SBATCH --partition=normal
#SBATCH --output=logs/train_roc_glove_e2e_%j.out
#SBATCH --error=logs/train_roc_glove_e2e_%j.err
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Training job for Diffusion-LM on ROCStories with GloVe embeddings (e2e mode)
# GloVe provides initial embeddings; they are then trained jointly with the model.
# HSE Supercomputer

echo "========================================"
echo "Diffusion-LM Training: ROCStories + GloVe (e2e)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Purge all modules and load PyTorch module
module purge
module load Python/PyTorch_GPU_v2.4 openmpi

# Activate venv
source hse_supercomputer/.venv_hse/bin/activate

# Show environment info
echo ""
echo "Environment information:"
which python
python -V
nvidia-smi

# Configure WandB for OFFLINE mode (required on HSE - no internet on compute nodes)
export WANDB_MODE=offline
echo "WandB mode: $WANDB_MODE"

# Checkpoint path follows the run_train.py naming convention:
# diff_{modality}_{padding_mode}_{exp_m}{in_channel}_{model_arch}_lr{lr}_{weight_decay}_{diff_steps}_{noise_schedule}_{loss_type}_h{hidden_size}_s{num_res_blocks}_d{dropout}_sd{seed}
export OPENAI_LOGDIR=diffusion_models/diff_roc_pad_glo128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_e2e
export TOKENIZERS_PARALLELISM=false

# Change to project directory
cd improved-diffusion/scripts

echo ""
echo "Starting ROCStories + GloVe (e2e) training..."
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
    --experiment glove \
    --lr_anneal_steps 400000 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode e2e \
    --vocab_size 11042 \
    --cache_mode no \
    --roc_train ../../datasets/ROCstory \
    --glove_file_path ../../predictability/glove/glove_rocstory_128d.txt \
    --vocab_file vocab/vocab.json

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
