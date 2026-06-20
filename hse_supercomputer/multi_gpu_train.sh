#!/bin/bash
#SBATCH --job-name=diffusion_multigpu
#SBATCH --partition=normal
#SBATCH --output=logs/train_multigpu_%j.out
#SBATCH --error=logs/train_multigpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4

# Multi-GPU training job for Diffusion-LM on HSE Supercomputer
# This uses MPI for distributed training across multiple GPUs

echo "========================================"
echo "Diffusion-LM Multi-GPU Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "========================================"

# Purge all modules and load required ones
module purge
module load Python/PyTorch_GPU_v2.4 openmpi

# Activate venv
source hse_supercomputer/.venv_hse/bin/activate

# Show environment info
echo ""
echo "Environment information:"
which python
python -V
which mpirun
nvidia-smi

# Configure WandB for OFFLINE mode (required on HSE - no internet on compute nodes)
# Metrics will be saved locally and synced later from login server
export WANDB_MODE=offline
echo "WandB mode: $WANDB_MODE"

# Set additional environment variables
export OPENAI_LOGDIR=diffusion_models/diff_e2e-tgt_pad_rand64_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_e2e_emb_training
export TOKENIZERS_PARALLELISM=false
export GPUS_PER_NODE=4

# Change to project directory
cd improved-diffusion/scripts

# Run training with MPI
# NOTE: Using python -u for unbuffered output
# NOTE: OMPI_ALLOW_RUN_AS_ROOT not needed - running as regular user on HSE
echo ""
echo "Starting multi-GPU training with MPI (4 GPUs)..."

srun python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality e2e-tgt \
    --save_interval 20000 \
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
    --in_channel 64 \
    --out_channel 64 \
    --padding_mode pad \
    --experiment random \
    --lr_anneal_steps 200000 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode emb \
    --vocab_size 821 \
    --e2e_train ../../datasets/e2e_data

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
