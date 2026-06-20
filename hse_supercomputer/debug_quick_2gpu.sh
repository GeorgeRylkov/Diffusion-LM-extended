#!/bin/bash
#SBATCH --job-name=debug_2gpu
#SBATCH --partition=normal
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2

# Quick debug job with 2 GPUs
# Uses MPI to run 1 process per GPU

echo "========================================"
echo "Quick Debug Run (2 GPUs)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Load environment
module purge
module load Python/PyTorch_GPU_v2.4 openmpi
source hse_supercomputer/.venv_hse/bin/activate

# Set environment variables
export WANDB_MODE=offline
export OPENAI_LOGDIR=debug_runs/test_2gpu_${SLURM_JOB_ID}
export TOKENIZERS_PARALLELISM=false

# Fix MPI communication on this cluster
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mpi_cuda_support=0

# Set GPUs per node so each rank picks the right GPU
export GPUS_PER_NODE=2

# Show environment info
echo ""
echo "Environment information:"
which python
python -V
nvidia-smi

# Create debug output directory
mkdir -p ${OPENAI_LOGDIR}

# Change to project directory
cd improved-diffusion/scripts

echo ""
echo "Starting quick debug run (2 GPUs, limited steps)..."
echo "Output directory: ${OPENAI_LOGDIR}"
echo ""

# Use srun with PMIx for Slurm-native MPI launch
srun --mpi=pmix python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality e2e-tgt \
    --save_interval 50 \
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
    --lr_anneal_steps 100 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode emb \
    --vocab_size 821 \
    --e2e_train ../../datasets/e2e_data

echo ""
echo "========================================"
echo "Debug run completed at: $(date)"
echo "Check output in: ${OPENAI_LOGDIR}"
echo "========================================"
