#!/bin/bash
#SBATCH --job-name="diffusion_test"
#SBATCH --partition=test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Test job for Diffusion-LM environment on HSE Supercomputer
# This job will verify that everything is set up correctly

echo "========================================"
echo "Diffusion-LM Test Job"
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

# Run verification script
echo ""
echo "Running verification script..."
python -u hse_supercomputer/verify_setup.py

echo ""
echo "========================================"
echo "Test completed at: $(date)"
echo "========================================"
