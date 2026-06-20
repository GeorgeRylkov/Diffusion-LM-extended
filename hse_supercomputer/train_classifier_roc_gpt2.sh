#!/bin/bash
#SBATCH --job-name=classifier_roc_gpt2
#SBATCH --partition=normal
#SBATCH --output=logs/classifier_roc_gpt2_%j.out
#SBATCH --error=logs/classifier_roc_gpt2_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Classifier training for ROCStories with GPT2 tokenizer (128d) diffusion model

echo "========================================"
echo "Classifier Training: ROCStories + GPT2 tokenizer"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_classifier/bin/activate

echo ""
echo "Environment information:"
which python
python -V
nvidia-smi

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

DIFFUSION_MODEL_PATH=improved-diffusion/scripts/diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_gpt2

echo ""
echo "Diffusion model path: ${DIFFUSION_MODEL_PATH}"
echo "Starting classifier training..."
python -u train_run.py \
    --experiment roc \
    --pretrained_model gpt2 \
    --model_type gpt2 \
    --task wp \
    --seed 101 \
    --epoch 15 \
    --bsz 20 \
    --notes gpt2 \
    --submit no \
    --app "--roc_train datasets/ROCstory --diffusion_model_path ${DIFFUSION_MODEL_PATH} --use_gpt2_tokenizer yes"

echo ""
echo "========================================"
echo "Classifier training completed at: $(date)"
echo "========================================"
