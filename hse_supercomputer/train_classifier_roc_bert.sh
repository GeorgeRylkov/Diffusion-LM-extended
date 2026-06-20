#!/bin/bash
#SBATCH --job-name=classifier_roc_bert
#SBATCH --partition=normal
#SBATCH --output=logs/classifier_roc_bert_%j.out
#SBATCH --error=logs/classifier_roc_bert_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Classifier training for ROCStories with BERT tokenizer diffusion model
# Serves all BERT experiments (random + frozen)
# HSE Supercomputer

echo "========================================"
echo "Classifier Training: ROCStories + BERT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Purge all modules and load PyTorch module
module purge
module load Python/PyTorch_GPU_v2.4 openmpi

# Activate classifier-specific venv (has stanza, benepar, nltk, etc.)
source hse_supercomputer/.venv_classifier/bin/activate

# Show environment info
echo ""
echo "Environment information:"
which python
python -V
nvidia-smi

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# Points to the BERT random model dir (has vocab.txt/tokenizer.json for BERT tokenizer)
# This classifier serves both BERT random and BERT frozen experiments
DIFFUSION_MODEL_PATH=improved-diffusion/scripts/diffusion_models/diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_v2

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
    --notes bert_v2 \
    --submit no \
    --app "--roc_train datasets/ROCstory --diffusion_model_path ${DIFFUSION_MODEL_PATH} --use_bert_tokenizer yes"

echo ""
echo "========================================"
echo "Classifier training completed at: $(date)"
echo "========================================"
