#!/bin/bash
#SBATCH --job-name=classifier_e2e_bert_uncased
#SBATCH --partition=normal
#SBATCH --output=logs/classifier_e2e_bert_uncased_%j.out
#SBATCH --error=logs/classifier_e2e_bert_uncased_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Classifier training for E2E-tgt with bert-base-uncased tokenizer (128d) diffusion model

echo "========================================"
echo "Classifier Training: E2E-tgt + bert-base-uncased"
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

DIFFUSION_MODEL_PATH=improved-diffusion/scripts/diffusion_models/diff_e2e-tgt_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased

echo ""
echo "Diffusion model path: ${DIFFUSION_MODEL_PATH}"
echo "Starting classifier training..."
python -u train_run.py \
    --experiment e2e-tgt \
    --pretrained_model gpt2 \
    --model_type gpt2 \
    --task wp \
    --seed 101 \
    --epoch 15 \
    --bsz 20 \
    --notes bert_uncased \
    --submit no \
    --app "--e2e_train datasets/e2e_data --diffusion_model_path ${DIFFUSION_MODEL_PATH} --use_bert_tokenizer yes"

echo ""
echo "========================================"
echo "Classifier training completed at: $(date)"
echo "========================================"
