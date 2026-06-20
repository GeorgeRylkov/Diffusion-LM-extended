#!/bin/bash
#SBATCH --job-name=test_token_ids
#SBATCH --partition=normal
#SBATCH --output=logs/test_token_ids_%j.out
#SBATCH --error=logs/test_token_ids_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

module purge
module load Python/PyTorch_GPU_v2.4 openmpi
source hse_supercomputer/.venv_hse/bin/activate
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

cd improved-diffusion/scripts
mkdir -p ../generation_outputs/test_token_ids

MODEL_PATH=diffusion_models/diff_roc_pad_rand768_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert/ema_0.9999_400000.pt

python -u text_sample.py \
    --model_path "$MODEL_PATH" \
    --batch_size 50 \
    --num_samples 50 \
    --top_p -1.0 \
    --out_dir ../generation_outputs/test_token_ids \
    --log_token_ids ../generation_outputs/test_token_ids/token_ids_log.jsonl
