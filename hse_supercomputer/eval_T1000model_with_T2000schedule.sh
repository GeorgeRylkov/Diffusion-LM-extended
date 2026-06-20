#!/bin/bash
#SBATCH --job-name=T1000_eval2000
#SBATCH --partition=normal
#SBATCH --output=logs/T1000_eval2000_%j.out
#SBATCH --error=logs/T1000_eval2000_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

echo "========================================"
echo "Cross-eval: T=1000 trained model with T=2000 eval schedule"
echo "If the peak appears -> it's the schedule, not the training"
echo "If no peak -> it's baked into T=2000 training"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

module purge
module load Python/PyTorch_GPU_v2.4 openmpi

source hse_supercomputer/.venv_hse/bin/activate

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd improved-diffusion

MODEL_DIR=scripts/diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_1000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased

for STEP in 050000 100000; do
    CKPT=${MODEL_DIR}/ema_0.9999_${STEP}.pt

    if [ ! -f "${CKPT}" ]; then
        echo "Checkpoint ${CKPT} not found, skipping."
        continue
    fi

    echo ""
    echo "=================================================="
    echo "Checkpoint: ${STEP} steps — eval with T=2000 schedule"
    echo "=================================================="
    python -u scripts/eval_reconstruction.py \
        --model_path "${CKPT}" \
        --batch_size 64 \
        --num_batches 8 \
        --num_timesteps 2000 \
        --override_diffusion_steps 2000 \
        --roc_train ../datasets/ROCstory
done

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
