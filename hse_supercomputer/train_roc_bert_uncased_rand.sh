#!/bin/bash
#SBATCH --job-name=diffusion_roc_bert_uncased_rand_s69
#SBATCH --partition=normal
#SBATCH --output=logs/train_roc_bert_uncased_rand_s69_%j.out
#SBATCH --error=logs/train_roc_bert_uncased_rand_s69_%j.err
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Training: Diffusion-LM on ROCStories with bert-base-uncased tokenizer + random 128d embeddings (e2e)
# Same architecture as spacy 128d model but with BERT WordPiece tokenizer (30522 vocab)
# Seed 69 for embedding geometry checks
echo "========================================"
echo "Diffusion-LM Training: ROCStories + bert-base-uncased tokenizer (random 128d e2e) seed 69"
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
echo "WandB mode: $WANDB_MODE"

export OPENAI_LOGDIR=diffusion_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd69_bert_uncased

cd improved-diffusion/scripts

echo ""
echo "Starting training..."
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
    --seed 69 \
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
