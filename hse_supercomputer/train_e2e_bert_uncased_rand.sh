#!/bin/bash
#SBATCH --job-name=train_e2e_bert_uncased_rand
#SBATCH --partition=normal
#SBATCH --output=logs/train_e2e_bert_uncased_rand_%j.out
#SBATCH --error=logs/train_e2e_bert_uncased_rand_%j.err
#SBATCH --time=25:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Training: Diffusion-LM on E2E-tgt with bert-base-uncased tokenizer + random 128d embeddings (e2e)
# Same architecture as spacy 128d E2E model but with BERT WordPiece tokenizer (30522 vocab)

echo "========================================"
echo "Diffusion-LM Training: E2E-tgt + bert-base-uncased tokenizer (random 128d e2e)"
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

export OPENAI_LOGDIR=diffusion_models/diff_e2e-tgt_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased

cd improved-diffusion/scripts

echo ""
echo "Starting training..."
python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality e2e-tgt \
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
    --experiment random \
    --lr_anneal_steps 200000 \
    --weight_decay 0.0 \
    --num_res_blocks 2 \
    --predict_xstart True \
    --training_mode e2e \
    --vocab_size 30522 \
    --cache_mode no \
    --use_bert_tokenizer yes \
    --e2e_train ../../datasets/e2e_data

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
