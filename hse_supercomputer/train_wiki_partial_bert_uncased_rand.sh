#!/bin/bash
#SBATCH --job-name=diffusion_wiki_partial_bert_uncased_rand
#SBATCH --partition=normal
#SBATCH --output=logs/train_wiki_partial_bert_uncased_rand_%j.out
#SBATCH --error=logs/train_wiki_partial_bert_uncased_rand_%j.err
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Training: Diffusion-LM on partial Wikipedia (~500K passages, 50K articles subsample)
# with bert-base-uncased tokenizer + random 128d embeddings (e2e).
# Mirrors train_roc_bert_uncased_rand.sh hyperparams but uses the Wikipedia modality.
# Trained jointly (not frozen). Its learned word_embedding provides the corpus norm
# reference for the wiki frozen experiment (see BERT_TARGET_CORPUS_NORMS['wiki']).
#
# Preprocessing used:
#   python prepare_wikipedia_corpus.py --input_dir datasets/roots_en_wikipedia/data \
#       --output_dir datasets/roots_en_wikipedia --max_articles 50000

echo "========================================"
echo "Diffusion-LM Training: Wikipedia PARTIAL (~500K passages) + bert-base-uncased (random 128d e2e)"
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

export OPENAI_LOGDIR=diffusion_models/diff_wiki_partial_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased

cd improved-diffusion/scripts

echo ""
echo "Starting training..."
python -u train.py \
    --checkpoint_path ${OPENAI_LOGDIR} \
    --model_arch transformer \
    --modality wiki \
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
    --wiki_corpus_train ../../datasets/roots_en_wikipedia

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
