#!/bin/bash
#SBATCH --job-name=clf_wiki_partial_bert
#SBATCH --partition=normal
#SBATCH --output=logs/classifier_wiki_partial_bert_uncased_%j.out
#SBATCH --error=logs/classifier_wiki_partial_bert_uncased_%j.err
#SBATCH --time=04:00:00
# ~3.5h train + buffer; ~60k steps after checkpoint-100000 at ~36min/10k steps
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

# Classifier for PARTIAL Wikipedia (~50k articles / ~500k passages) + bert-base-uncased.
# Matches train_wiki_partial_bert_uncased_rand.sh: same diffusion checkpoint dir and
# wiki_corpus_train (datasets/roots_en_wikipedia) for a consistent AR eval model.
#
# Resume: continues from checkpoint-100000 toward max_steps=160000 (~5 epoch token budget
# vs ROC 20-epoch style). Saves/evals every 10k; keeps 3 checkpoints + best eval_loss.
# --epoch 15 preserves the original output_dir name (wiki_e=15_...); stopping is max_steps.
#
# For full-wiki diffusion instead, point DIFFUSION_MODEL_PATH at e.g.
#   improved-diffusion/scripts/diffusion_models/diff_wiki_pad_rand128_..._bert_uncased
# and set --wiki_corpus_train to the full corpus tree.

echo "========================================"
echo "Classifier Training: Wikipedia PARTIAL + bert-base-uncased (matches partial diffusion)"
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

DIFFUSION_MODEL_PATH=improved-diffusion/scripts/diffusion_models/diff_wiki_partial_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased
RESUME_CKPT=classifier_models/wiki_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_wp_bert_uncased_wiki_partial/checkpoint-100000

echo ""
echo "Diffusion model path: ${DIFFUSION_MODEL_PATH}"
echo "Resume checkpoint: ${RESUME_CKPT}"
echo "Starting classifier training (resume → max 160k steps)..."
python -u train_run.py \
    --experiment wiki \
    --pretrained_model gpt2 \
    --model_type gpt2 \
    --task wp \
    --seed 101 \
    --epoch 15 \
    --bsz 20 \
    --notes bert_uncased_wiki_partial \
    --submit no \
    --max_steps 160000 \
    --resume_from_checkpoint "${RESUME_CKPT}" \
    --no_overwrite_output_dir \
    --clm_save_steps 10000 \
    --clm_eval_steps 10000 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --app "--wiki_corpus_train datasets/roots_en_wikipedia --diffusion_model_path ${DIFFUSION_MODEL_PATH} --use_bert_tokenizer yes"

echo ""
echo "========================================"
echo "Classifier training completed at: $(date)"
echo "========================================"
