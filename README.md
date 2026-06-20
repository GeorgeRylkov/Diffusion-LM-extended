# Training an Encoder for Latent Space Representation in Continuous Text Diffusion Models

Master's thesis, HSE University, 2025–2026.

This repository extends [Diffusion-LM](https://arxiv.org/pdf/2205.14217.pdf)
with experiments on pretrained embedding spaces and larger corpora.

## Extensions

**Embedding modes.** Beyond the original random/GloVe embeddings, we add:

- BERT-tiny 128d — frozen pretrained and jointly learned (e2e) variants
- BERT-base 768d — frozen and e2e (limited by noise schedule mismatch at high dimension)
- GPT-2 PCA 128d — GPT-2 `wte` embeddings reduced via PCA, frozen during training
- fastText 128d — subword-aware embeddings, handles out-of-vocabulary words

All embedding variants are compared under matched tokenizer and dimension,
pairing a frozen pretrained model with a jointly trained one.

**Wikipedia corpus.** Training and evaluation on ~27M passages extracted from
the ROOTS English Wikipedia dataset, in addition to the original E2E and ROCStories corpora.

**Latent space analysis.** Geometric comparison of frozen vs. e2e embedding spaces:
- Local Neighborhood Similarity (LNS, Boggust et al. 2022) — per-token Jaccard overlap of k-NN sets
- Hubness (Radovanovic et al. 2010) — skewness of the neighbour-count distribution
- Isotropy — mean pairwise cosine similarity, PCA effective rank
- PIP distance — Frobenius norm of the difference of Gram matrices

**Infrastructure.** Training scripts for the HSE supercomputer (SLURM + 3× V100).

## Main Finding

Among all tested configurations, **frozen pretrained BERT-tiny embeddings achieve
the best generation quality** (ROCStories: PPL 3.04, MAUVE 0.503), outperforming
their jointly learned counterpart (PPL 3.10, MAUVE 0.194). This holds on Wikipedia as well.
The result contradicts prior work reporting an advantage for jointly learned embeddings.

Geometry analysis shows that jointly learned spaces develop higher hubness and
a more uniform PCA spectrum, but their local neighborhoods diverge substantially
from the pretrained structure — which may explain the quality gap.

---

# Diffusion-LM Improves Controllable Text Generation

https://arxiv.org/pdf/2205.14217.pdf 



-----------------------------------------------------
## Conda Setup:
```python 
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb
```

-----------------------------------------------------
## Train Diffusion-LM:

```cd improved-diffusion; mkdir diffusion_models;```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 200000  --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data " --notes xstart_e2e```

```python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 400000  --seed 101 --noise_schedule sqrt  --in_channel 128 --modality roc --submit no --padding_mode pad  --app "--predict_xstart True --training_mode e2e  --vocab_size 11043  --roc_train ../datasets/ROCstory " --notes xstart_e2e --bsz 64```


-------------------
## Decode Diffusion-LM:
mkdir generation_outputs 

``python scripts/batch_decode.py {path-to-diffusion-lm} -1.0 ema``


------------------- 
## Controllable Text Generation 
First, train the classsifier used to guide the generation (e.g. a syntactic parser) 

``  
python train_run.py --experiment e2e-tgt-tree  --app "--init_emb {path-to-diffusion-lm} --n_embd {16} --learned_emb yes " --pretrained_model bert-base-uncased --epoch 6 --bsz 10
``

Then, we can use the trained classifier to guide generation. 
(currently, need to update the classifier directory in scripts/infill.py. I will clean this up in the next release.)

``python 
python scripts/infill.py --model_path {path-to-diffusion-lm} --eval_task_ 'control_tree' --use_ddim True  --notes "tree_adagrad" --eta 1. --verbose pipe``



-----------------------------------------------------

For details of the methods and results, please refer to our paper. 


```bibtex
@article{Li-2022-DiffusionLM,
  title={Diffusion-LM Improves Controllable Text Generation},
  author={Xiang Lisa Li and John Thickstun and Ishaan Gulrajani and Percy Liang and Tatsunori Hashimoto},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.14217}
}
```
