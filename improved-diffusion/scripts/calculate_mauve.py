"""Compute MAUVE score between generated and reference texts."""

import argparse
import glob
import json
import os
import random
import re

import numpy as np
import torch
import mauve


def _debug_bert_cache_files(model_name):
    """Print per-file cache presence for BERT tokenizer assets."""
    required = [
        'config.json',
        'tokenizer_config.json',
        'tokenizer.json',
        'vocab.txt',
        'special_tokens_map.json',
    ]
    print('\nBERT cache debug:')
    print(f'  model: {model_name}')
    print(f'  HF_HOME={os.environ.get("HF_HOME", "")}')
    print(f'  HUGGINGFACE_HUB_CACHE={os.environ.get("HUGGINGFACE_HUB_CACHE", "")}')
    print(f'  TRANSFORMERS_CACHE={os.environ.get("TRANSFORMERS_CACHE", "")}')
    print(f'  HF_HUB_OFFLINE={os.environ.get("HF_HUB_OFFLINE", "")}')

    # Case 1: explicit local directory path passed to --bert_preprocess.
    if os.path.isdir(model_name):
        missing = []
        print(f'  local_dir={model_name}')
        for fname in required:
            fpath = os.path.join(model_name, fname)
            ok = os.path.isfile(fpath)
            print(f'    {"FOUND " if ok else "MISSING"} {fname} ({fpath})')
            if not ok:
                missing.append(fname)
        if missing:
            print(f'  Missing local files: {missing}')
        return

    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    hub_cache = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.join(hf_home, 'hub'))
    tf_cache = os.environ.get('TRANSFORMERS_CACHE', os.path.join(hf_home, 'transformers'))

    # Case 2: new huggingface_hub cache layout.
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_root = os.path.join(hub_cache, model_cache_name)
    snapshot_dir = None
    ref_main = os.path.join(model_root, 'refs', 'main')
    if os.path.isfile(ref_main):
        with open(ref_main, 'r') as f:
            rev = f.read().strip()
        snapshot_dir = os.path.join(model_root, 'snapshots', rev)
    elif os.path.isdir(os.path.join(model_root, 'snapshots')):
        snaps = sorted(glob.glob(os.path.join(model_root, 'snapshots', '*')))
        if snaps:
            snapshot_dir = snaps[-1]

    missing_hub = []
    print(f'  hub_snapshot={snapshot_dir or "<not found>"}')
    for fname in required:
        ok = bool(snapshot_dir) and os.path.isfile(os.path.join(snapshot_dir, fname))
        print(f'    {"FOUND " if ok else "MISSING"} {fname} (hub)')
        if not ok:
            missing_hub.append(fname)
    if missing_hub:
        print(f'  Missing in hub snapshot: {missing_hub}')

    # Case 3: old transformers cache layout (hashed files + *.json metadata).
    meta_map = {}
    if os.path.isdir(tf_cache):
        for meta_path in glob.glob(os.path.join(tf_cache, '*.json')):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                url = meta.get('url', '')
                fname = url.rsplit('/', 1)[-1] if url else None
                if fname:
                    data_path = meta_path[:-5]  # strip ".json"
                    meta_map[fname] = os.path.isfile(data_path)
            except Exception:
                continue
    print(f'  transformers_cache={tf_cache}')
    missing_tf = []
    for fname in required:
        ok = meta_map.get(fname, False)
        print(f'    {"FOUND " if ok else "MISSING"} {fname} (transformers-cache)')
        if not ok:
            missing_tf.append(fname)
    if missing_tf:
        print(f'  Missing in transformers cache: {missing_tf}')


def load_texts(filepath):
    texts = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if filepath.endswith('.json'):
                texts.append(json.loads(line)[0])
            else:
                texts.append(line)
    return texts


def clean_generated_text(text, tokens_only=False):
    """Strip special/padding tokens, optionally fix tokenization artifacts."""
    for tok in ['[CLS]', '[SEP]', '[PAD]', 'START', 'END', 'PAD', '<|endoftext|>']:
        text = text.replace(tok, '')
    if not tokens_only:
        text = re.sub(r'\s+([.,!?;:\'\")])', r'\1', text)
        text = re.sub(r"(\w)\s+(n't|'s|'re|'ve|'ll|'d|'m)", r"\1\2", text)
    text = ' '.join(text.split())
    return text.strip()


_spacy_tokenizer = None

def _get_spacy_tokenizer():
    global _spacy_tokenizer
    if _spacy_tokenizer is None:
        from spacy.lang.en import English
        _spacy_tokenizer = English().tokenizer
    return _spacy_tokenizer


def preprocess_reference_with_vocab(text, vocab_dict):
    """Apply the same tokenization+vocab pipeline used during training.

    1. Spacy-tokenize the raw text (matching get_corpus_rocstory)
    2. Replace OOV words with 'UNK'
    3. Join tokens with spaces (matching the model's output format)
    """
    tokenizer = _get_spacy_tokenizer()
    tokens = [t.text for t in tokenizer(text)]
    processed = [w if w in vocab_dict else 'UNK' for w in tokens]
    return ' '.join(processed)


def preprocess_reference_with_bert(text, bert_tokenizer):
    """Run reference through BERT encode→decode to match generated text space.

    For uncased models this lowercases; also normalizes punctuation/spacing
    to match what tokenizer.decode() produces at inference time.
    """
    ids = bert_tokenizer.encode(text.strip(), add_special_tokens=False)
    return bert_tokenizer.decode(ids)


def preprocess_reference_with_gpt2(text, gpt2_tokenizer):
    """Run reference through GPT2 encode→decode to match generated text space.

    Normalizes whitespace and special characters to match what
    tokenizer.decode() produces at inference time.
    """
    ids = gpt2_tokenizer.encode(text.strip())
    return gpt2_tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser(description='MAUVE evaluation')
    parser.add_argument('--generated', type=str, required=True,
                        help='Path to generated samples file (.txt or .json)')
    parser.add_argument('--references', type=str, required=True,
                        help='Path to reference texts file (.txt or .json)')
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='Path to vocab.json. When provided, references are '
                             'preprocessed with the same tokenization+vocab pipeline '
                             'used during training (OOV words become UNK)')
    parser.add_argument('--tokens_only', action='store_true',
                        help='Only remove special tokens, skip punctuation detokenization')
    parser.add_argument('--no_clean', action='store_true',
                        help='Skip all cleaning of generated texts (use as-is)')
    parser.add_argument('--bert_preprocess', type=str, default=None,
                        help='BERT model name (e.g. bert-base-uncased) to preprocess '
                             'references through encode→decode, matching inference output')
    parser.add_argument('--gpt2_preprocess', type=str, default=None,
                        help='GPT2 model name (e.g. gpt2) to preprocess '
                             'references through encode→decode, matching inference output')
    parser.add_argument('--n_refs', type=int, default=1000,
                        help='Number of references to sample from the reference file')
    parser.add_argument('--mauve_model', type=str, default='gpt2-large',
                        help='Model for MAUVE featurization')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    vocab_dict = None
    if args.vocab_path:
        with open(args.vocab_path, 'r') as f:
            vocab_dict = json.load(f)
        print(f'Loaded vocab with {len(vocab_dict)} entries from {args.vocab_path}')

    generated = load_texts(args.generated)
    if args.no_clean:
        generated = [t for t in generated if t.strip()]
    else:
        generated = [clean_generated_text(t, tokens_only=args.tokens_only) for t in generated]
        generated = [t for t in generated if t]

    clean_dir = os.path.join(os.path.dirname(args.generated), 'cleaned_texts_mauve')
    os.makedirs(clean_dir, exist_ok=True)
    clean_path = os.path.join(clean_dir, os.path.basename(args.generated))
    with open(clean_path, 'w') as f:
        for t in generated:
            print(t, file=f)
    print(f'Saved {len(generated)} cleaned texts to {clean_path}')

    references = load_texts(args.references)
    references = [t for t in references if t.strip()]

    if len(references) > args.n_refs:
        references = random.sample(references, args.n_refs)

    if vocab_dict:
        print(f'\nPreprocessing {len(references)} references with spacy + vocab (OOV -> UNK)...')
        references = [preprocess_reference_with_vocab(t, vocab_dict) for t in references]
        ref_unk_count = sum(1 for r in references if 'UNK' in r)
        total_words = sum(len(r.split()) for r in references)
        total_unks = sum(r.split().count('UNK') for r in references)
        print(f'  References with UNK: {ref_unk_count}/{len(references)} '
              f'({100*ref_unk_count/len(references):.1f}%)')
        print(f'  Total UNK tokens: {total_unks}/{total_words} '
              f'({100*total_unks/total_words:.1f}%)')

        ref_basename = os.path.splitext(os.path.basename(args.references))[0]
        ref_clean_path = os.path.join(clean_dir, f'{ref_basename}_preprocessed.txt')
        with open(ref_clean_path, 'w') as f:
            for r in references:
                print(r, file=f)
        print(f'  Saved preprocessed references to {ref_clean_path}')

    if args.bert_preprocess:
        from transformers import AutoTokenizer
        try:
            bert_tok = AutoTokenizer.from_pretrained(args.bert_preprocess)
        except Exception:
            _debug_bert_cache_files(args.bert_preprocess)
            raise
        print(f'\nPreprocessing {len(references)} references with {args.bert_preprocess} encode→decode...')
        references = [preprocess_reference_with_bert(t, bert_tok) for t in references]

        ref_basename = os.path.splitext(os.path.basename(args.references))[0]
        ref_clean_path = os.path.join(clean_dir, f'{ref_basename}_{args.bert_preprocess}_preprocessed.txt')
        with open(ref_clean_path, 'w') as f:
            for r in references:
                print(r, file=f)
        print(f'  Saved preprocessed references to {ref_clean_path}')

    if args.gpt2_preprocess:
        from transformers import AutoTokenizer
        gpt2_tok = AutoTokenizer.from_pretrained(args.gpt2_preprocess)
        print(f'\nPreprocessing {len(references)} references with {args.gpt2_preprocess} encode→decode...')
        references = [preprocess_reference_with_gpt2(t, gpt2_tok) for t in references]

        ref_basename = os.path.splitext(os.path.basename(args.references))[0]
        ref_clean_path = os.path.join(clean_dir, f'{ref_basename}_{args.gpt2_preprocess}_preprocessed.txt')
        with open(ref_clean_path, 'w') as f:
            for r in references:
                print(r, file=f)
        print(f'  Saved preprocessed references to {ref_clean_path}')

    print(f'\nGenerated: {len(generated)} texts')
    print(f'References: {len(references)} texts')
    print(f'\nExample generated: {generated[0]}')
    print(f'Example reference: {references[0]}\n')

    torch.cuda.empty_cache()
    results = mauve.compute_mauve(
        p_text=references,
        q_text=generated,
        featurize_model_name=args.mauve_model,
        device_id=0,
        verbose=False,
        seed=args.seed,
    )

    print(f'MAUVE: {results.mauve:.4f}')
    print(f'Frontier integral: {results.frontier_integral:.4f}')


if __name__ == '__main__':
    main()
