"""
Train FastText embeddings on the Spacy-tokenized ROCStory corpus.

FastText learns character n-gram embeddings, so it can produce vectors for
any word — even OOV words — by composing n-gram vectors. This eliminates
the need for UNK tokens entirely.

Uses gensim's FastText implementation (no C++ build required).

Usage:
    python train_fasttext_rocstory.py [--dim 128] [--corpus datasets/ROCstory/roc_train_corpus.txt]

Requirements:
    pip install gensim
"""

import argparse
import os
import numpy as np
from gensim.models import FastText
from gensim.models.word2vec import LineSentence


def train_fasttext(corpus_path, output_dir, vector_size=128, epochs=25,
                   min_count=1, window_size=10, lr=0.05, minn=3, maxn=6):
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*70}")
    print("FastText Training Configuration")
    print(f"{'='*70}")
    print(f"  Corpus:          {corpus_path}")
    print(f"  Output:          {output_dir}")
    print(f"  Dimension:       {vector_size}")
    print(f"  Epochs:          {epochs}")
    print(f"  Min count:       {min_count}")
    print(f"  Window size:     {window_size}")
    print(f"  Learning rate:   {lr}")
    print(f"  Char n-gram:     {minn}-{maxn}")
    print(f"  Workers:         {os.cpu_count() or 4}")
    print(f"{'='*70}\n")

    sentences = LineSentence(corpus_path)

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window_size,
        min_count=min_count,
        sg=1,  # skipgram
        epochs=epochs,
        alpha=lr,
        min_alpha=lr / epochs,
        min_n=minn,
        max_n=maxn,
        workers=os.cpu_count() or 4,
    )

    # Save gensim model
    model_path = os.path.join(output_dir, "fasttext_rocstory.model")
    model.save(model_path)
    print(f"\nSaved gensim FastText model to: {model_path}")

    # Export to GloVe-compatible text format for Diffusion-LM
    txt_path = os.path.join(output_dir, f"fasttext_rocstory_{vector_size}d.txt")
    words = list(model.wv.key_to_index.keys())
    with open(txt_path, "w") as f:
        for word in words:
            vec = model.wv[word]
            vec_str = " ".join(f"{v:.6f}" for v in vec)
            f.write(f"{word} {vec_str}\n")

    print(f"Exported {len(words)} word vectors to: {txt_path}")

    # Statistics
    vecs = np.array([model.wv[w] for w in words])
    norms = np.linalg.norm(vecs, axis=-1)

    print(f"\n{'='*70}")
    print("Embedding Statistics")
    print(f"{'='*70}")
    print(f"  Vocabulary size: {len(words)}")
    print(f"  Dimension:       {vector_size}")
    print(f"  Mean norm:       {norms.mean():.4f}")
    print(f"  Std norm:        {norms.std():.4f}")
    print(f"  Min norm:        {norms.min():.4f}  ({words[norms.argmin()]})")
    print(f"  Max norm:        {norms.max():.4f}  ({words[norms.argmax()]})")
    print(f"  Relative std:    {norms.std() / norms.mean():.2%}")

    sorted_idx = np.argsort(norms)
    print(f"\n  Closest to origin (smallest norm):")
    for i in sorted_idx[:15]:
        print(f"    {words[i]:<20} norm={norms[i]:.4f}")

    print(f"\n  Farthest from origin (largest norm):")
    for i in sorted_idx[-15:][::-1]:
        print(f"    {words[i]:<20} norm={norms[i]:.4f}")

    # Test OOV word composition
    print(f"\n{'='*70}")
    print("OOV Word Handling (n-gram composition)")
    print(f"{'='*70}")
    test_oov = ["unforgettable", "misunderstanding", "cryptocurrency", "XYZ123"]
    for word in test_oov:
        vec = model.wv.get_vector(word)
        norm = np.linalg.norm(vec)
        in_vocab = word in model.wv
        print(f"  {word:<25} norm={norm:.4f}  in_vocab={in_vocab}")

    print(f"\n{'='*70}")
    print("Integration with Diffusion-LM")
    print(f"{'='*70}")
    print(f"\nThe text file is GloVe-compatible:")
    print(f"  {txt_path}")
    print(f"\nUse as: --glove_file_path {txt_path}")
    print(f"{'='*70}\n")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train FastText embeddings on ROCStory corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", default="datasets/ROCstory/roc_train_corpus.txt",
                        help="Path to Spacy-tokenized corpus")
    parser.add_argument("--output-dir", default="predictability/fasttext",
                        help="Output directory")
    parser.add_argument("--dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Minimum word frequency (1 = keep all)")
    parser.add_argument("--window", type=int, default=10,
                        help="Context window size")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate")
    parser.add_argument("--minn", type=int, default=3,
                        help="Min character n-gram length")
    parser.add_argument("--maxn", type=int, default=6,
                        help="Max character n-gram length")

    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"ERROR: Corpus not found: {args.corpus}")
        print("Run: python prepare_rocstory_corpus.py")
        raise SystemExit(1)

    train_fasttext(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        vector_size=args.dim,
        epochs=args.epochs,
        min_count=args.min_count,
        window_size=args.window,
        lr=args.lr,
        minn=args.minn,
        maxn=args.maxn,
    )


if __name__ == "__main__":
    main()
