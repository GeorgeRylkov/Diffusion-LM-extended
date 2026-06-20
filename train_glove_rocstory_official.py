"""Train GloVe embeddings on ROCStories using the official Stanford binaries."""

import os
import sys
import subprocess
import argparse
import struct
from pathlib import Path


def check_glove_installation(glove_dir):
    """Check if GloVe binaries exist"""
    build_dir = os.path.join(glove_dir, 'build')
    required_bins = ['vocab_count', 'cooccur', 'shuffle', 'glove']
    
    for binary in required_bins:
        bin_path = os.path.join(build_dir, binary)
        if not os.path.exists(bin_path):
            print(f"ERROR: GloVe binary not found: {bin_path}")
            print("\nPlease:")
            print("  1. Clone: git clone https://github.com/stanfordnlp/glove")
            print("  2. Build: cd glove && make")
            print(f"  3. Set --glove-dir to the glove directory")
            return False
    return True


def run_command(cmd, description):
    """Run a shell command and display output"""
    print(f"\n{'='*70}")
    print(description)
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nERROR: Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    return result


def convert_binary_to_text(vocab_file, bin_file, txt_file, vector_size):
    """Convert GloVe binary format to text format"""
    print(f"\n{'='*70}")
    print("Converting binary embeddings to text format...")
    print(f"{'='*70}")
    
    # Read vocabulary
    vocab = []
    with open(vocab_file, 'r') as f:
        for line in f:
            word = line.strip().split()[0]
            vocab.append(word)
    
    print(f"Vocabulary size: {len(vocab):,}")
    
    # Read binary embeddings
    embeddings = []
    with open(bin_file, 'rb') as f:
        for i in range(len(vocab)):
            vec_bytes = f.read(8 * vector_size)
            if len(vec_bytes) != 8 * vector_size:
                print(f"Warning: Incomplete vector at position {i}")
                break
            vec = struct.unpack('d' * vector_size, vec_bytes)
            embeddings.append(vec)
    
    print(f"Read {len(embeddings):,} vectors")
    
    # Write text format
    with open(txt_file, 'w') as f:
        for word, embedding in zip(vocab, embeddings):
            vec_str = ' '.join(f'{v:.6f}' for v in embedding)
            f.write(f'{word} {vec_str}\n')
    
    print(f"Saved to: {txt_file}")
    print(f"File size: {os.path.getsize(txt_file) / 1024 / 1024:.2f} MB")


def train_glove_embeddings(
    glove_dir,
    corpus_file,
    output_dir,
    vector_size=50,
    window_size=10,
    min_count=3,
    max_iter=25,
    num_threads=8,
    memory=4.0,
    x_max=100,
    alpha=0.75,
    verbose=2
):
    """Train GloVe embeddings using official implementation"""
    
    # Setup paths
    build_dir = os.path.join(glove_dir, 'build')
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_file = os.path.join(output_dir, 'vocab.txt')
    cooccur_file = os.path.join(output_dir, 'cooccurrence.bin')
    cooccur_shuf_file = os.path.join(output_dir, 'cooccurrence.shuf.bin')
    save_file = os.path.join(output_dir, 'glove_rocstory')
    
    print(f"\n{'='*70}")
    print(f"GloVe Training Configuration")
    print(f"{'='*70}")
    print(f"GloVe directory:     {glove_dir}")
    print(f"Corpus file:         {corpus_file}")
    print(f"Output directory:    {output_dir}")
    print(f"Vector size:         {vector_size}")
    print(f"Window size:         {window_size}")
    print(f"Min word count:      {min_count}")
    print(f"Max iterations:      {max_iter}")
    print(f"Threads:             {num_threads}")
    print(f"Memory (GB):         {memory}")
    print(f"X_max:               {x_max}")
    print(f"Alpha:               {alpha}")
    print(f"{'='*70}")
    
    # Step 1: Build vocabulary
    cmd = [
        os.path.join(build_dir, 'vocab_count'),
        '-min-count', str(min_count),
        '-verbose', str(verbose)
    ]
    
    with open(corpus_file, 'r') as infile, open(vocab_file, 'w') as outfile:
        result = subprocess.run(cmd, stdin=infile, stdout=outfile, text=True)
        if result.returncode != 0:
            print("ERROR: vocab_count failed")
            sys.exit(1)
    
    # Count vocabulary size
    with open(vocab_file, 'r') as f:
        vocab_size_count = sum(1 for _ in f)
    
    print(f"\n✓ Step 1/4 Complete: Vocabulary built ({vocab_size_count:,} words)")
    
    # Step 2: Build co-occurrence matrix
    cmd = [
        os.path.join(build_dir, 'cooccur'),
        '-memory', str(memory),
        '-vocab-file', vocab_file,
        '-verbose', str(verbose),
        '-window-size', str(window_size)
    ]
    
    with open(corpus_file, 'r') as infile, open(cooccur_file, 'wb') as outfile:
        result = subprocess.run(cmd, stdin=infile, stdout=outfile)
        if result.returncode != 0:
            print("ERROR: cooccur failed")
            sys.exit(1)
    
    print(f"✓ Step 2/4 Complete: Co-occurrence matrix built")
    
    # Step 3: Shuffle co-occurrences
    cmd = [
        os.path.join(build_dir, 'shuffle'),
        '-memory', str(memory),
        '-verbose', str(verbose)
    ]
    
    with open(cooccur_file, 'rb') as infile, open(cooccur_shuf_file, 'wb') as outfile:
        result = subprocess.run(cmd, stdin=infile, stdout=outfile)
        if result.returncode != 0:
            print("ERROR: shuffle failed")
            sys.exit(1)
    
    print(f"✓ Step 3/4 Complete: Co-occurrences shuffled")
    
    # Step 4: Train GloVe
    cmd = [
        os.path.join(build_dir, 'glove'),
        '-save-file', save_file,
        '-threads', str(num_threads),
        '-input-file', cooccur_shuf_file,
        '-x-max', str(x_max),
        '-iter', str(max_iter),
        '-vector-size', str(vector_size),
        '-binary', '2',  # Binary only
        '-vocab-file', vocab_file,
        '-verbose', str(verbose),
        '-alpha', str(alpha)
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: glove training failed")
        sys.exit(1)
    
    print(f"✓ Step 4/4 Complete: GloVe training finished")
    
    # Convert to text format
    bin_file = f"{save_file}.bin"
    txt_file = os.path.join(output_dir, f'glove_rocstory_{vector_size}d.txt')
    
    convert_binary_to_text(vocab_file, bin_file, txt_file, vector_size)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✓ Training Complete!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  Embeddings (text):  {txt_file}")
    print(f"  Embeddings (binary): {bin_file}")
    print(f"  Vocabulary:         {vocab_file}")
    print(f"\nSummary:")
    print(f"  Vocabulary size:     {vocab_size_count:,} words")
    print(f"  Embedding dimension: {vector_size}")
    print(f"  Training iterations: {max_iter}")
    print(f"\n{'='*70}")
    print(f"Integration with Diffusion-LM:")
    print(f"{'='*70}")
    print(f"\n1. Update line 295 in improved-diffusion/improved_diffusion/text_datasets.py:")
    print(f"   glove_model = load_glove_model('{txt_file}')")
    print(f"\n2. Train with: --experiment glove --in_channel {vector_size}")
    print(f"\n{'='*70}\n")
    
    # Cleanup option
    cleanup = input("Remove intermediate files (cooccurrence files)? [y/N]: ").strip().lower()
    if cleanup == 'y':
        try:
            os.remove(cooccur_file)
            os.remove(cooccur_shuf_file)
            print("✓ Intermediate files removed")
        except Exception as e:
            print(f"Warning: Could not remove files: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Train GloVe embeddings using official Stanford implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--glove-dir', type=str, required=True,
                       help='Path to GloVe repository directory (containing build/)')
    parser.add_argument('--corpus', type=str, 
                       default='datasets/ROCstory/roc_train_corpus.txt',
                       help='Path to prepared corpus file')
    parser.add_argument('--output-dir', type=str,
                       default='predictability/glove',
                       help='Output directory for embeddings')
    parser.add_argument('--dim', type=int, default=50,
                       help='Embedding dimension')
    parser.add_argument('--window', type=int, default=10,
                       help='Context window size')
    parser.add_argument('--min-count', type=int, default=3,
                       help='Minimum word frequency')
    parser.add_argument('--iter', type=int, default=25,
                       help='Number of training iterations')
    parser.add_argument('--threads', type=int, default=8,
                       help='Number of threads')
    parser.add_argument('--memory', type=float, default=4.0,
                       help='Memory for co-occurrence matrix (GB)')
    parser.add_argument('--x-max', type=int, default=100,
                       help='Cutoff for weighting function')
    parser.add_argument('--alpha', type=float, default=0.75,
                       help='Weighting function exponent')
    
    args = parser.parse_args()
    
    # Validate GloVe installation
    if not check_glove_installation(args.glove_dir):
        sys.exit(1)
    
    # Check corpus exists
    if not os.path.exists(args.corpus):
        print(f"ERROR: Corpus not found: {args.corpus}")
        print("\nPlease run: python prepare_rocstory_corpus.py")
        sys.exit(1)
    
    # Train embeddings
    train_glove_embeddings(
        glove_dir=args.glove_dir,
        corpus_file=args.corpus,
        output_dir=args.output_dir,
        vector_size=args.dim,
        window_size=args.window,
        min_count=args.min_count,
        max_iter=args.iter,
        num_threads=args.threads,
        memory=args.memory,
        x_max=args.x_max,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()
