"""Convert roots_en_wikipedia Parquet shards into ROCstories-compatible JSON Lines."""

import argparse
import glob
import json
import os
import random

import nltk
from tqdm import tqdm


def load_parquet_texts(input_dir, max_articles=None):
    """Load text from Parquet shards via pyarrow.

    If max_articles is set, read shards in sorted filename order and stop after
    that many rows (first N articles, no shuffle). Otherwise load everything.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    if max_articles is None:
        tables = []
        for f in tqdm(parquet_files, desc="Loading parquet shards"):
            tables.append(pq.read_table(f, columns=["text"]))

        combined = pa.concat_tables(tables)
        texts = combined.column("text").to_pylist()
        del tables, combined
        print(f"Total articles loaded: {len(texts):,}")
        return texts

    texts = []
    for f in tqdm(parquet_files, desc="Loading parquet shards (first N)"):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4096, columns=["text"]):
            for text in batch.column(0).to_pylist():
                texts.append(text)
                if len(texts) >= max_articles:
                    print(f"Total articles loaded (capped): {len(texts):,}")
                    return texts
    print(f"Total articles loaded: {len(texts):,}")
    return texts


def chunk_into_passages(paragraphs, sent_tokenizer, max_words):
    """Split paragraphs into sentences and greedily join into passages."""
    passages = []
    for para in tqdm(paragraphs, desc="Chunking into passages"):
        sentences = sent_tokenizer.tokenize(para)
        current = ""
        for sent in sentences:
            if current and len(current.split()) + len(sent.split()) >= max_words:
                passages.append(current.strip())
                current = sent
            else:
                current = (current + " " + sent) if current else sent
        if current.strip():
            passages.append(current.strip())
    return passages


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Wikipedia corpus for Diffusion-LM")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to directory containing .parquet shards")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for wiki_train.json / wiki_valid.json")
    parser.add_argument("--max_words", type=int, default=64,
                        help="Target max words per passage (default: 64, matching tencdm)")
    parser.add_argument("--min_chars", type=int, default=600,
                        help="Minimum paragraph length in characters (default: 600)")
    parser.add_argument("--valid_ratio", type=float, default=0.002,
                        help="Fraction of passages for validation (default: 0.002)")
    parser.add_argument("--max_articles", type=int, default=None,
                        help="Use only the first N articles (sorted shard order; "
                             "None = load all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        nltk.data.find("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)
    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    # --- Step 1: Load articles (optional cap: first N in shard order) ---
    texts = load_parquet_texts(args.input_dir, max_articles=args.max_articles)

    # --- Step 2-3: Split into paragraphs, filter short ones ---
    print(f"\nSplitting articles into paragraphs (min {args.min_chars} chars)...")
    all_paragraphs = []
    for text in tqdm(texts, desc="Splitting paragraphs"):
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) >= args.min_chars:
                all_paragraphs.append(para)
    del texts
    print(f"Paragraphs after filtering: {len(all_paragraphs):,}")

    # --- Step 5-6: Sentence split + greedy join ---
    passages = chunk_into_passages(all_paragraphs, sent_tokenizer, args.max_words)
    del all_paragraphs
    print(f"Total passages: {len(passages):,}")

    # --- Step 7: Train/valid split ---
    random.shuffle(passages)
    n_valid = max(1, int(len(passages) * args.valid_ratio))
    valid_passages = passages[:n_valid]
    train_passages = passages[n_valid:]
    print(f"\nSplit: {len(train_passages):,} train / {len(valid_passages):,} valid")

    # --- Step 8: Write JSON Lines ---
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "wiki_train.json")
    valid_path = os.path.join(args.output_dir, "wiki_valid.json")

    for path, data, label in [
        (train_path, train_passages, "train"),
        (valid_path, valid_passages, "valid"),
    ]:
        with open(path, "w") as f:
            for passage in tqdm(data, desc=f"Writing {label}"):
                print(json.dumps([passage]), file=f)
        print(f"Written {len(data):,} {label} passages to {path}")

    # --- Stats ---
    all_passages = train_passages + valid_passages
    word_counts = [len(p.split()) for p in all_passages]
    avg_words = sum(word_counts) / len(word_counts)
    total_words = sum(word_counts)

    stats_path = os.path.join(args.output_dir, "corpus_stats.txt")
    stats_lines = [
        f"Source: {args.input_dir}",
        f"max_words: {args.max_words}",
        f"min_chars: {args.min_chars}",
        f"max_articles: {args.max_articles or 'all'}",
        f"seed: {args.seed}",
        f"",
        f"Total passages: {len(all_passages):,}",
        f"  Train: {len(train_passages):,}",
        f"  Valid: {len(valid_passages):,}",
        f"Total words: {total_words:,}",
        f"Avg words per passage: {avg_words:.1f}",
        f"Min words: {min(word_counts)}",
        f"Max words: {max(word_counts)}",
        f"",
        f"Sample passages:",
    ]
    for i in range(min(5, len(train_passages))):
        stats_lines.append(f"  [{i}] {train_passages[i][:200]}...")
    stats_text = "\n".join(stats_lines)

    with open(stats_path, "w") as f:
        f.write(stats_text + "\n")

    print(f"\n{'='*70}")
    print("Corpus Preparation Complete!")
    print(f"{'='*70}")
    print(stats_text)
    print(f"{'='*70}")
    print(f"\nStatistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
