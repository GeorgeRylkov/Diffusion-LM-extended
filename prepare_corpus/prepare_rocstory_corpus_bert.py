"""Prepare ROCStory corpus with BERT WordPiece tokenization."""

import json
import os
from collections import Counter

from transformers import AutoTokenizer

BERT_MODEL = "bert-base-cased"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_corpus(input_file, output_file, stats_file=None):
    print(f"Loading tokenizer: {BERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    print(f"Full BERT vocab size: {tokenizer.vocab_size:,}")
    print()
    print(f"Reading from: {input_file}")
    print(f"Writing to:   {output_file}")
    print()

    total_stories = 0
    total_tokens = 0
    all_tokens = []

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for i, line in enumerate(f_in):
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i:,} stories...")

            try:
                story = json.loads(line.strip())[0]
                tokens = tokenizer.tokenize(story.strip())

                if tokens:
                    f_out.write(" ".join(tokens) + "\n")
                    total_tokens += len(tokens)
                    all_tokens.extend(tokens)
                    total_stories += 1

            except (json.JSONDecodeError, IndexError) as e:
                print(f"  Warning: Skipping malformed line {i}: {e}")
                continue

    vocab = Counter(all_tokens)

    print(f"\n{'=' * 70}")
    print("Corpus Preparation Complete!")
    print(f"{'=' * 70}")
    print(f"Output file: {output_file}")
    print()
    print("Statistics:")
    print(f"  Total stories:              {total_stories:,}")
    print(f"  Total tokens:               {total_tokens:,}")
    print(f"  Avg tokens per story:       {total_tokens / total_stories:.1f}")
    print(f"  Unique tokens in corpus:    {len(vocab):,}")
    print(f"  Full BERT vocab size:       {tokenizer.vocab_size:,}")
    print(f"  Tokens with freq >= 5:      {sum(1 for c in vocab.values() if c >= 5):,}")
    print(f"  Tokens with freq >= 3:      {sum(1 for c in vocab.values() if c >= 3):,}")
    print()
    subword_count = sum(1 for t in vocab if t.startswith("##"))
    print(f"  Subword tokens (##...):     {subword_count:,} / {len(vocab):,}")
    print()
    print("Top 30 most frequent tokens:")
    for word, count in vocab.most_common(30):
        print(f"    {word:15s} {count:,}")
    print(f"{'=' * 70}")

    if stats_file:
        with open(stats_file, "w") as f:
            f.write(f"BERT model: {BERT_MODEL}\n")
            f.write(f"Full BERT vocab size: {tokenizer.vocab_size}\n")
            f.write(f"Total stories: {total_stories}\n")
            f.write(f"Total tokens: {total_tokens}\n")
            f.write(f"Avg tokens per story: {total_tokens / total_stories:.2f}\n")
            f.write(f"Unique tokens in corpus: {len(vocab)}\n")
            f.write(f"Subword tokens (##...): {subword_count}\n")
            f.write("\nTop 100 tokens:\n")
            for word, count in vocab.most_common(100):
                f.write(f"{word}\t{count}\n")
        print(f"\nStatistics saved to: {stats_file}")

    return total_stories, total_tokens, len(vocab)


if __name__ == "__main__":
    prepare_corpus(
        input_file=os.path.join(SCRIPT_DIR, "datasets/ROCstory/roc_train.json"),
        output_file=os.path.join(SCRIPT_DIR, "datasets/ROCstory/roc_train_corpus_bert.txt"),
        stats_file=os.path.join(SCRIPT_DIR, "datasets/ROCstory/corpus_stats_bert.txt"),
    )
