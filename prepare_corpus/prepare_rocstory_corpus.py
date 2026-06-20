"""Prepare ROCStory plain-text corpus for GloVe training (Spacy tokenization)."""

import json
from collections import Counter
from spacy.lang.en import English


def tokenize_text(text, tokenizer):
    return [tok.text for tok in tokenizer(text.strip())]


def prepare_corpus(input_file, output_file, stats_file=None):
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print()
    
    nlp = English()
    tokenizer = nlp.tokenizer
    
    total_stories = 0
    total_tokens = 0
    all_tokens = []
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for i, line in enumerate(f_in):
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i:,} stories...")
            
            try:
                story = json.loads(line.strip())[0]
                tokens = tokenize_text(story, tokenizer)
                
                if tokens:  # Skip empty stories
                    f_out.write(' '.join(tokens) + '\n')
                    total_tokens += len(tokens)
                    all_tokens.extend(tokens)
                    total_stories += 1
                    
            except (json.JSONDecodeError, IndexError) as e:
                print(f"  Warning: Skipping malformed line {i}: {e}")
                continue
    
    # Calculate statistics
    vocab = Counter(all_tokens)
    
    print(f"\n{'='*70}")
    print(f"Corpus Preparation Complete!")
    print(f"{'='*70}")
    print(f"Output file: {output_file}")
    print()
    print(f"Statistics:")
    print(f"  Total stories: {total_stories:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens per story: {total_tokens/total_stories:.1f}")
    print(f"  Unique tokens: {len(vocab):,}")
    print(f"  Tokens with freq ≥ 5: {sum(1 for c in vocab.values() if c >= 5):,}")
    print(f"  Tokens with freq ≥ 3: {sum(1 for c in vocab.values() if c >= 3):,}")
    print()
    print(f"Top 20 most frequent words:")
    for word, count in vocab.most_common(20):
        print(f"    {word:15s} {count:,}")
    print(f"{'='*70}")
    
    # Save statistics if requested
    if stats_file:
        with open(stats_file, 'w') as f:
            f.write(f"Total stories: {total_stories}\n")
            f.write(f"Total tokens: {total_tokens}\n")
            f.write(f"Avg tokens per story: {total_tokens/total_stories:.2f}\n")
            f.write(f"Unique tokens: {len(vocab)}\n")
            f.write(f"\nTop 100 words:\n")
            for word, count in vocab.most_common(100):
                f.write(f"{word}\t{count}\n")
        print(f"\nStatistics saved to: {stats_file}")
    
    return total_stories, total_tokens, len(vocab)

if __name__ == "__main__":
    prepare_corpus(
        input_file='datasets/ROCstory/roc_train.json',
        output_file='datasets/ROCstory/roc_train_corpus.txt',
        stats_file='datasets/ROCstory/corpus_stats.txt'
    )
    
    print("\n" + "="*70)
    print("Next steps:")
    print("="*70)
    print("1. Train GloVe embeddings:")
    print("   bash train_glove_rocstory.sh")
    print()
    print("2. Or use the custom parameters script:")
    print("   python train_glove_rocstory_official.py")
    print("="*70)
