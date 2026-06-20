"""
Pre-tokenize the Wikipedia JSON corpus into a HuggingFace Arrow DatasetDict.

Motivation
----------
`prepare_wikipedia_corpus.py` produces line-oriented JSON
(`wiki_train.json`, `wiki_valid.json`) where each line is `["passage text"]`.
At training time, `text_datasets.py::get_corpus_rocstory` currently loads the
entire JSON into a Python `sentence_lst` (one list of string tokens per
passage) and then runs `helper_tokenize_stream` to convert tokens to ids and
pad. For the partial Wikipedia (~500K passages) that costs ~2 GB of RAM per
process; for the full corpus (~27M passages) it costs ~100 GB per process and
OOMs multi-GPU nodes because each DDP rank keeps its own copy of the text.

This script does the tokenization exactly once, offline, and writes a
memory-mappable Arrow dataset with a single `input_ids` column (int32,
fixed length = seqlen, right-padded with the tokenizer's pad id). At train
time every rank on the same node can mmap the same file, dropping per-rank
RAM to a few GB regardless of world size.

Output layout
-------------
    <output_dir>/
        tokenizer_info.json    # tokenizer name, vocab size, seqlen, pad id
        train/                 # HF Dataset (Arrow) with 'input_ids' column
        valid/                 # HF Dataset (Arrow) with 'input_ids' column

Note on seqlen vs max_words
---------------------------
`prepare_wikipedia_corpus.py` enforces a word-count budget (max_words, default
64) when greedily joining sentences into passages. `seqlen` here is the
WordPiece-token count the model actually operates on (must equal
image_size**2 in train.py). They are different units operating on different
stages of the pipeline and both are needed:
  * max_words caps the semantic content of one training passage and is
    frozen into wiki_train.json / wiki_valid.json at preprocessing time.
  * seqlen caps the tensor shape and is applied here during tokenization,
    truncating any passage whose WordPiece length exceeds seqlen.
For bert-base-uncased, a passage of N whitespace words tokenizes to roughly
1.3-1.7N WordPiece tokens. Rule of thumb: seqlen >= 1.5 * max_words if you
want to retain most passages intact. seqlen=64 with max_words=64 (the current
pipeline's default) truncates ~50% of passages — this matches the behavior of
the original single-GPU pipeline for bit-for-bit compatibility.

Usage
-----
    python tokenize_wiki_corpus.py \
        --input_dir  datasets/roots_en_wikipedia \
        --output_dir datasets/roots_en_wikipedia/bert_uncased_seq64 \
        --tokenizer  bert-base-uncased \
        --seqlen     64 \
        --num_proc   8
"""

import argparse
import json
import os

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing wiki_train.json / wiki_valid.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the tokenized Arrow DatasetDict")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased",
                        help="HuggingFace tokenizer name or path")
    parser.add_argument("--seqlen", type=int, default=64,
                        help="Fixed sequence length (pad/truncate to this). "
                             "Must match image_size**2 of the training run "
                             "(default 64 == image_size 8).")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Number of processes for dataset .map()")
    parser.add_argument("--writer_batch_size", type=int, default=10_000,
                        help="Arrow writer batch size (controls peak RAM per worker)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output_dir if it exists")
    return parser.parse_args()


def load_jsonl_passages(path):
    """Stream a ROCstory-style JSONL (['passage text'] per line) into a Dataset."""
    # Each line is a JSON array with exactly one string. `load_dataset('json', ...)`
    # expects flat objects, so we stream manually.
    def gen():
        with open(path, "r") as f:
            for line in f:
                text = json.loads(line)[0]
                yield {"text": text}
    return Dataset.from_generator(gen)


def build_tokenize_fn(tokenizer, seqlen):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # GPT-2 etc. don't have a pad token by default.
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch):
        enc = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=seqlen,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {"input_ids": enc["input_ids"]}

    return tokenize_batch, pad_id


def main():
    args = parse_args()

    if os.path.exists(args.output_dir) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already exists. Pass --overwrite to replace."
        )
    os.makedirs(args.output_dir, exist_ok=True)

    train_json = os.path.join(args.input_dir, "wiki_train.json")
    valid_json = os.path.join(args.input_dir, "wiki_valid.json")
    for p in [train_json, valid_json]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Run prepare_wikipedia_corpus.py first."
            )

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # Load JSONL -> Dataset via streaming generator so we don't build a 27M-row
    # Python list in RAM.
    print(f"Streaming train split from {train_json}")
    ds_train = load_jsonl_passages(train_json)
    print(f"Streaming valid split from {valid_json}")
    ds_valid = load_jsonl_passages(valid_json)
    print(f"  train: {len(ds_train):,} passages")
    print(f"  valid: {len(ds_valid):,} passages")

    tokenize_batch, pad_id = build_tokenize_fn(tokenizer, args.seqlen)

    print(f"\nTokenizing to input_ids (seqlen={args.seqlen}, pad_id={pad_id}, "
          f"num_proc={args.num_proc})")
    tokenized = DatasetDict({
        "train": ds_train.map(
            tokenize_batch,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=["text"],
            writer_batch_size=args.writer_batch_size,
            desc="tokenize train",
        ),
        "valid": ds_valid.map(
            tokenize_batch,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=["text"],
            writer_batch_size=args.writer_batch_size,
            desc="tokenize valid",
        ),
    })

    # Save DatasetDict -> output_dir/{train,valid}/ Arrow files. Shard the
    # train split so each arrow file is ~400 MB instead of one 7-8 GB blob;
    # this matches TenCDM's layout (data/load.py) and gives better disk cache
    # locality, especially when multiple ranks mmap the same file on a node.
    print(f"\nSaving tokenized dataset to {args.output_dir}")
    tokenized.save_to_disk(
        args.output_dir,
        num_shards={"train": 20, "valid": 1},
    )

    info = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "seqlen": args.seqlen,
        "pad_id": pad_id,
        "num_train": len(tokenized["train"]),
        "num_valid": len(tokenized["valid"]),
    }
    info_path = os.path.join(args.output_dir, "tokenizer_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nWrote {info_path}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
