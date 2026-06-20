"""Extract embeddings from BERT-tokenized Diffusion-LM models and export
for TF Embedding Projector (TSV) and Embedding Comparator (via .npz).

e2e model (rand128_bert_uncased): random init, jointly trained with diffusion.
frozen model (bert128_bert_tiny_frozen): BERT-initialized, frozen during training.

Optional: filter by corpus frequency (--min_count), or take the top --top_k_tokens
ids by count (e.g. 5180 to match ROCStories-sized analyses), using JSON lines,
Parquet shards, or a saved token_frequencies.tsv.
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import BertTokenizerFast

E2E_DIR = "trained_models/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_uncased"
FROZEN_DIR = "trained_models/diff_roc_pad_bert128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd101_bert_tiny_frozen"


def load_e2e_embeddings(model_dir, checkpoint="ema_0.9999_400000.pt"):
    state = torch.load(f"{model_dir}/{checkpoint}", map_location="cpu")
    return state["word_embedding.weight"]


def load_frozen_embeddings(model_dir):
    state = torch.load(f"{model_dir}/random_emb.torch", map_location="cpu")
    return state["weight"]


def count_token_ids_from_json(corpus_path, tokenizer):
    """Count subword token ids over texts (one JSON object per line; each value a string)."""
    counts = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            texts = json.loads(line.strip())
            for text in texts:
                tokens = tokenizer.encode(text, add_special_tokens=True)
                counts.update(tokens)
    return counts


def count_token_ids_from_parquet_dir(
    parquet_dir,
    tokenizer,
    text_column="text",
    batch_texts=512,
):
    """Count subword token ids over all *.parquet files under parquet_dir (e.g. HF shards)."""
    pattern = os.path.join(parquet_dir, "*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files matching {pattern}")
    counts = Counter()
    for i, path in enumerate(paths):
        table = pq.read_table(path, columns=[text_column])
        texts = [t for t in table.column(text_column).to_pylist() if t]
        for start in range(0, len(texts), batch_texts):
            chunk = texts[start : start + batch_texts]
            enc = tokenizer(
                chunk,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            for ids in enc["input_ids"]:
                counts.update(ids)
        n_sub = sum(counts.values())
        print(f"  parquet shard {i + 1}/{len(paths)}  ({os.path.basename(path)})  total_subword_occurrences={n_sub}")
    return counts


def filter_embeddings_by_freq(emb_e2e, emb_frozen, emb_init, keep_ids, vocab_size):
    """Subset embedding rows to sorted token ids; drop ids outside the matrix."""
    keep_ids = sorted({tid for tid in keep_ids if 0 <= tid < vocab_size})
    if not keep_ids:
        raise ValueError("After filtering, no token ids remain.")
    idx = torch.tensor(keep_ids, dtype=torch.long)
    return (
        emb_e2e.index_select(0, idx),
        emb_frozen.index_select(0, idx),
        emb_init.index_select(0, idx),
        keep_ids,
    )


def load_token_frequencies_tsv(path: str) -> Counter:
    """Load token_id -> count from token_frequencies.tsv (header: token_id\\ttoken\\tcount)."""
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            counts[int(parts[0])] = int(parts[2])
    return counts


def export_token_frequencies(counts: Counter, id2word: dict[int, str], out_dir: str) -> str:
    """Write token_id / token / count as TSV, sorted by count desc."""
    path = os.path.join(out_dir, "token_frequencies.tsv")
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("token_id\ttoken\tcount\n")
        for tid, c in rows:
            tok = id2word.get(int(tid), "<UNK_ID>")
            f.write(f"{int(tid)}\t{tok}\t{int(c)}\n")
    return path


def export_tsv(embeddings, words, prefix, out_dir):
    vec_path = f"{out_dir}/{prefix}_vectors.tsv"
    meta_path = f"{out_dir}/{prefix}_metadata.tsv"

    with open(vec_path, "w") as f:
        for row in embeddings:
            f.write("\t".join(str(x) for x in row) + "\n")

    with open(meta_path, "w") as f:
        for w in words:
            f.write(w.replace("\n", "<NL>") + "\n")

    print(f"  {vec_path}  ({embeddings.shape})")
    print(f"  {meta_path}  ({len(words)} labels)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e_dir", default=E2E_DIR)
    parser.add_argument("--e2e_checkpoint", default="ema_0.9999_400000.pt")
    parser.add_argument("--frozen_dir", default=FROZEN_DIR)
    parser.add_argument("--output_dir", default="embeddings/bert_128d")
    parser.add_argument("--show_top_k", type=int, default=20)
    freq = parser.add_mutually_exclusive_group()
    freq.add_argument(
        "--count_json",
        default=None,
        metavar="PATH",
        help="Filter to token ids with frequency >= min_count in this JSON-lines corpus "
        "(each line: JSON list of text strings, e.g. ROCStories roc_train.json).",
    )
    freq.add_argument(
        "--count_parquet_dir",
        default=None,
        metavar="DIR",
        help="Filter using token frequencies over all *.parquet files in this directory "
        "(reads column --count_parquet_text_column), e.g. roots_en_wikipedia/data.",
    )
    freq.add_argument(
        "--token_frequencies_tsv",
        default=None,
        metavar="PATH",
        help="Skip corpus scanning: load counts from token_frequencies.tsv (e.g. from a "
        "prior run with --save_token_frequencies), then keep ids with count >= min_count.",
    )
    parser.add_argument(
        "--count_parquet_text_column",
        default="text",
        help="Text column name in parquet files (default: text).",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=50,
        help="With corpus counts loaded: keep token ids with count >= this value "
        "(default 50). Ignored if --top_k_tokens is set.",
    )
    parser.add_argument(
        "--top_k_tokens",
        type=int,
        default=None,
        metavar="K",
        help="With corpus counts loaded: keep exactly the K token ids with highest "
        "counts (ties broken by token_id). Matches a fixed vocabulary size like ROC "
        "Stories min_count=50 (~5180 types). Ignores --min_count.",
    )
    parser.add_argument(
        "--save_token_frequencies",
        action="store_true",
        help="When counting from corpus (--count_json/--count_parquet_dir), save "
        "token_id/token/count to output_dir/token_frequencies.tsv (sorted by count desc).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.e2e_dir)
    vocab = tokenizer.get_vocab()
    id2word = {v: k for k, v in vocab.items()}
    print(f"Vocab size: {len(vocab)}")

    emb_e2e = load_e2e_embeddings(args.e2e_dir, checkpoint=args.e2e_checkpoint)
    emb_frozen = load_frozen_embeddings(args.frozen_dir)
    emb_init = load_frozen_embeddings(args.e2e_dir)

    keep_ids = None
    counts = None
    if args.count_json:
        print(f"Counting token ids from {args.count_json}...")
        counts = count_token_ids_from_json(args.count_json, tokenizer)
    elif args.count_parquet_dir:
        print(
            f"Counting token ids from parquet under {args.count_parquet_dir} "
            f"(column={args.count_parquet_text_column})..."
        )
        counts = count_token_ids_from_parquet_dir(
            args.count_parquet_dir,
            tokenizer,
            text_column=args.count_parquet_text_column,
        )
    elif args.token_frequencies_tsv:
        print(f"Loading token counts from {args.token_frequencies_tsv}...")
        counts = load_token_frequencies_tsv(args.token_frequencies_tsv)

    if counts is not None:
        if args.top_k_tokens is not None:
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            take = min(args.top_k_tokens, len(ranked))
            keep_ids = sorted([tid for tid, _ in ranked[:take]])
            print(
                f"Keeping top {take} token ids by corpus frequency "
                f"(requested top_k_tokens={args.top_k_tokens})"
            )
        else:
            keep_ids = sorted([tid for tid, c in counts.items() if c >= args.min_count])
            print(f"Keeping {len(keep_ids)} token ids (min_count={args.min_count})")

    if counts is not None and args.save_token_frequencies:
        freq_path = export_token_frequencies(counts, id2word, args.output_dir)
        print(f"Saved token frequencies to {freq_path}")

    if keep_ids is not None:
        emb_e2e, emb_frozen, emb_init, keep_ids = filter_embeddings_by_freq(
            emb_e2e, emb_frozen, emb_init, keep_ids, emb_e2e.shape[0],
        )
        words = [id2word[i] for i in keep_ids]
    else:
        words = [id2word[i] for i in range(len(id2word))]

    print(f"e2e embeddings:    {emb_e2e.shape}")
    print(f"frozen embeddings: {emb_frozen.shape}")
    print(f"e2e init:          {emb_init.shape}")

    print("\n--- Norms ---")
    print(f"e2e mean norm:     {torch.norm(emb_e2e, dim=-1).mean():.4f}")
    print(f"frozen mean norm:  {torch.norm(emb_frozen, dim=-1).mean():.4f}")
    print(f"e2e init mean norm:{torch.norm(emb_init, dim=-1).mean():.4f}")

    diff = emb_e2e - emb_init
    per_word_drift = torch.norm(diff, dim=-1)
    cos_sim = torch.nn.functional.cosine_similarity(emb_e2e, emb_init, dim=-1)

    print("\n--- e2e drift from initialization ---")
    print(f"Mean L2 drift:     {per_word_drift.mean():.4f}")
    print(f"Max L2 drift:      {per_word_drift.max():.4f}")
    print(f"Min L2 drift:      {per_word_drift.min():.4f}")
    print(f"Mean cosine sim:   {cos_sim.mean():.4f}")

    n_tok = emb_e2e.shape[0]
    topk = torch.topk(per_word_drift, k=min(args.show_top_k, n_tok))
    print(f"\nTop {args.show_top_k} most changed tokens (e2e vs init):")
    for i, (idx, drift) in enumerate(zip(topk.indices, topk.values)):
        word = words[idx.item()]
        sim = cos_sim[idx].item()
        print(f"  {i+1:3d}. {word:20s}  drift={drift:.4f}  cos_sim={sim:.4f}")

    npz_path = f"{args.output_dir}/embeddings_comparison.npz"
    save_kw = dict(
        e2e=emb_e2e.detach().numpy(),
        frozen=emb_frozen.detach().numpy(),
        init=emb_init.detach().numpy(),
        vocab=json.dumps(vocab),
    )
    if keep_ids is not None:
        save_kw["keep_token_ids"] = np.array(keep_ids, dtype=np.int64)
    np.savez(npz_path, **save_kw)
    print(f"\nSaved .npz to {npz_path}")

    print("\nExporting TSV for TF Embedding Projector...")
    export_tsv(emb_e2e.detach().numpy(), words, "e2e", args.output_dir)
    export_tsv(emb_frozen.detach().numpy(), words, "frozen", args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
