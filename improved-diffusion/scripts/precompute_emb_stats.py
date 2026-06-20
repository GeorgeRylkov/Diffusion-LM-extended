"""Precompute corpus-weighted embedding mean/std (and mean L2) for Diffusion-LM.

Wiki: mmap Arrow rows + streaming token chunks (no flat 1.7B-token tensor in RAM).
ROC: legacy JSONL tokenization.

Remote/cluster-oriented: local tokenizer/BERT-tiny paths, mmap checkpoint load,
low-RAM streaming (float32 gathers, float64 accumulators), optional tiny
`--e2e_word_embedding_pt` to skip deserializing the full EMA dict.

See improved_diffusion/text_datasets.py :: BERT_TARGET_CORPUS_NORMS for where
the printed mean L2 norm goes.

One-off extraction of only word embeddings from a full EMA (login node):
  python -c "import torch; d=torch.load('ema.pt',map_location='cpu',weights_only=True);\\
    torch.save({'word_embedding.weight': d['word_embedding.weight'].float().cpu()}, 'we.pt')"
Then pass --e2e_word_embedding_pt we.pt and omit --e2e_checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path = [p for p in sys.path if "Diffusion-LM/transformers" not in p]
from transformers import AutoTokenizer, BertModel  # noqa: E402

WORD_EMB_KEY = "word_embedding.weight"


def _from_pretrained_kwargs(model_id_or_path):
    path = os.path.expanduser(model_id_or_path)
    if os.path.isdir(path):
        return {"local_files_only": True}
    return {}


def collect_corpus_ids_roc(tok, roc_train_dir):
    path = f"{roc_train_dir}/roc_train.json"
    all_ids = []
    with open(path, "r") as f:
        for line in f:
            text = json.loads(line)[0].strip()
            ids = tok.convert_tokens_to_ids(tok.tokenize(text))
            all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)


def load_wiki_arrow_train_split(wiki_tok_dir, split="train"):
    import datasets as hf_datasets

    print(f"[arrow-fastpath] loading pre-tokenized wiki from {wiki_tok_dir}")
    dsdict = hf_datasets.load_from_disk(wiki_tok_dir)

    info_path = os.path.join(wiki_tok_dir, "tokenizer_info.json")
    pad_id = 0
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        print(f"[arrow-fastpath] tokenizer_info: {info}")
        pad_id = info.get("pad_id", pad_id)

    if split not in dsdict:
        raise ValueError(
            f"invalid split for Wiki Arrow dataset: {split!r} "
            f"(available: {list(dsdict.keys())})"
        )
    chosen = dsdict[split]
    # Torch column format avoids a numpy round-trip copy per row window.
    try:
        try:
            chosen = chosen.with_format("torch", columns=["input_ids"], device="cpu")
        except TypeError:
            chosen = chosen.with_format("torch", columns=["input_ids"])
    except Exception as e:
        print(f"  [arrow-fastpath] with_format(torch) skipped ({e}); using default I/O.")
    n_rows = len(chosen)
    seq_len = len(chosen[0]["input_ids"])
    n_flat_with_pad = n_rows * seq_len
    print(
        f"  Arrow rows: {n_rows:,}, seqlen: {seq_len} (positions with pad: {n_flat_with_pad:,})"
    )
    print("  Streaming row windows over mmap (no full in-RAM flatten).")
    return chosen, pad_id


def iter_wiki_arrow_token_chunks(ds, pad_id, row_chunk=1024, token_chunk=32_768):
    n = len(ds)
    i = 0
    while i < n:
        j = min(i + row_chunk, n)
        batch = ds[i:j]["input_ids"]
        if torch.is_tensor(batch):
            t = batch.reshape(-1).to(dtype=torch.long, copy=False)
        else:
            t = torch.as_tensor(np.asarray(batch, dtype=np.int64)).reshape(-1)
        t = t[t != pad_id]
        for k in range(0, len(t), token_chunk):
            yield t[k : k + token_chunk].contiguous()
        i = j


def _wiki_chunk_factory(ds, pad_id, row_chunk, token_chunk):
    def _factory():
        return iter_wiki_arrow_token_chunks(ds, pad_id, row_chunk, token_chunk)

    return _factory


def _maybe_gc(chunk_idx: int, every: int) -> None:
    if every > 0 and chunk_idx % every == 0:
        gc.collect()


def compute_stats(embeddings, token_ids, *, chunk_size=32_768):
    emb_dim = embeddings.shape[1]
    total_sum = torch.zeros(emb_dim, dtype=torch.float64)
    total_sq = torch.zeros(emb_dim, dtype=torch.float64)
    n = 0
    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start : start + chunk_size]
        x = embeddings[chunk_ids].float()
        total_sum += x.sum(dim=0).double()
        total_sq += (x * x).sum(dim=0).double()
        n += len(chunk_ids)
    mean = total_sum / n
    var = total_sq / n - mean**2
    std = var.clamp(min=0).sqrt()
    return mean.float(), std.float(), n


def mean_l2_norm(embeddings, token_ids, chunk_size=32_768):
    total_norm = 0.0
    n = 0
    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start : start + chunk_size]
        x = embeddings[chunk_ids].float()
        total_norm += torch.linalg.vector_norm(x, dim=-1).sum().item()
        n += len(chunk_ids)
    return total_norm / n


def compute_stats_and_mean_l2_streaming(
    embeddings,
    chunk_iter_factory,
    *,
    progress_label=None,
    log_every_n_chunks=0,
    gc_every_chunks=0,
    max_tokens=0,
):
    emb_dim = embeddings.shape[1]
    total_sum = torch.zeros(emb_dim, dtype=torch.float64)
    total_sq = torch.zeros(emb_dim, dtype=torch.float64)
    total_l2 = 0.0
    n = 0
    chunk_idx = 0
    stopped_early = False
    t0 = time.perf_counter()
    for chunk_ids in chunk_iter_factory():
        chunk_idx += 1
        if chunk_ids.numel() == 0:
            continue
        if max_tokens > 0 and n >= max_tokens:
            stopped_early = True
            break
        if max_tokens > 0 and n + chunk_ids.numel() > max_tokens:
            chunk_ids = chunk_ids[: max_tokens - n]
            if chunk_ids.numel() == 0:
                stopped_early = True
                break
        x = embeddings[chunk_ids].float()
        total_sum += x.sum(dim=0).double()
        total_sq += (x * x).sum(dim=0).double()
        total_l2 += torch.linalg.vector_norm(x, dim=-1).sum().item()
        n += chunk_ids.numel()
        _maybe_gc(chunk_idx, gc_every_chunks)
        if (
            progress_label
            and log_every_n_chunks > 0
            and chunk_idx % log_every_n_chunks == 0
        ):
            elapsed = time.perf_counter() - t0
            rate = n / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{progress_label}] chunk_batch={chunk_idx}, "
                f"tokens={n:,}, tok/s≈{rate:,.0f}"
            )
        if max_tokens > 0 and n >= max_tokens:
            stopped_early = True
            break
    if stopped_early and progress_label:
        print(
            f"  [{progress_label}] early-stop at tokens={n:,} (--max_tokens={max_tokens:,})"
        )
    mean = total_sum / n
    var = total_sq / n - mean**2
    std = var.clamp(min=0).sqrt()
    return mean.float(), std.float(), n, total_l2 / n


def mean_l2_norm_streaming(
    embeddings, chunk_iter_factory, *, gc_every_chunks=0, max_tokens=0
):
    total_norm = 0.0
    n = 0
    chunk_idx = 0
    for chunk_ids in chunk_iter_factory():
        chunk_idx += 1
        if chunk_ids.numel() == 0:
            continue
        if max_tokens > 0 and n >= max_tokens:
            break
        if max_tokens > 0 and n + chunk_ids.numel() > max_tokens:
            chunk_ids = chunk_ids[: max_tokens - n]
            if chunk_ids.numel() == 0:
                break
        x = embeddings[chunk_ids].float()
        total_norm += torch.linalg.vector_norm(x, dim=-1).sum().item()
        n += chunk_ids.numel()
        _maybe_gc(chunk_idx, gc_every_chunks)
    return total_norm / n


def mean_l2_norms_multi_streaming(
    embeddings_list, chunk_iter_factory, *, gc_every_chunks=0, max_tokens=0
):
    sums = [0.0 for _ in embeddings_list]
    n = 0
    chunk_idx = 0
    for chunk_ids in chunk_iter_factory():
        chunk_idx += 1
        if chunk_ids.numel() == 0:
            continue
        if max_tokens > 0 and n >= max_tokens:
            break
        if max_tokens > 0 and n + chunk_ids.numel() > max_tokens:
            chunk_ids = chunk_ids[: max_tokens - n]
            if chunk_ids.numel() == 0:
                break
        for idx, emb in enumerate(embeddings_list):
            x = emb[chunk_ids].float()
            sums[idx] += torch.linalg.vector_norm(x, dim=-1).sum().item()
        n += chunk_ids.numel()
        _maybe_gc(chunk_idx, gc_every_chunks)
    return [s / n for s in sums]


def _resolve_word_embedding_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        if WORD_EMB_KEY in obj:
            return obj[WORD_EMB_KEY]
        if len(obj) == 1:
            return next(iter(obj.values()))
    raise KeyError(
        f"expected a tensor or dict with {WORD_EMB_KEY!r}; got {type(obj).__name__}"
    )


def load_word_embedding_weight(path: str, *, mmap: bool = True) -> torch.Tensor:
    """Load a small .pt: raw tensor or dict holding word_embedding.weight."""
    load_kw: dict = {"map_location": "cpu", "weights_only": True}
    if mmap:
        try:
            ckpt = torch.load(path, mmap=True, **load_kw)
        except TypeError:
            ckpt = torch.load(path, **load_kw)
    else:
        ckpt = torch.load(path, **load_kw)
    if torch.is_tensor(ckpt):
        return ckpt.contiguous()
    w = _resolve_word_embedding_tensor(ckpt)
    del ckpt
    gc.collect()
    return w.contiguous()


def load_word_embedding_from_ema(path: str, *, mmap: bool = True) -> torch.Tensor:
    """Load full EMA dict, take word_embedding.weight, drop the rest (prefer --e2e_word_embedding_pt)."""
    load_kw: dict = {"map_location": "cpu", "weights_only": True}
    if mmap:
        try:
            ckpt = torch.load(path, mmap=True, **load_kw)
        except TypeError:
            ckpt = torch.load(path, **load_kw)
    else:
        ckpt = torch.load(path, **load_kw)
    n_keys = len(ckpt) if isinstance(ckpt, dict) else 0
    if not isinstance(ckpt, dict) or WORD_EMB_KEY not in ckpt:
        raise KeyError(
            f"e2e checkpoint missing {WORD_EMB_KEY!r}; "
            f"keys (first 20): {list(ckpt.keys())[:20] if isinstance(ckpt, dict) else 'n/a'}"
        )
    w_e2e = ckpt.pop(WORD_EMB_KEY)
    del ckpt
    gc.collect()
    print(
        f"  [e2e] Loaded from full checkpoint ({n_keys} top-level keys, mmap={mmap}); "
        f"kept {WORD_EMB_KEY} only. Shape: {w_e2e.shape}, dtype={w_e2e.dtype}"
    )
    return w_e2e


def main():
    parser = argparse.ArgumentParser()
    emb_src = parser.add_mutually_exclusive_group(required=True)
    emb_src.add_argument(
        "--e2e_checkpoint",
        default=None,
        help="Full EMA .pt (many keys). Prefer --e2e_word_embedding_pt on low-RAM nodes.",
    )
    emb_src.add_argument(
        "--e2e_word_embedding_pt",
        default=None,
        help="Small .pt with only word_embedding.weight (or a lone tensor). Skips full EMA load.",
    )
    parser.add_argument(
        "--no_mmap_checkpoint",
        action="store_true",
        help="Disable mmap for torch.load (work around buggy NFS / old PyTorch).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--roc_train",
        default=None,
        help="ROCstory dir containing roc_train.json",
    )
    group.add_argument(
        "--wiki_tokenized_dir",
        default=None,
        help="Pre-tokenized wiki Arrow DatasetDict (tokenize_wiki_corpus.py)",
    )
    parser.add_argument("--output", default="emb_norm_stats.pt")
    parser.add_argument(
        "--tokenizer_model",
        default="bert-base-uncased",
        help="Tokenizer id or local snapshot dir (offline compute nodes).",
    )
    parser.add_argument(
        "--bert_tiny_model",
        default="prajjwal1/bert-tiny",
        help="BERT-tiny snapshot for frozen source pass (unless --skip_frozen_source).",
    )
    parser.add_argument(
        "--arrow_row_chunk",
        type=int,
        default=1024,
        help="Wiki: Arrow rows per read (smaller → lower peak RAM).",
    )
    parser.add_argument(
        "--arrow_token_chunk",
        type=int,
        default=32_768,
        help="Wiki: max token ids per embedding gather (smaller → lower peak RAM).",
    )
    parser.add_argument(
        "--skip_frozen_source",
        action="store_true",
        help="Skip BERT-tiny and src_* / verification (e2e stats only).",
    )
    parser.add_argument(
        "--corpus_log_every_chunks",
        type=int,
        default=200,
        help="Wiki streaming: log every N iterator chunks (0 = off).",
    )
    parser.add_argument(
        "--gc_every_chunks",
        type=int,
        default=500,
        help="Call gc.collect() every N chunks (0 = never). Reduces RSS creep on long runs.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="Early-stop each streaming pass after N corpus tokens (0 = full corpus). "
        "Use on large corpora (e.g. wiki) where later chunks slow down from page-cache "
        "pressure; ~700M tokens gives 4-decimal mean/std/L2 stability for a 128-dim table.",
    )
    args = parser.parse_args()
    mmap = not args.no_mmap_checkpoint
    max_tokens = args.max_tokens

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_model, **_from_pretrained_kwargs(args.tokenizer_model)
    )

    wiki_chunks = None
    all_ids = None
    if args.roc_train is not None:
        print(f"Tokenizing full ROC corpus from {args.roc_train}...")
        all_ids = collect_corpus_ids_roc(tok, args.roc_train)
        print(f"Total corpus tokens: {len(all_ids):,}")
    else:
        print(f"Preparing mmap'd Arrow stream from {args.wiki_tokenized_dir}...")
        ds, pad_id = load_wiki_arrow_train_split(args.wiki_tokenized_dir, split="train")
        wiki_chunks = _wiki_chunk_factory(
            ds, pad_id, args.arrow_row_chunk, args.arrow_token_chunk
        )

    log_chunks = args.corpus_log_every_chunks
    gc_every = args.gc_every_chunks
    pl_frozen = "frozen" if log_chunks else None
    pl_e2e = "e2e" if log_chunks else None

    w_tiny = None
    src_mean = src_std = None
    n_src = 0
    tiny_l2 = None

    if args.skip_frozen_source:
        print("\n[skip_frozen_source] Skipping BERT-tiny load and source corpus pass.")
    else:
        print(f"\nLoading BERT-tiny embeddings from {args.bert_tiny_model!r}...")
        bert = BertModel.from_pretrained(
            args.bert_tiny_model, **_from_pretrained_kwargs(args.bert_tiny_model)
        )
        w_tiny = bert.embeddings.word_embeddings.weight.data.clone()
        del bert
        gc.collect()
        print(f"  Shape: {w_tiny.shape}")

        print("Computing source (bert-tiny) stats over full corpus...")
        if wiki_chunks is not None:
            src_mean, src_std, n_src, tiny_l2 = compute_stats_and_mean_l2_streaming(
                w_tiny,
                wiki_chunks,
                progress_label=pl_frozen,
                log_every_n_chunks=log_chunks,
                gc_every_chunks=gc_every,
                max_tokens=max_tokens,
            )
        else:
            src_mean, src_std, n_src = compute_stats(w_tiny, all_ids)
            tiny_l2 = mean_l2_norm(w_tiny, all_ids)
        print(f"  Corpus tokens (pad stripped): {n_src:,}")
        print(f"  Avg per-dim std: {src_std.mean():.6f}")
        print(f"  Corpus mean L2 norm: {tiny_l2:.4f}")

    print("\n[e2e] Loading word embedding weights...")
    t_ckpt = time.perf_counter()
    if args.e2e_word_embedding_pt is not None:
        w_e2e = load_word_embedding_weight(args.e2e_word_embedding_pt, mmap=mmap)
        print(
            f"  [e2e] Loaded small embedding file in {time.perf_counter() - t_ckpt:.1f}s. "
            f"Shape: {w_e2e.shape}, dtype={w_e2e.dtype}"
        )
    else:
        w_e2e = load_word_embedding_from_ema(args.e2e_checkpoint, mmap=mmap)
        print(f"  [e2e] torch.load + extract in {time.perf_counter() - t_ckpt:.1f}s.")

    w_e2e = w_e2e.contiguous()
    print("[e2e] Stage: corpus scan for target (e2e) stats...")
    t_scan = time.perf_counter()
    if wiki_chunks is not None:
        tgt_mean, tgt_std, n_tgt, e2e_norm = compute_stats_and_mean_l2_streaming(
            w_e2e,
            wiki_chunks,
            progress_label=pl_e2e,
            log_every_n_chunks=log_chunks,
            gc_every_chunks=gc_every,
            max_tokens=max_tokens,
        )
    else:
        tgt_mean, tgt_std, n_tgt = compute_stats(w_e2e, all_ids)
        e2e_norm = mean_l2_norm(w_e2e, all_ids)
    print(f"  [e2e] Corpus scan finished in {time.perf_counter() - t_scan:.1f}s.")
    print(f"  Tokens: {n_tgt:,}")
    print(f"  Avg per-dim std: {tgt_std.mean():.6f}")
    print(f"  Corpus mean L2 norm: {e2e_norm:.4f}")

    if not args.skip_frozen_source and n_src != n_tgt:
        print(f"WARNING: src vs tgt token counts differ: {n_src:,} vs {n_tgt:,}")

    stats = {
        "tgt_mean": tgt_mean,
        "tgt_std": tgt_std,
        "n_tokens": n_tgt if args.skip_frozen_source else n_src,
        "tokenizer": args.tokenizer_model,
        "corpus_source": (
            f"roc:{args.roc_train}"
            if args.roc_train is not None
            else f"wiki_arrow:{args.wiki_tokenized_dir}"
        ),
        "frozen_source_skipped": args.skip_frozen_source,
        "max_tokens_per_pass": max_tokens,
    }
    if not args.skip_frozen_source:
        stats["src_mean"] = src_mean
        stats["src_std"] = src_std
        stats["src_model"] = args.bert_tiny_model
    torch.save(stats, args.output)
    print(f"\nSaved stats to {args.output}")

    if args.skip_frozen_source:
        print(
            "\n[skip_frozen_source] Skipping Option A/B verification (needs BERT-tiny + src stats)."
        )
    else:
        w_normed_a = (w_tiny - src_mean) / src_std.clamp(min=1e-6) * tgt_std + tgt_mean
        print("\nVerification — Option A: per-dim distribution matching (full corpus):")
        if wiki_chunks is not None:
            l2_a, l2_e2e_v = mean_l2_norms_multi_streaming(
                [w_normed_a, w_e2e],
                wiki_chunks,
                gc_every_chunks=gc_every,
                max_tokens=max_tokens,
            )
            print(f"  Normed bert-tiny  -> mean L2 norm: {l2_a:.4f}")
            print(f"  E2E target        -> mean L2 norm: {l2_e2e_v:.4f}")
        else:
            print(
                f"  Normed bert-tiny  -> mean L2 norm: {mean_l2_norm(w_normed_a, all_ids):.4f}"
            )
            print(
                f"  E2E target        -> mean L2 norm: {mean_l2_norm(w_e2e, all_ids):.4f}"
            )

        w_zscore = (w_tiny - src_mean) / src_std.clamp(min=1e-6)
        if wiki_chunks is not None:
            zscore_norm = mean_l2_norm_streaming(
                w_zscore,
                wiki_chunks,
                gc_every_chunks=gc_every,
                max_tokens=max_tokens,
            )
        else:
            zscore_norm = mean_l2_norm(w_zscore, all_ids)
        scale = e2e_norm / zscore_norm
        w_normed_b = w_zscore * scale
        print("\nVerification — Option B: z-score + uniform rescale (full corpus):")
        print(f"  After z-score     -> mean L2 norm: {zscore_norm:.4f}")
        print(f"  E2E target        -> mean L2 norm: {e2e_norm:.4f}")
        print(f"  Scale factor: {scale:.4f}")
        if wiki_chunks is not None:
            rescaled_l2 = mean_l2_norm_streaming(
                w_normed_b,
                wiki_chunks,
                gc_every_chunks=gc_every,
                max_tokens=max_tokens,
            )
        else:
            rescaled_l2 = mean_l2_norm(w_normed_b, all_ids)
        print(f"  After rescale     -> mean L2 norm: {rescaled_l2:.4f}")

    print("\n" + "=" * 72)
    print("NEXT STEP")
    print("=" * 72)
    print("The 'E2E target -> mean L2 norm' value above is what goes into")
    print(
        "improved-diffusion/improved_diffusion/text_datasets.py :: BERT_TARGET_CORPUS_NORMS"
    )
    print(f"for the relevant modality key.  Measured: {e2e_norm:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
