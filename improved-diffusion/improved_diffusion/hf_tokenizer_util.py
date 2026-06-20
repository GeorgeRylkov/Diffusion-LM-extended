"""Hugging Face tokenizer loading with offline-friendly Hub snapshot resolution."""

from __future__ import annotations

import os
from typing import Any

# Short model ids map to canonical Hub repos under ~/.cache/huggingface/hub/models--* .
# ``bert-base-cased`` alone does not match the cached snapshot folder name from snapshot_download.
HF_TOKENIZER_REPO_ALIASES = {
    "bert-base-cased": "google-bert/bert-base-cased",
}


def auto_tokenizer_from_pretrained(pretrained_model_name_or_path: Any, **kwargs: Any):
    """Like AutoTokenizer.from_pretrained, but checks local Hub snapshot cache first."""
    from transformers import AutoTokenizer

    name = pretrained_model_name_or_path
    if not isinstance(name, str):
        return AutoTokenizer.from_pretrained(name, **kwargs)
    if os.path.isdir(name):
        return AutoTokenizer.from_pretrained(name, **kwargs)

    hub_id = HF_TOKENIZER_REPO_ALIASES.get(name, name)
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=hub_id,
            repo_type="model",
            local_files_only=True,
        )
        return AutoTokenizer.from_pretrained(local_dir, **kwargs)
    except Exception:
        pass

    return AutoTokenizer.from_pretrained(name, **kwargs)
