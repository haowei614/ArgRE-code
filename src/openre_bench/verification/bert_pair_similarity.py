"""BERT-base-uncased mean-pooled embeddings and cosine similarity (Phase 2 prescreen).

Matches the paper's Stage~1 description: pairwise cosine similarity on
``bert-base-uncased`` vectors. Used to flag high-similarity requirement pairs
before LLM-based classification.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

_MODEL_ID = "bert-base-uncased"
_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


def _get_model_and_tokenizer() -> tuple[Any, Any]:
    with _LOCK:
        if "model" in _CACHE and "tokenizer" in _CACHE:
            return _CACHE["model"], _CACHE["tokenizer"]
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        model = AutoModel.from_pretrained(_MODEL_ID)
        model.eval()
        _CACHE["tokenizer"] = tokenizer
        _CACHE["model"] = model
        return model, tokenizer


def _embed_mean_pooled(text: str) -> np.ndarray:
    import torch

    model, tokenizer = _get_model_and_tokenizer()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        vec = (summed / counts).squeeze(0).cpu().numpy()
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float64, copy=False)


def pairwise_cosine_similarity_bert_uncased(text_a: str, text_b: str) -> float:
    """Return cosine similarity between mean-pooled ``bert-base-uncased`` embeddings."""

    a = text_a.strip()
    b = text_b.strip()
    if not a or not b:
        return 0.0
    va = _embed_mean_pooled(a)
    vb = _embed_mean_pooled(b)
    return float(np.clip(np.dot(va, vb), -1.0, 1.0))
