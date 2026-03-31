"""Phase~2 Stage~2: LLM three-way classification of high-similarity requirement pairs."""

from __future__ import annotations

import json
import re
from typing import Any

from openre_bench.llm import LLMContract

_VALID_LABELS = frozenset({"redundant", "resource_bound", "logical_incompatibility", "none"})


def classify_pair_conflict_type_llm(
    *,
    text_a: str,
    text_b: str,
    llm_client: LLMContract,
    llm_model: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Assign one of redundant / resource_bound / logical_incompatibility / none."""

    payload = {
        "task": (
            "Classify the relationship between two candidate requirements from a negotiation pair."
        ),
        "requirement_a": text_a[:2500],
        "requirement_b": text_b[:2500],
        "labels": {
            "redundant": "Substantively duplicate or one subsumes the other.",
            "resource_bound": "Trade-off over finite resources (latency, energy, budget, compute).",
            "logical_incompatibility": "Mutually exclusive states or impossible to satisfy both as stated.",
            "none": "No substantive conflict under these definitions.",
        },
        "output_schema": {
            "label": "one of redundant | resource_bound | logical_incompatibility | none",
            "confidence": "float in [0,1]",
            "rationale": "brief string",
        },
        "required_output_json": {"label": "none", "confidence": 0.0, "rationale": "..."},
        "model_note": llm_model,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object with keys label, confidence, rationale. "
                "Label must be one of: redundant, resource_bound, logical_incompatibility, none. "
                "No markdown fences."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    raw = llm_client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    parsed = _parse_label_json(raw)
    label = "none"
    confidence = 0.0
    rationale = ""
    if isinstance(parsed, dict):
        raw_label = str(parsed.get("label", "none")).strip().lower().replace(" ", "_")
        if raw_label in _VALID_LABELS:
            label = raw_label
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        rationale = str(parsed.get("rationale", ""))[:800]
    return {
        "label": label,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": rationale,
        "raw_response_excerpt": raw[:400],
    }


def _parse_label_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
