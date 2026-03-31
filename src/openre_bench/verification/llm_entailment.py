"""LLM-based regulatory clause entailment (Phase 4 Layer 3, paper-aligned)."""

from __future__ import annotations

import json
import re
from typing import Any

from openre_bench.llm import LLMContract


def compliance_coverage_llm_entailment(
    *,
    clauses: list[str],
    requirement_texts: list[str],
    llm_client: LLMContract,
    llm_model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Return coverage payload parallel to ``_compliance_coverage`` using LLM entailment."""

    if not clauses:
        return {
            "satisfied_applicable_clauses": 0,
            "total_applicable_clauses": 0,
            "coverage_ratio": 0.0,
            "clauses": [],
            "method": "llm_entailment",
            "llm_model": llm_model,
        }

    satisfied = 0
    clause_results: list[dict[str, Any]] = []
    bundle = "\n".join(f"[{i}] {t[:1200]}" for i, t in enumerate(requirement_texts[:40]))

    for clause in clauses:
        payload = {
            "task": (
                "Decide whether ANY candidate requirement text logically entails or "
                "substantively satisfies the regulatory clause for compliance purposes."
            ),
            "clause": clause[:2000],
            "candidate_requirements_block": bundle,
            "output_schema": {"entails": "boolean", "rationale": "short string"},
            "required_output_json": {"entails": True, "rationale": "..."},
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Return exactly one JSON object with keys entails (boolean) and rationale "
                    "(string). No markdown fences."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        raw = llm_client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        parsed = _parse_entailment_json(raw)
        entails = bool(parsed.get("entails")) if isinstance(parsed, dict) else False
        if entails:
            satisfied += 1
        clause_results.append(
            {
                "clause": clause[:120],
                "matched": entails,
                "rationale": str(parsed.get("rationale", ""))[:500] if isinstance(parsed, dict) else "",
            }
        )

    total = len(clauses)
    ratio = satisfied / total if total else 0.0
    return {
        "satisfied_applicable_clauses": satisfied,
        "total_applicable_clauses": total,
        "coverage_ratio": round(ratio, 6),
        "clauses": clause_results,
        "method": "llm_entailment",
        "llm_model": llm_model,
    }


def _parse_entailment_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    fence = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if fence:
        text = fence.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
