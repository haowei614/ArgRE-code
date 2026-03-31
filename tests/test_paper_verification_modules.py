"""Smoke tests for paper-aligned verification helpers (no network by default)."""

from __future__ import annotations

import pytest

from openre_bench.verification.llm_entailment import _parse_entailment_json


def test_parse_entailment_json_raw_object() -> None:
    raw = '{"entails": true, "rationale": "ok"}'
    parsed = _parse_entailment_json(raw)
    assert parsed is not None
    assert parsed.get("entails") is True


@pytest.mark.parametrize(
    "module",
    [
        "openre_bench.verification.bert_pair_similarity",
        "openre_bench.verification.chroma_hallucination",
        "openre_bench.verification.llm_entailment",
    ],
)
def test_verification_submodules_import(module: str) -> None:
    __import__(module)
