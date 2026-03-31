"""Paper-aligned verification helpers (ArgRE / IEEE Access methodology).

These modules implement optional tooling described in the paper: BERT-based
pairwise prescreening for negotiation, ChromaDB + ada-002 corpus similarity for
hallucination screening, and LLM entailment for compliance coverage. They are
disabled by default; enable via :class:`openre_bench.pipeline._core.PipelineConfig`
or CLI flags.
"""

from openre_bench.verification.bert_pair_similarity import pairwise_cosine_similarity_bert_uncased

__all__ = [
    "pairwise_cosine_similarity_bert_uncased",
]
