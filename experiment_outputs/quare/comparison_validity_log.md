# Comparison Validity Log

- Generated at: 2026-02-16T18:39:07Z
- Cases dir: `data/case_studies`
- Output dir: `experiment_outputs/quare`
- Seeds: `[101, 202, 303]`
- Settings: `['single_agent', 'multi_agent_without_negotiation', 'multi_agent_with_negotiation', 'negotiation_integration_verification']`
- System: `quare`
- Model: `gpt-4o-mini`
- Temperature: `0.7`
- Round cap: `3`
- Max tokens: `4000`
- RAG enabled: `True`
- RAG backend: `local_tfidf`
- RAG corpus dir: `data/knowledge_base`

## Completeness
- Expected runs: 60
- Actual runs: 60

## Strict Fail Conditions
- Missing required deliverables.
- Run count mismatch versus expected matrix cardinality.
- Missing required keys in `comparison_runs.jsonl`.
- Missing or malformed provenance metadata (hash contracts).
- Missing or malformed execution/comparability metadata contracts.
- Missing required columns in required CSV outputs.
- Any run with `validation_passed=false`.
- Any run with `rag_enabled=false` or missing `rag_backend`.
- Any run with `rag_fallback_used=true`.
- Any run with `fallback_tainted=true` or retry-tainted metadata.

## Errors
- None

## Warnings
- None
