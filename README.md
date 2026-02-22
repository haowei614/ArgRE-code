# OpenRE-Bench

OpenRE-Bench is a paper-first implementation effort for **MARE (Multi-Agents Collaboration Framework for Requirements Engineering)** and QUARE-vs-MARE comparison experiments.

## Paper-First Rule

- `paper/submitting-paper-no-commit.pdf` is the single source of truth for comparison logic.
- `paper/2405.03256v1.pdf` is the source of truth for MARE architecture and semantics.
- If markdown and paper conflict, follow the paper and update markdown.

## Current Implementation Scope

- `src/openre_bench/` provides paper-faithful MARE runtime orchestration for the 5-agent/9-action workflow with LLM-driven multi-agent execution (and explicit fallback taint tracking) while preserving schema-compatible artifacts, validation, matrix runs, and reporting.
- `--system mare` and `--system quare` are both supported with strict provenance/comparability metadata.
- QUARE runs can execute live LLM turns in phase 2 negotiation when negotiation is enabled and OpenAI credentials are available; fallback state is recorded in execution flags.
- `/auto` provides resumable end-to-end orchestration with finality gates and per-agent conversation logs under `report/logs/<run_key>/`.
- MARE multi-agent settings execute all paper roles/actions with shared-workspace provenance captured in `run_record.notes.runtime_semantics`.

## Tech Stack

- Language/runtime: Python 3.11+
- Package and environment manager: `uv`
- Packaging: `pyproject.toml` (`hatchling`)
- LLM client layer: `litellm`
- LLM inference provider: OpenAI API key (`OPENAI_API_KEY`) or local `.api_key`
- Dev tools: `ruff`, `pytest` (managed through `uv`)

## Quick Start (uv)

1. `uv sync --all-groups`
2. `uv run openre_bench --version`
3. `cp .env.example .env` and set `OPENAI_API_KEY`, or create `.api_key` with your raw key
4. `uv run openre_bench --check-openai`
5. `uv run ruff check .`
6. `uv run pytest`

## OpenAI Configuration

OpenRE-Bench uses `litellm` and routes inference to OpenAI through:

- Key precedence: `.api_key` -> environment variables
- `OPENAI_API_KEY` (required when `.api_key` is absent)
- `OPENAI_KEY` (optional fallback env key name)
- `.api_key` (optional dotenv-style key file; highest precedence)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `OPENAI_BASE_URL` (optional custom endpoint)

Useful checks:

- `uv run openre_bench --check-openai`
- `uv run openre_bench --llm-ping`

## Comparison Anchors (QUARE vs OpenRE-Bench)

Use the paper first, then these derived guides:

- `paper/OpenRE-Bench_Analysis_Notes.md`
- `paper/OpenRE-Bench_Comparison_Protocol.md`
- `paper/fair-comparison-checklist.md`

These documents are implementation aids. They must be corrected whenever drift is found.

## Success Criteria Anchor (OpenRE-Bench I/O Parity)

For comparison implementation work, completion is anchored to OpenRE-Bench contract parity:

- Input parity: accept `case_name`, `case_description`, `requirement` from `../OpenRE-Bench/data/case_studies/*.json`-style inputs.
- Output parity: emit
  - `phase1_initial_models.json`
  - `phase2_negotiation_trace.json`
  - `phase3_integrated_kaos_model.json`
  - `phase4_verification_report.json`
- Evaluation parity: run under controlled settings and report RQ-aligned metrics with explicit comparability state.

## Core Commands

Validate one run:

`uv run openre_bench --validate-comparison --case-input <case.json> --run-record <run.json> --artifacts-dir <artifacts_dir>`

Run one case:

`uv run openre_bench --run-case --case-input ../OpenRE-Bench/data/case_studies/ATM_input.json --artifacts-dir artifacts/atm-run --run-record artifacts/atm-run/run_record.json --system mare`

Dual runtime identities:

- `--system mare`
- `--system quare`

## Comparison Matrix Harness

Run matrix experiments:

`uv run openre_bench --run-comparison-matrix --cases-dir ../OpenRE-Bench/data/case_studies --output-dir artifacts/smoke-matrix --matrix-seeds 101,202,303`

Default settings:

- `single_agent`
- `multi_agent_without_negotiation`
- `multi_agent_with_negotiation`
- `negotiation_integration_verification`

Key outputs:

- `comparison_runs.jsonl`
- `comparison_metrics_by_case.csv`
- `comparison_metrics_summary.csv`
- `comparison_ablation_table.csv`
- `comparison_validity_log.md`

## Trace Audit Export

`uv run openre_bench --export-trace-audit --matrix-output-dir artifacts/smoke-matrix`

Output:

- `comparison_trace_audit.md`

## `/auto` Strict Report Workflow

Run end-to-end resumable QUARE-vs-MARE reporting:

`uv run openre_bench /auto --cases-dir ../OpenRE-Bench/data/case_studies --rag-corpus-dir ../OpenRE-Bench/data/knowledge_base --matrix-seeds 101,202,303`

Important outputs:

- `report/logs/<run_key>/` execution logs
- `report/logs/<run_key>/conversation_index.jsonl`
- `report/logs/<run_key>/conversation_coverage.json`
- `report/logs/<run_key>/conversation_coverage.md`
- `report/logs/<run_key>/conversations/<system>/<case>/<setting>/seed-<seed>/<run_id>/`
- `report/runs/<run_key>/proofs/finality_threshold_verdict.json`
- `report/runs/<run_key>/proofs/conversation_log_evidence.json`
- `report/runs/<run_key>/proofs/quare_vs_mare_deltas.json`
- `report/README.md`, `report/analysis.md`, `report/proofs/*.json` (latest run mirror)

## Blind Evaluation Preparation

`uv run openre_bench --blind-eval-prepare --matrix-output-dir artifacts/smoke-matrix --blind-output-dir artifacts/smoke-matrix/blind-eval --judge-script src/openre_bench/comparison_harness.py`

Outputs:

- `blinded_comparison_runs.jsonl`
- `blinded_comparison_metrics_by_case.csv`
- `blind_mapping_private.json`
- `blind_eval_protocol.md`

## MARE Summary (Paper Target)

From `paper/2405.03256v1.pdf`, MARE defines:

- Four tasks: elicitation, modeling, verification, specification
- Five roles: `Stakeholders`, `Collector`, `Modeler`, `Checker`, `Documenter`
- Nine actions: `SpeakUserStories`, `ProposeQuestion`, `AnswerQuestion`, `WriteReqDraft`, `ExtractEntity`, `ExtractRelation`, `CheckRequirement`, `WriteSRS`, `WriteCheckReport`

This architecture is now enforced in MARE multi-agent runtime semantics and validated by guardrails/tests.

## Repository Layout

- `paper/` paper artifacts and comparison protocol notes
- `src/openre_bench/` implementation scaffold (CLI, pipeline, validator, matrix harness, `/auto`)
- `tests/` regression tests
- `report/` generated run/report evidence
- `ROADMAP.md` current milestone status and remaining gap

## Status

Current stage: **paper-first parity scaffold with strict reporting/finality gates**.

- Implemented: schema contracts, validator, matrix harness, `/auto` orchestration, conversation-log completeness gates, reproducibility proofs.
- Runtime fidelity status: MARE 5-agent/9-action semantics are implemented with contract-preserving outputs and strict guardrails.
