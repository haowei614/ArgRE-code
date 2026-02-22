# MARE Implementation Roadmap

## Paper-First Policy

- `paper/2405.03256v1.pdf` and `paper/submitting-paper-no-commit.pdf` are the logic source of truth.
- Runtime and markdown must stay aligned with paper semantics and current code reality.
- Schema parity is necessary but not sufficient for paper-faithful claims.

## Current Snapshot

OpenRE-Bench currently provides a production-ready parity scaffold:

- CLI orchestration in `src/openre_bench/cli.py`
- Contract schemas in `src/openre_bench/schemas.py`
- MARE multi-agent LLM-driven phase artifact generation (with explicit fallback tainting) in `src/openre_bench/pipeline.py`
- Validation gates in `src/openre_bench/comparison_validator.py`
- Matrix execution and metrics in `src/openre_bench/comparison_harness.py`
- End-to-end `/auto` reporting, resumability, and finality checks in `src/openre_bench/auto_report.py`

## Milestone Status

### M0: Environment and Contracts - Done

- `uv` environment, package wiring, and CLI entrypoint are complete.
- Case input, run record, and phase artifact contracts are implemented and validated.

### M1: Deterministic Parity Pipeline - Done

- `--run-case` emits protocol-compatible phase artifacts and run records.
- Run-level validation and strict metadata checks are active.

### M2: Matrix and Reporting Governance - Done

- `--run-comparison-matrix` produces by-run and aggregate metric deliverables.
- `--export-trace-audit` and `--blind-eval-prepare` are available.
- `/auto` generates finality verdicts, proof bundles, and conversation-log coverage evidence.

### M3: Paper-Faithful Runtime Semantics - Done

- MARE multi-agent runtime now executes paper-faithful 5-agent/9-action orchestration.
- Shared-workspace action provenance is persisted in `run_record.notes.runtime_semantics` and validated by guardrails.

### M4: Reproduction-Grade Evidence - In Progress

- Keep full-matrix reruns reproducible with explicit controls and proofs.
- Maintain strict GO/NO-GO thresholds and artifact-completeness gates.

### M5: Full Baseline Fidelity - Pending

- Close remaining gap between scaffold behavior and full MARE paper semantics.
- Reconfirm markdown/code/paper alignment after behavior-level closure.

## Exit Criteria

- Runtime behavior is paper-faithful, not only schema-compatible.
- Comparison outputs remain protocol-compatible and reproducible.
- No known markdown drift remains against paper and implementation reality.
