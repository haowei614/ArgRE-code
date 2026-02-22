# Repository Guidelines

This repository is paper-first: implementation and documentation must follow the PDFs in `paper/`, and markdown must be corrected whenever drift appears.

## Project Structure & Module Organization

- `paper/` — paper artifacts and comparison protocol notes.
- `src/openre_bench/` — executable implementation scaffold (CLI, pipeline, validator, matrix harness, `/auto`).
- `tests/` — regression coverage for runtime behavior and report/finality gates.
- `report/` — generated comparison reports and evidence artifacts.
- `pyproject.toml` — dependency and tooling configuration (`uv`, `ruff`, `pytest`).

Keep paper/comparison guidance in `paper/`. Keep executable logic in `src/openre_bench/`.

## Paper and Protocol Anchors

Comparison tasks against `../OpenRE-Bench` must use:

- `paper/submitting-paper-no-commit.pdf` (source of truth)
- `paper/OpenRE-Bench_Comparison_Protocol.md` (derived protocol)
- `paper/OpenRE-Bench_Analysis_Notes.md` (derived implementation notes)

Rules:

- If markdown conflicts with the paper, follow the paper and fix markdown.
- Log reproducibility controls per run: model, temperature, seeds, round cap, fallback flags, artifact paths.
- Keep I/O parity with OpenRE-Bench case inputs and phase artifact filenames.

## Implementation Reality Check

- Current runtime includes paper-faithful MARE 5-agent/9-action orchestration with LLM-driven multi-agent execution plus QUARE phase-2 LLM negotiation.
- Do not overstate scaffold outputs as full paper-faithful runtime reproduction quality.
- Preserve explicit comparability metadata and `N/A` handling for intentionally partial settings.

## Build, Test, and Development Commands

- `uv sync --all-groups`
- `uv run openre_bench --version`
- `uv run openre_bench --check-openai`
- `uv run openre_bench --run-case --case-input <case.json> --artifacts-dir <dir> --run-record <run.json> --system <mare|quare>`
- `uv run openre_bench --run-comparison-matrix --cases-dir <dir> --output-dir <dir> --matrix-seeds 101,202,303`
- `uv run openre_bench --export-trace-audit --matrix-output-dir <dir>`
- `uv run openre_bench --blind-eval-prepare --matrix-output-dir <dir> --blind-output-dir <dir> --judge-script src/openre_bench/comparison_harness.py`
- `uv run openre_bench /auto --cases-dir <dir> --rag-corpus-dir <dir> --matrix-seeds 101,202,303`
- `uv run ruff check .`
- `uv run pytest`

## Coding Style & Naming Conventions

- Python modules: `snake_case.py`; classes: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Use descriptive filenames and keep generated evidence under `report/`.
- Use `ruff` defaults with line length `100` unless a file-specific reason is documented.

## Testing Guidelines

- Add focused tests for behavior changes under `tests/`.
- For `/auto` changes, verify conversation-log completeness and finality-gate behavior.
- Validate changes with `uv run pytest` and include run evidence paths when reporting results.

## Security & Configuration

- LLM inference uses OpenAI credentials (`OPENAI_API_KEY` or `.api_key`).
- Never commit secrets.
- Use `.env.example` only for non-secret templates.

## Commit and PR Guidelines

- Use concise imperative commit subjects; optional colon-style gitmoji allowed (example: `:sparkles:`).
- Keep commits atomic and avoid unrelated file changes.
- PRs should include summary, rationale, exact artifact paths changed, and validation commands run.
