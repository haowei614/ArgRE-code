# Agent4Reqs: Technical Specification

This document supplements the [Methodology](methodology.md) with implementation-level details, CLI usage, and output format specifications.

## 1. Command Line Interface

```bash
# Single case run
uv run openre_bench --run-case \
  --case-input data/case_studies/ATM_input.json \
  --artifacts-dir artifacts/atm-run \
  --run-record artifacts/atm-run/run_record.json \
  --system quare

# Comparison matrix (all cases × all settings × all seeds)
uv run openre_bench --run-comparison-matrix \
  --cases-dir data/case_studies \
  --output-dir experiment_outputs/quare \
  --matrix-seeds 101,202,303

# End-to-end automated reporting
uv run openre_bench /auto \
  --cases-dir data/case_studies \
  --rag-corpus-dir data/knowledge_base \
  --matrix-seeds 101,202,303
```

### System Modes

| Flag | Architecture | Agents | Pipeline |
|---|---|---|---|
| `--system mare` | Task-specialized | Stakeholder, Collector, Modeler, Checker, Documenter | 9-action workflow |
| `--system quare` | Quality-specialized | Safety, Efficiency, Green, Trustworthiness, Responsibility | 5-phase pipeline |

## 2. Configuration

### LLM Access

| Source | Mechanism | Precedence |
|---|---|---|
| `.api_key` file | Raw key in project root | Highest |
| `OPENAI_API_KEY` env | Standard environment variable | Medium |
| `OPENAI_KEY` env | Fallback env variable | Lowest |

Additional environment variables:
- `OPENAI_MODEL` — Model name (default: `gpt-4o-mini`)
- `OPENAI_BASE_URL` — Custom API endpoint

### Runtime Parameters

| Parameter | Default | Description |
|---|---|---|
| Temperature | 0.7 | LLM sampling temperature |
| Max Tokens | 4,000 | Maximum tokens per LLM response |
| Round Cap | 3 | Maximum negotiation rounds |
| RAG Backend | `local_tfidf` | Knowledge retrieval backend |

## 3. Output Artifacts

### Per-Run Phase Artifacts

| File | Phase | Content |
|---|---|---|
| `phase1_initial_models.json` | Generation | Per-agent KAOS models with RAG context |
| `phase2_negotiation_trace.json` | Negotiation | Full negotiation message history |
| `phase3_integrated_kaos_model.json` | Integration | Merged KAOS model with topology validation |
| `phase4_verification_report.json` | Verification | Compliance scores and structural checks |
| `run_record.json` | Metadata | Provenance, timing, execution flags |

### Matrix-Level Outputs

| File | Format | Content |
|---|---|---|
| `comparison_runs.jsonl` | JSONL | One record per run with all metrics |
| `comparison_metrics_by_case.csv` | CSV | Per-case metric breakdown |
| `comparison_metrics_summary.csv` | CSV | Aggregate statistics |
| `comparison_ablation_table.csv` | CSV | Setting-by-setting ablation results |
| `comparison_validity_log.md` | Markdown | Validation status and error log |

## 4. Evaluation Metrics

### RQ1: Coverage and Diversity

| Metric | Description |
|---|---|
| **Requirement Count** | Number of KAOS elements generated per phase |
| **CHV** (Convex Hull Volume) | Volume of the convex hull in 5D quality space — measures coverage breadth |
| **MDC** (Mean Distance to Centroid) | Average distance of requirements from centroid — measures dispersion |

### RQ2: Negotiation Effectiveness

| Metric | Description |
|---|---|
| **BERTScore F1** (P3 vs P1) | Semantic similarity between final and initial requirements |
| **BERTScore F1** (P2 vs P1) | Semantic retention during negotiation |
| **Negotiation Steps** | Number of structured messages exchanged |
| **CRR** (Conflict Resolution Rate) | Percentage of detected conflicts resolved |

### RQ3: Structural Validity and Compliance

| Metric | Description |
|---|---|
| **DAG Topology** | Whether the model forms a valid directed acyclic graph |
| **S_logic** | Logical consistency score (0–1) |
| **Compliance Coverage** | Percentage of relevant standard clauses addressed |
| **ISO 29148 Scores** | Unambiguity, correctness, verifiability, set consistency, set feasibility (1–5 Likert) |

## 5. Ablation Settings

The comparison matrix supports four ablation settings to isolate component contributions:

| Setting | Agents | Negotiation | Verification | Purpose |
|---|---|---|---|---|
| `single_agent` | 1 | No | No | Baseline |
| `multi_agent_without_negotiation` | 5 | No | No | Isolate multi-agent contribution |
| `multi_agent_with_negotiation` | 5 | Yes | No | Isolate negotiation contribution |
| `negotiation_integration_verification` | 5 | Yes | Yes | Full pipeline |

## 6. Reproducibility Controls

Each run records:
- **Model and parameters** — Model name, temperature, max tokens, seed
- **RAG state** — Backend type, corpus directory, fallback status
- **Execution flags** — `execution_mode`, `fallback_tainted`, `retry_count`
- **Provenance hashes** — SHA-256 of input case and output artifacts
- **Timing** — Per-phase and total wall-clock duration

Runs with `fallback_tainted=true` are flagged as invalid for strict comparability.
