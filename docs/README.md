# OpenRE-Bench Documentation

This directory contains reference documentation for the OpenRE-Bench (Agent4Reqs) system.

## Contents

| Document | Description |
|---|---|
| [methodology.md](methodology.md) | Multi-agent architecture, KAOS modeling framework, pipeline phases, and data models |
| [technical_specification.md](technical_specification.md) | CLI usage, configuration, output formats, evaluation metrics, and reproducibility |
| [orchestrator_agent_description.md](orchestrator_agent_description.md) | Detailed description of the Orchestrator Agent's roles and responsibilities |
| [INCOSE_FORMAT_GUIDE.md](INCOSE_FORMAT_GUIDE.md) | Guide for converting KAOS elements to INCOSE-compliant requirement statements |
| [PRECISION_F1_CALCULATION_GUIDE.md](PRECISION_F1_CALCULATION_GUIDE.md) | Manual labeling and Precision/F1 calculation procedures for evaluation |

## Quick Start

```bash
# Install dependencies
uv sync --all-groups

# Verify setup
uv run openre_bench --version
uv run openre_bench --check-openai

# Run a single case
uv run openre_bench --run-case \
  --case-input data/case_studies/ATM_input.json \
  --artifacts-dir artifacts/atm-run \
  --run-record artifacts/atm-run/run_record.json \
  --system quare

# Run full comparison matrix
uv run openre_bench --run-comparison-matrix \
  --cases-dir data/case_studies \
  --output-dir experiment_outputs/quare \
  --matrix-seeds 101,202,303
```

For detailed CLI options and configuration, see the [Technical Specification](technical_specification.md).
