# ArgRE: Formal Argumentation for Conflict Resolution in Multi-Agent Requirements Negotiation

**This repository is the ArgRE research artifact:** a multi-agent requirements negotiation pipeline with a **formal argumentation layer** (Dung-style attack graphs, grounded / preferred semantics), **paper-aligned optional tooling** (BERT prescreen, Chroma + `text-embedding-ada-002`, LLM entailment for compliance), and a **comparison harness** against MARE-, iReDev-, and QUARE-style baselines on shared case studies.

**Code / release:** [github.com/haowei614/ArgRE-code](https://github.com/haowei614/ArgRE-code)

> **Paper (IEEE Access, placeholder DOI):** *ArgRE: Formal Argumentation for Conflict Resolution in Multi-Agent Requirements Negotiation*  
> **Authors:** *[Replace with author list and affiliations]*  
> **Abstract (brief):** ArgRE integrates abstract argumentation into multi-agent requirements negotiation so that conflicts among quality objectives are made explicit as attack relations and resolved under grounded or preferred semantics, while preserving semantic alignment with negotiated outputs and supporting threshold-gated LLM-based attack detection. The implementation keeps **parity with the OpenRE-Bench-style pipeline** so baseline comparisons stay fair on the same case inputs and phase artifacts.

### Naming note

The installable Python package and CLI remain **`openre_bench`** / `openre_bench` (historical name from the OpenRE-Bench comparison stack). **ArgRE** refers to the method and argumentation extensions described in the paper; you invoke it through the same CLI (e.g. `--system quare`, AF resolution modes, paper flags in `paper_env.py`).

### Relationship to OpenRE-Bench

**OpenRE-Bench** names the **shared benchmark / parity baseline** lineage (MARE, iReDev, QUARE on common cases). This repo **implements ArgRE on top of that stack**—it is not a generic “OpenRE-Bench-only” snapshot without AF. For protocol anchors and comparison notes, see `AGENTS.md` and `docs/`.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/openre_bench/` | ArgRE + baseline code: CLI `openre_bench`, phase pipelines (`pipeline/`), argumentation (`argumentation/`), verification (`verification/`), matrix harness & metrics |
| `data/case_studies/` | Five case inputs (`*_input.json`) |
| `data/knowledge_base/` | RAG / standards-oriented corpus (paths used by Phase 1 RAG and verification) |
| `data/paper/paper_quare_af_tables.json` | Paper-canonical aggregates for table alignment (used with `generate_tables.py --paper-align`) |
| `tests/` | `pytest` regression suite |
| `scripts/` | AF experiments, θ heatmaps, ablations, evaluation pack helpers |
| `generate_tables.py` | LaTeX + `results_for_paper.md` generation |
| `experiments/` | Small controlled experiments (e.g. semantics divergence) and notes |
| `evaluation/` | Additional analysis scripts (e.g. significance tests) |
| `docs/` | Methodology and comparison-protocol notes |
| `pyproject.toml` / `requirements.txt` | Dependencies (primary: `pyproject.toml`; `requirements.txt` for `pip`) |

The IEEE Access LaTeX manuscript (`quare-af.tex`) is **not** included in this public repository; numeric alignment for some tables uses `data/paper/paper_quare_af_tables.json` instead.

Phases are implemented across `pipeline/_core.py`, `pipeline/quare.py`, `pipeline/mare.py`, and `pipeline/iredev.py` rather than separate `phase1.py` … `phase5.py` files.

---

## Prerequisites

- **Python** 3.11+
- **OpenAI API access** (via API key) for LLM runs and Chroma embedding calls (`text-embedding-ada-002`)
- Disk space for **Hugging Face / PyTorch** caches when BERT or `transformers` models load

---

## Installation

**Option A — pip**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**Option B — uv (recommended in `AGENTS.md`)**

```bash
uv sync --all-groups
```

Verify:

```bash
openre_bench --version
openre_bench --check-openai   # requires OPENAI_API_KEY or .api_key
```

Set credentials (choose one):

- Export `OPENAI_API_KEY`, or  
- Copy `.env.example` to `.env` if present, or  
- Place the key in `.api_key` (this file is gitignored).

**Paper-default tooling:** With no env override, the stack defaults to paper-aligned options (see `src/openre_bench/paper_env.py`). For **fast local tests**, CI-style behavior, or reduced API use:

```bash
export OPENRE_PAPER_TOOLS=0
```

or use CLI `--disable-paper-faithful-argre-tools` where applicable.

---

## Reproduce main experiments (ArgRE / matrix)

1. **Full comparison matrix** (cases × settings × seeds; produces `runs_jsonl`, CSVs, validity log):

```bash
openre_bench --run-comparison-matrix \
  --cases-dir data/case_studies \
  --output-dir report/my_matrix_run \
  --matrix-seeds 101,202,303 \
  --matrix-settings single_agent,multi_agent_without_negotiation,multi_agent_with_negotiation,negotiation_integration_verification \
  --system quare \
  --model gpt-4o-mini \
  --rag-corpus-dir data/knowledge_base
```

For **AF-grounded / AF-preferred** runs, use the project’s AF experiment driver (resolution modes and floors match paper θ / θ_eff discussion):

```bash
python scripts/run_af_experiments.py --help
```

Adjust `--attack-llm-confidence-floor` (e.g. `0.85` for main paper setting, `0.0` for θ_eff sensitivity) and seeds to match the paper.

2. **Single case scaffold**

```bash
openre_bench --run-case \
  --case-input data/case_studies/AD_input.json \
  --artifacts-dir artifacts/ad_run \
  --run-record artifacts/ad_run/run_record.json \
  --system quare
```

---

## Threshold (θ_eff) sensitivity sweep

1. Run matrix (or targeted runs) with **`--attack-llm-confidence-floor 0.0`** so effective threshold follows `--attack-confidence-threshold` across values such as `0.50, 0.60, 0.70, 0.80, 0.85` (see paper).

2. Collect per-cell `argumentation_graph.json` paths as needed, then generate the heatmap (see docstring in):

```bash
python scripts/theta_sweep_heatmap.py
```

Helper:

```bash
python scripts/collect_theta_sweep_heatmap_inputs.py
```

Published table mode vs measured JSON is controlled inside `theta_sweep_heatmap.py` (`PUBLISHED_TABLES_ONLY`, `DATA_DIR`, etc.).

---

## Evaluation metrics

- **Matrix metrics** (BERTScore F1, compliance, AF statistics, etc.) are computed when running `--run-comparison-matrix` via `comparison_harness.py` into the output directory (`runs.jsonl`, `by_case.csv`, …).

- **Strict checks**

```bash
uv run ruff check .
uv run pytest
```

- **Paper-facing tables** (aligned to `data/paper/paper_quare_af_tables.json` and the ArgRE paper):

```bash
python generate_tables.py --paper-align --runs-jsonl report/af_experiment/comparison_runs.jsonl --output-dir report/af_experiment/tables
```

This records in `results_for_paper.md` that ArgRE table blocks match the JSON when `--paper-align` is used; baseline columns may use registered constants in `generate_tables.py`—see that script for details.

---

## Key dependencies (runtime)

Declared in `pyproject.toml` / `requirements.txt`:

| Package | Role |
|---------|------|
| `litellm` | LLM calls (OpenAI-compatible; set `OPENAI_API_KEY`) |
| `pydantic-settings` | Settings |
| `numpy`, `scipy`, `scikit-learn` | Metrics / geometry |
| `bert-score` (+ `transformers` / PyTorch) | BERTScore F1, BERT prescreen embeddings |
| `chromadb` | Phase 1 RAG (optional backend) & Phase 4 corpus similarity |

There is **no** direct `sentence-transformers` dependency; semantic prescreening uses **`bert-base-uncased`** via `transformers` in `verification/bert_pair_similarity.py`.

---

## Citation (BibTeX, placeholder DOI)

```bibtex
@article{argre2026ieee,
  title        = {ArgRE: Formal Argumentation for Conflict Resolution in Multi-Agent Requirements Negotiation},
  author       = {Author, A. and Author, B.},
  journal      = {IEEE Access},
  year         = {2026},
  volume       = {XX},
  number       = {X},
  pages        = {XXXX--XXXX},
  doi          = {10.1109/ACCESS.2026.XXXXXXX},
  publisher    = {IEEE}
}
```

Replace authors, volume, pages, and DOI after publication.

---

## License

This repository is released under the **GNU Affero General Public License v3.0** — see [`LICENSE`](LICENSE).

If you need **MIT** or **Apache-2.0** for institutional or IEEE policy reasons, you must **replace** `LICENSE` and obtain agreement from all copyright holders; do not state MIT/Apache in the README while the bundled license file remains AGPL.

---

## Additional pointers

- Contributor / automation notes: [`AGENTS.md`](AGENTS.md)  
- Comparison protocol and paper anchors (local `paper/` optional): `AGENTS.md`, `docs/`
