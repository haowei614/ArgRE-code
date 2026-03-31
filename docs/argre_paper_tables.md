# ArgRE — Main paper tables (published numbers)

**Snapshot for the ArgRE manuscript (IEEE Access).** Values below match the tables reported in the paper; they are **not** regenerated on each `git clone`. For scriptable LaTeX fragments, run `generate_tables.py` (optional) with `data/paper/paper_quare_af_tables.json` and `--paper-align` where applicable.

---

## Run configuration (main experiments)

| Field | Value |
| --- | --- |
| **LLM** | `gpt-4o-mini-2024-07-18` |
| **Temperature** | 0.7 (attack-detection path: 1.0 where noted in paper) |
| **Round cap** | 3 |
| **Seeds** | 101, 202, 303 |
| **Cases** | AD, ATM, Library, RollCall, Bookkeeping |
| **$\theta_{\mathrm{eff}}$ (cross-pair LLM attacks)** | 0.85 (main runs: $\mathcal{R}_{\mathrm{llm}}=\emptyset$) |

> [!NOTE]
> **Provenance**
>
> - **Table 4** (graph stats), **ArgRE(G)/(P) per-case BERTScore** fragments, and **Complex downstream** aggregates are aligned with `data/paper/paper_quare_af_tables.json` (canonical JSON for reporting).
> - **Tables 6–8** include **MARE / iReDev / ArgRE-NoAF** columns as in the paper; baseline columns may also be mirrored in `generate_tables.py` (`BASELINE_*`) when generating `.tex`.
> - **Table 5 (DJS)** is from the **human evaluation** (three raters); raw scorer sheets are **not** in this repository.

---

## Table 4 — Argumentation graph statistics (Basic vs Complex conflict)

Averaged over seeds 101, 202, 303. **Basic:** pairwise protocol only (GCI = 0). **Complex conflict:** + cross-pair arbitration (GCI ≈ 0.2–0.3 per case; avg 0.25).

| Case | $\|A\|$ B | $\|R_{att}\|$ B | $\|E_g\|$ B | $\|E_p\|$ B | TC % B | $\|A\|$ C | $\|R_{att}\|$ C | $\|E_g\|$ C | $\|E_p\|$ C | TC % C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AD | 25.3 | 12.0 | 14.0 | 14.0 | 19.0 | 27.0 | 14.0 | 11.0 | 15.3 | 21.2 |
| ATM | 17.3 | 7.0 | 11.3 | 11.3 | 53.0 | 19.0 | 8.5 | 8.7 | 12.0 | 55.5 |
| Library | 22.0 | 10.7 | 13.3 | 13.3 | 30.4 | 23.7 | 12.0 | 10.3 | 14.0 | 32.1 |
| RollCall | 24.0 | 13.3 | 13.3 | 13.3 | 30.2 | 25.3 | 15.0 | 9.7 | 14.7 | 31.8 |
| Bookkeeping | 13.3 | 3.0 | 10.3 | 10.3 | 71.9 | 14.0 | 3.5 | 8.3 | 11.0 | 72.5 |
| **Average** | **20.4** | **9.2** | **12.4** | **12.4** | **40.9** | **21.8** | **10.6** | **9.6** | **13.4** | **42.6** |

**Topology (footnote-style):** depth Basic 1.9 / Complex 2.1; weakly connected components Basic 7.3 / Complex 7.5; attack mix P1 / P2 / P3 ≈ **85% / 9% / 6%**.

---

## Table 5 — Decision Justification Score (DJS), 1–5 (↑)

Mean ± SD over three raters; $n$ = requirements per cell. **Paired:** present in both ArgRE and NoAF outputs. **B-only:** accepted under NoAF but not under ArgRE.

| Case | ArgRE | NoAF | $n_{\mathrm{pair}}$ | NoAF (B-only) | $n$ |
| --- | --- | --- | ---: | --- | ---: |
| AD | 4.27 ± 0.70 | 3.13 ± 0.74 | 5 | 3.03 ± 0.89 | 12 |
| ATM | 4.27 ± 0.64 | 3.13 ± 1.07 | 10 | 2.67 ± 1.15 | 1 |
| Bookkeeping | 4.44 ± 0.51 | 2.61 ± 1.04 | 6 | — | 0 |
| **Overall** | **4.32 ± 0.62** | **3.07 ± 1.01** | **21** | **3.03 ± 0.90** | **13** |

**Tests:** Wilcoxon paired ArgRE > NoAF: $p < 0.001$, Cliff’s $\delta = 0.92$ (large). Krippendorff’s $\alpha = 0.33$ (ordinal).

---

## Table 6 — Semantic preservation (BERTScore F1, Phase 3 vs Phase 1, ↑)

| Case | MARE | iReDev | ArgRE-NoAF | ArgRE (G) | ArgRE (P) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AD | 88.4 | 92.7 | 94.8 | 94.9 | 94.6 |
| ATM | 89.6 | 92.0 | 95.5 | 95.1 | 95.3 |
| Library | 87.7 | 92.6 | 94.8 | 95.0 | 95.1 |
| RollCall | 88.8 | 93.0 | 94.5 | 94.4 | 94.4 |
| Bookkeeping | 90.5 | 92.9 | 94.8 | 95.0 | 95.0 |
| **Average** | **89.0** | **92.6** | **94.9** | **94.9** | **94.7** |

---

## Table 7 — Structural validity and compliance

| Metric | MARE | iReDev | ArgRE-NoAF | ArgRE (G) | ArgRE (P) |
| --- | --- | --- | --- | --- | --- |
| DAG valid | ✓ | ✓ | ✓ | ✓ | ✓ |
| $S_{\mathrm{logic}}$ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Compliance (%) | 47.6 | 47.8 | 98.2 | 84.7 | 84.7 |
| Verifiability | 3.95 | 3.96 | 4.96 | 4.70 | 4.60 |
| Feasibility | 3.74 | 3.75 | 4.96 | 4.60 | 4.50 |

---

## Table 8 — ISO/IEC/IEEE 29148 quality scores (1–5)

| Criterion | MARE | iReDev | ArgRE-NoAF | ArgRE (G) | ArgRE (P) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unambiguous | 4.41 | 4.19 | 4.24 | 4.20 | 4.30 |
| Correctness | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 |
| Verifiability | 3.95 | 3.96 | 4.96 | 4.70 | 4.60 |
| Consistency | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 |
| Feasibility | 3.74 | 3.75 | 4.96 | 4.60 | 4.50 |

---

## Appendix — Downstream quality (Basic vs Complex conflict scenario)

Same seed average as in the paper (5 cases × 3 seeds); values from `paper_quare_af_tables.json` → `complex_downstream`.

| Metric | Basic | Complex (G) | Complex (P) |
| --- | ---: | ---: | ---: |
| BERTScore (%) | 94.9 | 94.6 | 94.8 |
| Compliance (%) | 84.7 | 78.3 | 86.2 |
| Verifiability (1–5) | 4.70 | 4.55 | 4.65 |
| Feasibility (1–5) | 4.60 | 4.45 | 4.55 |
</think>


<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace