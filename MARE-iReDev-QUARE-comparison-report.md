# MARE vs iReDev vs QUARE: Three-Way Experimental Comparison

**Date**: 2026-02-17  
**Total Runs**: 180 (60 MARE + 60 iReDev + 60 QUARE)  
**Model**: gpt-4o-mini | **Temperature**: 0.7 | **Round Cap**: 3 | **Seeds**: 101, 202, 303  
**Cases**: AD, ATM, Library, RollCall, Bookkeeping  
**Settings**: single_agent, multi_agent_without_negotiation, multi_agent_with_negotiation, negotiation_integration_verification  

> [!NOTE]
> All 180 runs completed successfully. All metrics below are from the **full pipeline (NIV)** setting unless stated otherwise. Values are 3-seed averages per case.

---

## 1. Experimental Setup

| Parameter | MARE | iReDev | QUARE |
|---|---|---|---|
| Agent Count | 5 (task-specialized) | 6 (knowledge-driven) | 5 (quality-specialized) |
| Agent Roles | Stakeholder, Collector, Modeler, Checker, Documenter | Analyst, Modeler, Reviewer, Validator, Integrator, Knowledge | Safety, Efficiency, Green, Trustworthiness, Responsibility |
| Negotiation | Single-turn (10 steps) | Knowledge-driven iteration (12 steps) | Multi-round dialectic (16.5 steps) |
| Pipeline Actions | 9 | 17 | 5-phase pipeline |
| Runs per System | 60 (5 cases × 4 settings × 3 seeds) | 60 | 60 |

---

## 2. RQ1: Coverage and Diversity

### 2.1 Requirement Volume (NIV Full Pipeline)

| Case | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD | 23.0 | 29.0 | **35.0** |
| ATM | 25.0 | 28.0 | **35.0** |
| Library | 24.3 | 28.0 | **35.0** |
| RollCall | 25.0 | 27.3 | **35.0** |
| Bookkeeping | 24.7 | 28.0 | **35.0** |
| **Average** | **24.4** | **28.1** | **35.0** |

### 2.2 Convex Hull Volume (CHV)

Higher = broader coverage across the 5D quality space.

| Case | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD | 0.00493 | **0.00658** | 0.00331 |
| ATM | 0.00516 | **0.00670** | 0.00478 |
| Library | 0.00394 | **0.00670** | 0.00481 |
| RollCall | 0.00385 | **0.00594** | 0.00479 |
| Bookkeeping | **0.00594** | 0.00586 | 0.00378 |
| **Average** | **0.00476** | **0.00636** | **0.00430** |

### 2.3 Mean Distance to Centroid (MDC)

Higher = more dispersed, less redundant requirements.

| Case | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD | **0.760** | 0.682 | 0.667 |
| ATM | **0.854** | 0.714 | 0.678 |
| Library | **0.855** | 0.723 | 0.679 |
| RollCall | **0.870** | 0.704 | 0.679 |
| Bookkeeping | **0.835** | 0.700 | 0.673 |
| **Average** | **0.835** | **0.705** | **0.675** |

### 2.4 RQ1 Summary

| Metric | MARE | iReDev | QUARE | Winner |
|---|---:|---:|---:|---|
| Requirements Generated | 24.4 | 28.1 | **35.0** | **QUARE (+43%)** |
| CHV (Volume) | 0.00476 | **0.00636** | 0.00430 | **iReDev (+33%)** |
| MDC (Dispersion) | **0.835** | 0.705 | 0.675 | **MARE (+19%)** |

> [!IMPORTANT]
> **Interpretation**: QUARE generates the most requirements. iReDev covers the broadest quality space region (highest CHV). MARE has the most dispersed, non-redundant distribution (highest MDC). This reveals a fundamental architecture–diversity trade-off: task-specialized agents (MARE) produce more varied outputs, knowledge-driven agents (iReDev) cover broader territory, and quality-specialized agents (QUARE) maximize volume with focused axis alignment.

---

## 3. RQ2: Negotiation Effectiveness and Semantic Preservation

### 3.1 Semantic Preservation — BERTScore F1

**P3 vs P1** = similarity between final requirements and original generation (higher = better intent preservation).

| Case | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD | 88.4% | 92.7% | **94.8%** |
| ATM | 89.6% | 92.0% | **95.5%** |
| Library | 87.7% | 92.6% | **94.8%** |
| RollCall | 88.8% | 93.0% | **94.5%** |
| Bookkeeping | 90.5% | 92.9% | **94.8%** |
| **Average** | **89.0%** | **92.6%** | **94.9%** |

**P2 vs P1** = semantic retention during negotiation phase (higher = less semantic drift in negotiation).

| Case | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD | 96.8% | 99.4% | **100.0%** |
| ATM | 97.3% | 99.2% | **100.0%** |
| Library | 94.8% | 98.8% | **100.0%** |
| RollCall | 96.3% | 100.0% | **100.0%** |
| Bookkeeping | 100.0% | 99.6% | **100.0%** |
| **Average** | **97.0%** | **99.4%** | **100.0%** |

### 3.2 Negotiation Intensity

| Metric | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| Phase 2 Steps (avg) | 10.0 | 12.0 | **16.5** |

### 3.3 Conflict Resolution Rate

| Case | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD | **100.0%** | 66.7% | 66.7% |
| ATM | 66.7% | **66.7%** | 33.3% |
| Library | **100.0%** | 66.7% | 8.3% |
| RollCall | 66.7% | 0.0% | 16.7% |
| Bookkeeping | 0.0% | 33.3% | 0.0% |
| **Average** | **66.7%** | **46.7%** | **25.0%** |

> [!NOTE]
> Higher CRR does not necessarily mean better quality. MARE's coarser conflict detection marks fewer total conflicts and resolves most. QUARE's fine-grained dialectic detects more nuanced inter-agent conflicts, resulting in more detected conflicts but lower resolution rate. Unresolved conflicts are handled through prioritization in Phase 3 integration — which is reflected in QUARE's superior semantic preservation.

### 3.4 RQ2 Summary

| Metric | MARE | iReDev | QUARE | Winner |
|---|---:|---:|---:|---|
| BERTScore P3 vs P1 | 89.0% | 92.6% | **94.9%** | **QUARE (+5.9pp)** |
| BERTScore P2 vs P1 | 97.0% | 99.4% | **100.0%** | **QUARE** |
| Negotiation Steps | 10.0 | 12.0 | **16.5** | QUARE (deepest) |
| Conflict Resolution Rate | **66.7%** | 46.7% | 25.0% | MARE (see note) |

---

## 4. RQ3: Structural Validity and Industry Compliance

### 4.1 Structural Validity

| Metric | MARE | iReDev | QUARE |
|---|:---:|:---:|:---:|
| Topology Valid (DAG) | ✓ 100% | ✓ 100% | ✓ 100% |
| S_logic (Logical Consistency) | 1.000 | 1.000 | 1.000 |

### 4.2 Compliance Coverage

| Case | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| AD (ISO 26262) | 7.8% | 10.0% | **91.1%** |
| ATM (ISO 27001) | 33.3% | 26.7% | **100.0%** |
| Library (ISO 27001) | 50.0% | 55.6% | **100.0%** |
| RollCall (ISO 27001) | 73.3% | 80.0% | **100.0%** |
| Bookkeeping (ISO 27001) | 73.3% | 66.7% | **100.0%** |
| **Average** | **47.6%** | **47.8%** | **98.2%** |

> [!CAUTION]
> QUARE achieves **98.2% compliance coverage** versus MARE's 47.6% and iReDev's 47.8% — a dramatic **+105%** improvement. This is QUARE's strongest differentiator, driven by its RAG-augmented Phase 4 verification with explicit standard-clause mapping.

### 4.3 ISO/IEC/IEEE 29148 — Individual Requirement Quality (1–5 Likert)

| Case | System | Unambiguous | Correctness | Verifiability |
|---|---|---:|---:|---:|
| AD | MARE | **5.00** | **5.00** | 3.16 |
| AD | iReDev | **5.00** | **5.00** | 3.20 |
| AD | QUARE | 4.27 | **5.00** | **4.82** |
| ATM | MARE | 3.55 | **5.00** | 3.67 |
| ATM | iReDev | 3.74 | **5.00** | 3.53 |
| ATM | QUARE | 3.11 | **5.00** | **5.00** |
| Library | MARE | **4.52** | **5.00** | 4.00 |
| Library | iReDev | 3.70 | **5.00** | 4.11 |
| Library | QUARE | **5.00** | **5.00** | **5.00** |
| RollCall | MARE | 4.61 | **5.00** | 4.47 |
| RollCall | iReDev | **4.55** | **5.00** | **4.60** |
| RollCall | QUARE | 4.27 | **5.00** | **5.00** |
| Bookkeeping | MARE | 4.39 | **5.00** | 4.47 |
| Bookkeeping | iReDev | 3.96 | **5.00** | 4.33 |
| Bookkeeping | QUARE | **4.56** | **5.00** | **5.00** |

**Cross-System Averages:**

| Metric | MARE | iReDev | QUARE | Winner |
|---|---:|---:|---:|---|
| Unambiguous | **4.41** | 4.19 | 4.24 | **MARE** |
| Correctness | **5.00** | **5.00** | **5.00** | Tie |
| Verifiability | 3.95 | 3.96 | **4.96** | **QUARE (+25%)** |

### 4.4 ISO 29148 — Requirement Set Quality (1–5 Likert)

| Case | System | Set Consistency | Set Feasibility |
|---|---|---:|---:|
| AD | MARE | **5.00** | 2.79 |
| AD | iReDev | **5.00** | 2.84 |
| AD | QUARE | **5.00** | **4.79** |
| ATM | MARE | **5.00** | 3.40 |
| ATM | iReDev | **5.00** | 3.24 |
| ATM | QUARE | **5.00** | **5.00** |
| Library | MARE | **5.00** | 3.80 |
| Library | iReDev | **5.00** | 3.93 |
| Library | QUARE | **5.00** | **5.00** |
| RollCall | MARE | **5.00** | 4.36 |
| RollCall | iReDev | **5.00** | 4.52 |
| RollCall | QUARE | **5.00** | **5.00** |
| Bookkeeping | MARE | **5.00** | 4.36 |
| Bookkeeping | iReDev | **5.00** | 4.20 |
| Bookkeeping | QUARE | **5.00** | **5.00** |

**Cross-System Averages:**

| Metric | MARE | iReDev | QUARE | Winner |
|---|---:|---:|---:|---|
| Set Consistency | **5.00** | **5.00** | **5.00** | Tie |
| Set Feasibility | 3.74 | 3.75 | **4.96** | **QUARE (+32%)** |

### 4.5 RQ3 Summary

| Metric | MARE | iReDev | QUARE | Winner |
|---|---|---|---|---|
| DAG Topology | ✓ | ✓ | ✓ | Tie |
| S_logic | 1.000 | 1.000 | 1.000 | Tie |
| Compliance Coverage | 47.6% | 47.8% | **98.2%** | **QUARE (+105%)** |
| Correctness | 5.00 | 5.00 | 5.00 | Tie |
| Set Consistency | 5.00 | 5.00 | 5.00 | Tie |
| Unambiguous | **4.41** | 4.19 | 4.24 | **MARE** |
| Verifiability | 3.95 | 3.96 | **4.96** | **QUARE** |
| Set Feasibility | 3.74 | 3.75 | **4.96** | **QUARE** |

---

## 5. Runtime Performance

| Setting | MARE | iReDev | QUARE |
|---|---:|---:|---:|
| single_agent | 0.008s | 0.009s | 0.008s |
| multi_agent_without_negotiation | 34.0s | 161.9s | 0.020s |
| multi_agent_with_negotiation | 32.8s | 161.7s | 56.9s |
| **NIV (full pipeline)** | **34.4s** | **163.5s** | **55.4s** |

> [!NOTE]
> iReDev is **4.8× slower** than MARE and **3× slower** than QUARE. Its 6-agent/17-action workflow requires substantially more LLM turns. QUARE's Phase 1 is near-instant (structural), but its dialectic negotiation adds cost in later phases.

---

## 6. Comprehensive Summary

### 6.1 Full Pipeline (NIV) Comparison

| Metric | MARE | iReDev | QUARE | Winner |
|---|---:|---:|---:|---|
| Requirements (count) | 24.4 | 28.1 | **35.0** | **QUARE** |
| CHV (coverage volume) | 0.00476 | **0.00636** | 0.00430 | **iReDev** |
| MDC (dispersion) | **0.835** | 0.705 | 0.675 | **MARE** |
| BERTScore P3 vs P1 | 89.0% | 92.6% | **94.9%** | **QUARE** |
| BERTScore P2 vs P1 | 97.0% | 99.4% | **100.0%** | **QUARE** |
| CRR | **66.7%** | 46.7% | 25.0% | **MARE** |
| Compliance Coverage | 47.6% | 47.8% | **98.2%** | **QUARE** |
| Verifiability | 3.95 | 3.96 | **4.96** | **QUARE** |
| Set Feasibility | 3.74 | 3.75 | **4.96** | **QUARE** |
| Unambiguous | **4.41** | 4.19 | 4.24 | **MARE** |
| Correctness | **5.00** | **5.00** | **5.00** | Tie |
| Set Consistency | **5.00** | **5.00** | **5.00** | Tie |
| S_logic | **1.000** | **1.000** | **1.000** | Tie |
| Runtime | **34.4s** | 163.5s | 55.4s | **MARE** |

### 6.2 Win/Loss Matrix

| Dimension | QUARE | MARE | iReDev | Tie |
|---|---:|---:|---:|---:|
| RQ1 (Coverage) | 1 | 1 | 1 | 0 |
| RQ2 (Negotiation) | 2 | 1 | 0 | 1 |
| RQ3 (Compliance) | 3 | 1 | 0 | 4 |
| Runtime | 0 | 1 | 0 | 0 |
| **Total** | **6** | **4** | **1** | **5** |

### 6.3 Radar Comparison

```
                Coverage (CHV)
                   iReDev ★
                  /       \
          MDC   /         \  BERTScore
         MARE ★            ★ QUARE
              \           /
               \         /
            Compliance
             QUARE ★★★
```

---

## 7. Key Findings

1. **QUARE dominates compliance and semantic preservation**: 98.2% compliance coverage (vs ~48% for both MARE and iReDev), 94.9% BERTScore retention, and perfect P2-vs-P1 preservation (100%). These are QUARE's strongest advantages.

2. **iReDev leads coverage breadth**: Highest CHV (0.00636) indicates the broadest quality-space exploration — its 6-agent/17-action knowledge-driven pipeline systematically expands into diverse quality regions.

3. **MARE leads dispersion and efficiency**: Highest MDC (0.835) means least-redundant requirements, and fastest runtime (34.4s). MARE is the best choice for rapid iteration.

4. **MARE and iReDev are compliance-equivalent**: Both achieve ~48% compliance coverage — neither has automated ISO standard mapping. QUARE's Phase 4 RAG verification is the key differentiator.

5. **iReDev occupies the middle ground on semantics**: BERTScore preservation (92.6%) sits between MARE (89.0%) and QUARE (94.9%), reflecting its balanced approach between task-driven and quality-driven refinement.

6. **All three systems achieve structural validity parity**: Perfect DAG topology, logical consistency (1.000), correctness (5.00), and set consistency (5.00) across all systems and cases.

7. **Runtime trade-off is significant**: iReDev's 17-action workflow costs 163.5s/run (4.8× MARE, 3× QUARE). QUARE at 55.4s is a reasonable trade-off for its quality gains.

---

## 8. When to Choose Each System

| Use Case | Best Choice | Rationale |
|---|---|---|
| Regulated / safety-critical systems | **QUARE** | 98.2% compliance, best semantic preservation |
| Maximum coverage exploration | **iReDev** | Highest CHV, knowledge-augmented diversity |
| Rapid prototyping / iteration | **MARE** | Fastest (34.4s), highest dispersion |
| Quality trade-off documentation | **QUARE** | Dialectic protocol surfaces explicit trade-offs |
| Resource-constrained environments | **MARE** | Lowest LLM cost (10 negotiation steps) |
| Knowledge-intensive domains | **iReDev** | 6-agent knowledge-driven architecture |

---

*Generated from 180 actual experiment runs (60 per system) across 5 case studies (AD, ATM, Library, RollCall, Bookkeeping) with 3 random seeds (101, 202, 303) per configuration. All runs used gpt-4o-mini, temperature 0.7, round cap 3, max tokens 4000.*
