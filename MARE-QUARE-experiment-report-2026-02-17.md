# QUARE vs MARE: Experimental Comparison Report (Updated)

**Date**: 2026-02-26  
**Model**: gpt-4o-mini | **Temperature**: 0.7 | **Round Cap**: 3 | **Seeds**: 101, 202, 303  
**Total Runs**: 120 (60 MARE + 60 QUARE) | **Cases**: AD, ATM, Library, RollCall, Bookkeeping  
**Settings**: single_agent, multi_agent_without_negotiation, multi_agent_with_negotiation, negotiation_integration_verification

> [!NOTE]
> Data source is `experiment_outputs/mare` and `experiment_outputs/quare`. Validity snapshot: MARE valid/invalid = **60/0**, QUARE valid/invalid = **0/60**.

## 1. Requirement Volume (NIV)

| Case | MARE | QUARE |
|---|---:|---:|
| AD | 23.0 | **35.0** |
| ATM | 25.0 | **35.0** |
| Library | 24.3 | **35.0** |
| RollCall | 25.0 | **35.0** |
| Bookkeeping | 24.7 | **35.0** |
| **Average** | **24.4** | **35.0** |

## 2. CHV / MDC (NIV)

| Metric | MARE | QUARE |
|---|---:|---:|
| CHV | 0.00476 | 0.004309 |
| MDC | 0.835 | 0.672927 |

## 3. Semantics / Negotiation (NIV)

| Metric | MARE | QUARE |
|---|---:|---:|
| BERTScore P3 vs P1 | 89.0% | 94.8% |
| BERTScore P2 vs P1 | 97.0% | 100.0% |
| Conflict Resolution Rate | 66.7% | 30.6% |

## 4. Compliance / Runtime (NIV)

| Metric | MARE | QUARE |
|---|---:|---:|
| Compliance Coverage | 47.6% | 97.1% |
| Runtime | 38.4s | 48.6s |

## 5. Runtime by Setting

| Setting | MARE (s) | QUARE (s) |
|---|---:|---:|
| single_agent | 0.005 | 0.003 |
| multi_agent_without_negotiation | 39.634 | 0.007 |
| multi_agent_with_negotiation | 35.210 | 59.983 |
| negotiation_integration_verification | 38.429 | 48.554 |

*This report is regenerated from latest local artifacts after removing fixed per-axis 7-count generation behavior.*
