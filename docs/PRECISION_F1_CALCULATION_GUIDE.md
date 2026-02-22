# Precision and F1 Calculation Guide

This document describes the manual labeling procedure and metric calculations used to evaluate the quality of system-generated KAOS elements against ground-truth requirements.

## 1. Definitions

| Term | Definition |
|---|---|
| **TP (True Positive)** | Generated element that exists in the original requirements (explicitly or implicitly) |
| **FP (False Positive)** | Generated element that is fabricated or irrelevant to the original requirements |
| **FN (False Negative)** | Original requirement that the system failed to capture |
| **Precision** | TP / (TP + FP) — proportion of generated elements that are correct |
| **Recall** | TP / (TP + FN) — proportion of original requirements captured |
| **F1** | 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean |

### TP Subcategories

- **Explicit TP**: Directly stated in the original requirements document
- **Implicit TP**: Not explicitly stated, but necessarily implied by the system context

## 2. Phase Characteristics

| Phase | Content | Notes |
|---|---|---|
| Phase 1 (Generation) | Per-agent independent generation | Most elements; may contain redundancies |
| Phase 2 (Negotiation) | Post-negotiation elements | Conflicts and duplicates partially removed |
| Phase 3 (Integration) | Merged KAOS model | Structured final output; primary evaluation target |
| Phase 4 (Verification) | Verified model | Typically identical to Phase 3 |

## 3. Manual Labeling Procedure

### Step 1: Prepare Materials

For each case study, obtain:
- The original requirements input (`data/case_studies/<case>_input.json`)
- The generated Phase 3 output (`phase3_integrated_kaos_model.json`)

### Step 2: Label Each Element

For every generated KAOS element, compare against the original requirements:

| Judgment | Label | Example |
|---|---|---|
| Explicitly mentioned in original requirements | TP (explicit) | "Users can borrow books" when input says "support borrowing" |
| Not mentioned, but necessarily implied | TP (implicit) | "User authentication" for a library borrowing system |
| Fabricated or irrelevant | FP | "Online book purchases" when input says nothing about purchasing |

### Step 3: Calculate Metrics

After labeling all Phase 3 elements:

1. Map Phase 3 labels to elements appearing in Phases 1, 2, and 4 by name matching
2. Calculate TP, FP, Precision, and F1 per phase
3. Generate summary tables across all cases

### Example Output

```
Case            Phase       Elements   TP     FP     Precision    F1
========================================================================
AD              Phase 1     81         7      74     8.64%        14.29%
AD              Phase 2     36         6      30     16.67%       20.51%
AD              Phase 3     81         7      74     8.64%        14.29%
AD              Phase 4     81         7      74     8.64%        14.29%
```

## 4. Labeling Guidelines

### Calibration

- **Neither too lenient nor too strict**: Only truly relevant items should be TP; reasonable implicit requirements should also count
- **Consistency**: Apply the same standards across all cases
- **Document decisions**: Record labeling principles for reproducibility
- **Multiple raters**: Use inter-rater reliability (Cohen's Kappa) to validate consistency

### Expected Ranges

Based on requirements engineering literature:

| Metric | Typical Range | Notes |
|---|---|---|
| Precision | 5–20% | >50% suggests overly lenient labeling |
| F1 | 10–30% | Depends on Precision/Recall balance |
| Phase 2 vs Phase 1 | Slightly higher Precision | Negotiation removes obvious errors |

### Handling Boundary Cases

For ambiguous elements (e.g., "The system should have good performance"):

- If original requirements mention performance concerns → **TP (explicit)**
- If performance is clearly necessary for the system type → **TP (implicit)**
- If specific metrics are fabricated with no basis in original requirements → **FP**

## 5. Important Notes

**Automated labeling is insufficient.** Keyword-based classification (e.g., flagging any element containing "ASIL" or "ISO 26262" as TP) produces inflated Precision scores. Manual expert review is required for reliable evaluation.
