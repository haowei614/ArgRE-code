# Blind DJS Sub-Study Package

Implements the supervisor-requested blind redesign for Reviewer 3, Concern 1:
the same DJS scale is applied to final requirement text only, with no
argumentation artifacts shown for either condition.

## Design

- 10 pairs from the 21-pair text-level comparison: all 4 non-identical pairs
  (AD-P04, AD-P05, ATM-P03, ATM-P06) plus 6 identical pairs sampled with a
  fixed seed (20260611), stratified AD:1 / ATM:2 / Bookkeeping:3
  (AD-P01; ATM-P05, ATM-P08; Bookkeeping-P03, P05, P06).
- Duplicate-text pairs within a case were collapsed before sampling so the
  6 identical pairs carry 6 distinct texts.
- Each unique requirement text (per case, after whitespace/casing
  normalization) appears once as a blind item; 13 unique items cover all
  20 (pair, condition) slots. Raters never score the same string twice and
  cannot tell which items belong to which condition.
- Item order is shuffled per rater (seeds R1=1101, R2=2202, R3=3303).
- Scale: original DJS rubric (5 Fully Justified ... 1 No Justification),
  wording unchanged from the first study.

## Files

- `generate_blind_substudy.py` — reproduces all materials (deterministic).
- `rater_instructions.md` — give to each rater together with their sheet.
- `blind_djs_sheet_R1.csv` / `_R2.csv` / `_R3.csv` — one per rater.
- `reveal_mapping_substudy.json` — blind ID to (pair, condition)
  mapping published for reproducibility after scoring was completed.
- `selected_pairs.json` — sampling record.
- `analyze_blind_substudy.py` — run after sheets are returned; reports
  per-pair ArgRE vs ArgRE-NoAF means, mean difference, tie proportion,
  Wilcoxon (zeros discarded), and Krippendorff's alpha.

## Workflow

Status: completed. The committed `blind_djs_sheet_R*.csv` files are returned, filled rater sheets with `djs_score_1_to_5` completed for every blind item. Blank templates are intentionally not committed; they can be reproduced with `python3 generate_blind_substudy.py` using fixed seed 20260611.

To reproduce the analysis, run `python3 analyze_blind_substudy.py`. Report descriptive means as the primary result. Identical-text pairs tie
   by construction, so the Wilcoxon p-value on 10 pairs is weak by design;
   the manuscript argument rests on the collapse of the mean gap relative to
   the original 4.32 vs 3.07.

## Expected outcome and interpretation

- Gap collapses (expected): confirms the 17/21 finding — the original DJS
  difference reflects the explanatory value of argumentation artifacts, not
  requirement content. Report in Section VI-A3; revise the future-work
  sentence in Section VIII-A to a completed validation.
- Gap persists: ArgRE text itself is rated as better justified, which is a
  stronger DJS result. Either outcome is reportable.


## Results

Under text-only blind scoring the DJS gap collapsed from 1.25 (4.32 vs 3.07) to 0.00 (3.13 vs 3.13); 7/10 pairs tied (6 by construction); Wilcoxon p = 1.00; Krippendorff's ordinal alpha = 0.36.

The committed CSV files are the returned, filled rater sheets. Blank templates can be fully reproduced by `generate_blind_substudy.py` with seed 20260611.
