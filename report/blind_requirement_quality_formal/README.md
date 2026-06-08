# Text-Level Pairwise Comparison

## Purpose
Supplements Section VI-A3: verifies that ArgRE and ArgRE-NoAF
produce substantively identical requirement text under matched conditions.

## Experimental Setup
- Case studies: AD, ATM, Bookkeeping
- Seeds: 101, 202, 303
- Configurations: ArgRE (AF, preferred semantics), ArgRE-NoAF
- Total runs: 18

## Results
- Total paired requirements: 21
- Identical after normalization: 17 (81%)
- Minor lexical variation: 4 (19%)

Pairs with minor lexical variation: AD-P04, AD-P05, ATM-P03, ATM-P06.

## Files
- `pairwise_comparison.csv` — full 21-pair comparison table
- `compare.py` — normalization and matching script
- `runs/` — Phase 3 outputs for all 18 runs
- `blind_eval_sheet.csv` — annotator sheet with method labels withheld
- `private_reveal_mapping.json` — pair-to-run mapping for post-hoc analysis

## Reproduce
From this directory:

```bash
python3 compare.py
```
