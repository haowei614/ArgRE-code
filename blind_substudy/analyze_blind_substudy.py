#!/usr/bin/env python3
"""
Analysis for the blind DJS sub-study. Run after the three rater sheets are
filled in (same filenames, djs_score_1_to_5 column completed).

Expands each unique-text item score to its (pair, condition) slots via
reveal_mapping_substudy.json, then compares ArgRE vs ArgRE-NoAF
per-pair mean DJS.

Statistics:
- Descriptive: per-condition mean/SD, per-pair differences.
- Wilcoxon signed-rank on the 10 pair-level differences. NOTE: pairs whose two
  texts are identical produce a difference of exactly 0 by construction; we
  report (a) the standard zero-discarding Wilcoxon, (b) the proportion of
  exact ties, and (c) descriptive means, and the manuscript should rely on the
  descriptive collapse of the gap rather than the p-value alone.
"""
import csv
import json
import statistics as st

RATERS = ["R1", "R2", "R3"]

mapping = json.load(open("reveal_mapping_substudy.json"))
items = {it["blind_id"]: it for it in mapping["items"]}

# blind_id -> rater -> score
scores = {bid: {} for bid in items}
for r in RATERS:
    with open(f"blind_djs_sheet_{r}.csv") as f:
        for row in csv.DictReader(f):
            v = row["djs_score_1_to_5"].strip()
            if not v:
                raise SystemExit(f"missing score: {r} {row['blind_id']}")
            scores[row["blind_id"]][r] = int(v)

# expand to pair/condition level, averaged over raters
pair_cond = {}  # (pair_id, condition) -> mean rater score
for bid, it in items.items():
    m = st.mean(scores[bid][r] for r in RATERS)
    for slot in it["maps_to"]:
        pair_cond[(slot["pair_id"], slot["condition"])] = m

pairs = sorted({p for p, _ in pair_cond})
diffs = []
print(f"{'pair':<18}{'ArgRE':>7}{'NoAF':>7}{'diff':>7}")
for p in pairs:
    a = pair_cond[(p, "ArgRE")]
    b = pair_cond[(p, "ArgRE-NoAF")]
    diffs.append(a - b)
    print(f"{p:<18}{a:>7.2f}{b:>7.2f}{a-b:>+7.2f}")

a_all = [pair_cond[(p, "ArgRE")] for p in pairs]
b_all = [pair_cond[(p, "ArgRE-NoAF")] for p in pairs]
print(f"\nArgRE      mean {st.mean(a_all):.2f}  SD {st.pstdev(a_all):.2f}")
print(f"ArgRE-NoAF mean {st.mean(b_all):.2f}  SD {st.pstdev(b_all):.2f}")
print(f"mean diff  {st.mean(diffs):+.2f}")
ties = sum(1 for d in diffs if d == 0)
print(f"exact ties (identical-text pairs): {ties}/{len(diffs)}")

try:
    from scipy.stats import wilcoxon
    nz = [d for d in diffs if d != 0]
    if len(nz) >= 1:
        stat, p = wilcoxon(a_all, b_all, zero_method="wilcox")
        print(f"Wilcoxon (zeros discarded): W={stat:.2f}, p={p:.4f}, n_nonzero={len(nz)}")
    else:
        print("All differences are zero; Wilcoxon not applicable.")
except ImportError:
    print("scipy not installed; install for Wilcoxon test (pip install scipy)")

def krippendorff_alpha_ordinal(data):
    """Compute Krippendorff's alpha with the ordinal distance function."""
    categories = [1, 2, 3, 4, 5]
    cat_index = {cat: index for index, cat in enumerate(categories)}
    coincidence = [[0.0 for _ in categories] for _ in categories]

    for unit_index in range(len(data[0])):
        values = [rater_scores[unit_index] for rater_scores in data]
        counts = {cat: values.count(cat) for cat in categories}
        n_values = sum(counts.values())
        if n_values <= 1:
            continue
        for left in categories:
            for right in categories:
                if left == right:
                    value = counts[left] * (counts[left] - 1) / (n_values - 1)
                else:
                    value = counts[left] * counts[right] / (n_values - 1)
                coincidence[cat_index[left]][cat_index[right]] += value

    marginals = [sum(row) for row in coincidence]
    total = sum(marginals)
    if total <= 1:
        return float("nan")

    def ordinal_delta(left_index, right_index):
        low, high = sorted((left_index, right_index))
        distance = sum(marginals[low : high + 1]) - (
            marginals[low] + marginals[high]
        ) / 2
        return distance * distance

    observed = (
        sum(
            coincidence[left][right] * ordinal_delta(left, right)
            for left in range(len(categories))
            for right in range(len(categories))
        )
        / total
    )
    expected = (
        sum(
            marginals[left] * marginals[right] * ordinal_delta(left, right)
            for left in range(len(categories))
            for right in range(len(categories))
        )
        / (total * (total - 1))
    )
    return 1 - (observed / expected)


# inter-rater reliability on the 13 blind items (Krippendorff's alpha, ordinal)
try:
    import krippendorff
    import numpy as np

    bids = sorted(items)
    data = [[scores[b][r] for b in bids] for r in RATERS]
    alpha = krippendorff.alpha(
        reliability_data=np.array(data, dtype=float), level_of_measurement="ordinal"
    )
except ImportError:
    bids = sorted(items)
    data = [[scores[b][r] for b in bids] for r in RATERS]
    alpha = krippendorff_alpha_ordinal(data)
print(f"Krippendorff's alpha (ordinal): {alpha:.2f}")
