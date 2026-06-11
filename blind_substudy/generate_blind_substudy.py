#!/usr/bin/env python3
"""
Blind DJS sub-study material generator.

Design:
- 10 pairs total: all 4 non-identical pairs (AD-P04, AD-P05, ATM-P03, ATM-P06)
  + 6 identical pairs sampled (AD 1, ATM 2, Bookkeeping 3), stratified to match
  the original DJS study case proportions (n = 5/10/6).
- Raters see ONLY the final requirement text (no argumentation graph, no trace
  cards, no method labels). Each unique text (per case study, after
  normalization) is shown ONCE; its score maps back to every (pair, condition)
  slot it occupies. This avoids asking raters to score the same string twice.
- Per-rater randomized item order; blind IDs carry no method information.

Reproduce: python3 generate_blind_substudy.py
"""
import csv
import json
import random
import re

SAMPLING_SEED = 20260611
RATER_ORDER_SEEDS = {"R1": 1101, "R2": 2202, "R3": 3303}

PAIRS = [
    # pair_id, case, argre_text, noaf_text, identical
    ("AD-P01", "AD",
     "It must execute an MRM to safely pull over or stop in a low-risk zone within a specific time window.",
     "It must execute an MRM to safely pull over or stop in a low-risk zone within a specific time window.", True),
    ("AD-P02", "AD",
     "Step 1 (E-step): Generate an optimal path in the Frenet Frame (SL-Graph).",
     "Step 1 (E-step): Generate an optimal path in the Frenet Frame (SL-Graph).", True),
    ("AD-P03", "AD",
     "ODD (Operational Design Domain): - The system operates in specific urban and highway scenarios.",
     "ODD (Operational Design Domain): - The system operates in specific urban and highway scenarios.", True),
    ("AD-P04", "AD",
     "[Conflict Scenario for Negotiation] - The SafetyAgent prioritizes \"Passive Safety\" and strictly enforcing safe distances and MRM execution.",
     "[System Context] The system is a Level 4 autonomous driving platform operating in mixed traffic environments.", False),
    ("AD-P05", "AD",
     "The EfficiencyAgent prioritizes \"Smoothness\" and \"Speed\" using the QP function to minimize Jerk.",
     "It must strictly adhere to the \"Apollo Pilot Safety Report\" and the \"EM Motion Planner\" specifications.", False),
    ("ATM-P01", "ATM",
     "If the saving-account balance is insufficient to cover the requested withdrawal amount, the application should inform the user and terminate the transaction.",
     "If the saving-account balance is insufficient to cover the requested withdrawal amount, the application should inform the user and terminate the transaction.", True),
    ("ATM-P02", "ATM",
     "Neither a checking-account nor a saving-account can have a negative balance.",
     "Neither a checking-account nor a saving-account can have a negative balance.", True),
    ("ATM-P03", "ATM",
     "The bank client must be able to deposit an amount to and withdraw an amount from his or her accounts using the bank application.",
     "If the saving-account balance is insufficient to cover the requested withdrawal amount, the application should inform the user and terminate the transaction.", False),
    ("ATM-P04", "ATM",
     "A bank client can have two types of accounts.",
     "A bank client can have two types of accounts.", True),
    ("ATM-P05", "ATM",
     "Recorded transactions must include the date, time, transaction type, amount and account balance after the transaction.",
     "Recorded transactions must include the date, time, transaction type, amount and account balance after the transaction.", True),
    ("ATM-P06", "ATM",
     "If the saving-account balance is insufficient to cover the requested withdrawal amount, the application should inform the user and terminate the transaction.",
     "The application should automatically withdraw funds from a related saving-account if the requested withdrawal amount on the checking-account is more than its current balance.", False),
    ("ATM-P07", "ATM",
     "Recorded transactions must include the date, time, transaction type, amount and account balance after the transaction.",
     "Recorded transactions must include the date, time, transaction type, amount and account balance after the transaction.", True),
    ("ATM-P08", "ATM",
     "Each transaction must be recorded, and the client must have the ability to review all transactions performed against a given account.",
     "Each transaction must be recorded, and the client must have the ability to review all transactions performed against a given account.", True),
    ("ATM-P09", "ATM",
     "A checking-account and a saving-account.",
     "A checking-account and a saving-account.", True),
    ("ATM-P10", "ATM",
     "For each checking account, one related saving-account can exists.",
     "For each checking account, one related saving-account can exists.", True),
    ("Bookkeeping-P01", "Bookkeeping",
     "A bookkeeping system must record financial transactions including income and expenses.",
     "A bookkeeping system must record financial transactions including income and expenses.", True),
    ("Bookkeeping-P02", "Bookkeeping",
     "A bookkeeping system must record financial transactions including income and expenses.",
     "A bookkeeping system must record financial transactions including income and expenses.", True),
    ("Bookkeeping-P03", "Bookkeeping",
     "The system must ensure double-entry bookkeeping principles are followed and maintain an audit trail of all transactions.",
     "The system must ensure double-entry bookkeeping principles are followed and maintain an audit trail of all transactions.", True),
    ("Bookkeeping-P04", "Bookkeeping",
     "The system must ensure double-entry bookkeeping principles are followed and maintain an audit trail of all transactions.",
     "The system must ensure double-entry bookkeeping principles are followed and maintain an audit trail of all transactions.", True),
    ("Bookkeeping-P05", "Bookkeeping",
     "The system must support multiple currencies and handle currency conversion.",
     "The system must support multiple currencies and handle currency conversion.", True),
    ("Bookkeeping-P06", "Bookkeeping",
     "Users must be able to generate financial reports such as balance sheets, income statements, and cash flow statements.",
     "Users must be able to generate financial reports such as balance sheets, income statements, and cash flow statements.", True),
]

NON_IDENTICAL = [p for p in PAIRS if not p[4]]
IDENTICAL_QUOTA = {"AD": 1, "ATM": 2, "Bookkeeping": 3}


def norm(t):
    return re.sub(r"\s+", " ", t.strip()).lower()


def main():
    rng = random.Random(SAMPLING_SEED)

    # --- Stratified sampling of identical pairs, preferring distinct texts ---
    selected = list(NON_IDENTICAL)
    for case, k in IDENTICAL_QUOTA.items():
        candidates = [p for p in PAIRS if p[4] and p[1] == case]
        # group by normalized text so duplicate-text pairs (e.g. Bookkeeping-P01/P02)
        # are not double-sampled
        groups = {}
        for p in sorted(candidates, key=lambda x: x[0]):
            groups.setdefault(norm(p[2]), []).append(p)
        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)
        chosen_keys = group_keys[:k]
        for key in chosen_keys:
            selected.append(groups[key][0])  # deterministic representative
    selected.sort(key=lambda p: p[0])
    assert len(selected) == 10, f"expected 10 pairs, got {len(selected)}"

    # --- Build unique blind items (per case, per normalized text) ---
    # text_key -> {case, text, slots: [(pair_id, condition)]}
    items = {}
    for pair_id, case, a_text, b_text, identical in selected:
        for cond, text in (("ArgRE", a_text), ("ArgRE-NoAF", b_text)):
            key = (case, norm(text))
            if key not in items:
                items[key] = {"case_study": case, "text": text, "slots": []}
            items[key]["slots"].append({"pair_id": pair_id, "condition": cond})

    item_list = sorted(items.values(), key=lambda d: (d["case_study"], norm(d["text"])))
    rng.shuffle(item_list)
    for i, it in enumerate(item_list, 1):
        it["blind_id"] = f"REQ-{i:02d}"

    # --- Outputs ---
    with open("selected_pairs.json", "w") as f:
        json.dump({
            "sampling_seed": SAMPLING_SEED,
            "design": "4 non-identical pairs (all) + 6 identical pairs stratified AD:1 ATM:2 Bookkeeping:3, duplicate-text pairs collapsed before sampling",
            "selected_pairs": [
                {"pair_id": p[0], "case_study": p[1],
                 "identical_after_normalization": p[4]}
                for p in selected
            ],
        }, f, indent=2)

    with open("reveal_mapping_substudy.json", "w") as f:
        json.dump({
            "note": "Reveal mapping from blind item IDs to (pair, condition) slots. Kept private from raters until scoring was complete; published here for reproducibility.",
            "sampling_seed": SAMPLING_SEED,
            "items": [
                {"blind_id": it["blind_id"], "case_study": it["case_study"],
                 "requirement_text": it["text"], "maps_to": it["slots"]}
                for it in sorted(item_list, key=lambda d: d["blind_id"])
            ],
        }, f, indent=2)

    for rater, seed in RATER_ORDER_SEEDS.items():
        order = list(item_list)
        random.Random(seed).shuffle(order)
        with open(f"blind_djs_sheet_{rater}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_no", "blind_id", "case_study",
                        "requirement_text", "djs_score_1_to_5", "comment"])
            for n, it in enumerate(order, 1):
                w.writerow([n, it["blind_id"], it["case_study"], it["text"], "", ""])

    n_items = len(item_list)
    n_slots = sum(len(it["slots"]) for it in item_list)
    print(f"pairs={len(selected)}  unique items={n_items}  pair-condition slots={n_slots}")
    for it in sorted(item_list, key=lambda d: d["blind_id"]):
        tag = ",".join(f"{s['pair_id']}/{s['condition'][:1]}" for s in it["slots"])
        print(f"  {it['blind_id']} [{it['case_study']}] -> {tag}")


if __name__ == "__main__":
    main()
