#!/usr/bin/env python3
"""Build a source-blinded final-requirement quality evaluation package.

The package is intended to support the reviewer-response experiment where
human evaluators see only final requirement texts, not traceability cards,
argumentation graphs, or method labels.

This script consumes the OpenRE human-evaluation Phase 3 export. It is useful
for auditing or for building a proxy package from existing outputs. For a
formal ArgRE(AF) vs ArgRE-NoAF study, prefer rebuilding from true AF/no-AF
artifacts when those artifacts are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = "/tmp/openre-review-materials/human_eval/phase3_requirements_all.csv"
DEFAULT_OUTPUT_DIR = "report/blind_requirement_quality"
DEFAULT_CASE_TARGETS = {
    "AD": 5,
    "ATM": 10,
    "Bookkeeping": 6,
}
DEFAULT_SEED = 20260607

ARGRE_SETTING = "negotiation_integration_verification"
NOAF_SETTING = "multi_agent_without_negotiation"
FRAMEWORK = "quare"


@dataclass(frozen=True)
class RequirementRecord:
    sample_id: str
    source_file: str
    run_id: str
    framework: str
    case_study: str
    setting: str
    seed: int
    source_requirement_id: str
    raw_requirement_text: str
    cleaned_requirement_text: str


@dataclass(frozen=True)
class PairedRequirement:
    pair_id: str
    case_study: str
    text_key: str
    argre: RequirementRecord
    noaf: RequirementRecord


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create blind final-requirement quality evaluation sheets."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}. Clone/export OpenRE human_eval data first."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_csv)
    pairs, shortfalls = build_pairs(records, seed=args.seed)
    blind_rows, mapping_rows = blind_pairs(pairs, seed=args.seed)

    write_csv(output_dir / "blind_eval_sheet.csv", blind_rows, BLIND_COLUMNS)
    write_csv(output_dir / "private_reveal_mapping.csv", mapping_rows, MAPPING_COLUMNS)
    write_json(
        output_dir / "private_reveal_mapping.json",
        {
            "metadata": build_metadata(input_csv=input_csv, seed=args.seed, pairs=pairs),
            "shortfalls": shortfalls,
            "mapping": mapping_rows,
        },
    )
    write_rubric(output_dir / "blind_eval_rubric.md")
    write_readme(output_dir / "README.md", input_csv=input_csv, pairs=pairs, shortfalls=shortfalls)

    print(f"Wrote blind package to {output_dir}")
    print(f"Pairs: {len(pairs)}; evaluator rows: {len(blind_rows)}")
    if shortfalls:
        print("Shortfalls:")
        for item in shortfalls:
            print(f"- {item['case_study']}: requested {item['requested']}, available {item['available']}")
    return 0


def load_records(path: Path) -> list[RequirementRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            RequirementRecord(
                sample_id=str(row.get("Sample_ID", "")).strip(),
                source_file=str(row.get("source_file", "")).strip(),
                run_id=str(row.get("run_id", "")).strip(),
                framework=str(row.get("framework", "")).strip().lower(),
                case_study=str(row.get("case_study", "")).strip(),
                setting=str(row.get("setting", "")).strip(),
                seed=to_int(row.get("seed", ""), default=0),
                source_requirement_id=str(row.get("source_requirement_id", "")).strip(),
                raw_requirement_text=str(row.get("raw_requirement_text", "")).strip(),
                cleaned_requirement_text=str(row.get("cleaned_requirement_text", "")).strip(),
            )
            for row in reader
        ]


def build_pairs(
    records: list[RequirementRecord],
    *,
    seed: int,
) -> tuple[list[PairedRequirement], list[dict[str, int | str]]]:
    rng = random.Random(seed)
    by_case_setting_text: dict[tuple[str, str, str], list[RequirementRecord]] = defaultdict(list)
    for record in records:
        if record.framework != FRAMEWORK:
            continue
        if record.case_study not in DEFAULT_CASE_TARGETS:
            continue
        if record.setting not in {ARGRE_SETTING, NOAF_SETTING}:
            continue
        key = normalized_text_key(record.cleaned_requirement_text)
        if not key:
            continue
        by_case_setting_text[(record.case_study, record.setting, key)].append(record)

    pairs: list[PairedRequirement] = []
    shortfalls: list[dict[str, int | str]] = []
    for case_study, target in DEFAULT_CASE_TARGETS.items():
        argre_keys = {
            key
            for c, setting, key in by_case_setting_text
            if c == case_study and setting == ARGRE_SETTING
        }
        noaf_keys = {
            key
            for c, setting, key in by_case_setting_text
            if c == case_study and setting == NOAF_SETTING
        }
        shared_keys = sorted(argre_keys & noaf_keys)
        rng.shuffle(shared_keys)
        selected_keys = shared_keys[:target]
        if len(selected_keys) < target:
            shortfalls.append(
                {
                    "case_study": case_study,
                    "requested": target,
                    "available": len(selected_keys),
                }
            )

        for index, key in enumerate(selected_keys, start=1):
            argre_record = choose_representative(
                by_case_setting_text[(case_study, ARGRE_SETTING, key)]
            )
            noaf_record = choose_representative(
                by_case_setting_text[(case_study, NOAF_SETTING, key)]
            )
            pairs.append(
                PairedRequirement(
                    pair_id=f"{case_study}-P{index:02d}",
                    case_study=case_study,
                    text_key=key,
                    argre=argre_record,
                    noaf=noaf_record,
                )
            )
    return pairs, shortfalls


def choose_representative(records: list[RequirementRecord]) -> RequirementRecord:
    return sorted(records, key=lambda item: (item.seed, item.run_id, item.source_requirement_id))[0]


BLIND_COLUMNS = (
    "Blind_ID",
    "Requirement_Text",
    "Clarity_1_5",
    "Completeness_1_5",
    "Verifiability_1_5",
    "Consistency_1_5",
    "Feasibility_1_5",
    "Notes",
)

MAPPING_COLUMNS = (
    "Blind_ID",
    "Pair_ID",
    "Case_Study",
    "Method",
    "Proxy_Definition",
    "Source_Framework",
    "Source_Setting",
    "Source_Seed",
    "Source_Run_ID",
    "Source_Requirement_ID",
    "Source_File",
    "Raw_Requirement_Text",
    "Cleaned_Requirement_Text",
)


def blind_pairs(
    pairs: list[PairedRequirement],
    *,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for pair in pairs:
        rows.extend(
            [
                mapping_row(
                    pair=pair,
                    method="ArgRE",
                    proxy_definition=f"{FRAMEWORK}:{ARGRE_SETTING}",
                    record=pair.argre,
                ),
                mapping_row(
                    pair=pair,
                    method="ArgRE-NoAF",
                    proxy_definition=f"{FRAMEWORK}:{NOAF_SETTING}",
                    record=pair.noaf,
                ),
            ]
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    mapping_rows: list[dict[str, str]] = []
    blind_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        blind_id = f"R{index:03d}"
        mapped = {"Blind_ID": blind_id, **row}
        mapping_rows.append(mapped)
        blind_rows.append(
            {
                "Blind_ID": blind_id,
                "Requirement_Text": row["Cleaned_Requirement_Text"],
                "Clarity_1_5": "",
                "Completeness_1_5": "",
                "Verifiability_1_5": "",
                "Consistency_1_5": "",
                "Feasibility_1_5": "",
                "Notes": "",
            }
        )
    return blind_rows, mapping_rows


def mapping_row(
    *,
    pair: PairedRequirement,
    method: str,
    proxy_definition: str,
    record: RequirementRecord,
) -> dict[str, str]:
    return {
        "Pair_ID": pair.pair_id,
        "Case_Study": pair.case_study,
        "Method": method,
        "Proxy_Definition": proxy_definition,
        "Source_Framework": record.framework,
        "Source_Setting": record.setting,
        "Source_Seed": str(record.seed),
        "Source_Run_ID": record.run_id,
        "Source_Requirement_ID": record.source_requirement_id,
        "Source_File": record.source_file,
        "Raw_Requirement_Text": record.raw_requirement_text,
        "Cleaned_Requirement_Text": record.cleaned_requirement_text,
    }


def build_metadata(*, input_csv: Path, seed: int, pairs: list[PairedRequirement]) -> dict[str, object]:
    return {
        "input_csv": str(input_csv),
        "random_seed": seed,
        "case_targets": DEFAULT_CASE_TARGETS,
        "pair_count": len(pairs),
        "evaluator_row_count": len(pairs) * 2,
        "method_proxy_definitions": {
            "ArgRE": f"{FRAMEWORK}:{ARGRE_SETTING}",
            "ArgRE-NoAF": f"{FRAMEWORK}:{NOAF_SETTING}",
        },
        "important_note": (
            "This package uses available QUARE phase-3 settings as proxies. "
            "If true AF-grounded/preferred artifacts are regenerated, rebuild this "
            "package from those artifacts before reporting as ArgRE(AF)."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_rubric(path: Path) -> None:
    path.write_text(
        """# Blind Final Requirement Quality Rubric

Evaluators should score each requirement text independently from 1 (poor) to 5 (excellent).
Do not infer or score the development method. Only the visible requirement text should be used.

## Criteria

- **Clarity**: Is the requirement understandable and free from avoidable ambiguity?
- **Completeness**: Does it capture enough condition, actor, object, and expected behavior to be useful?
- **Verifiability**: Can the requirement be tested, measured, or checked through observable evidence?
- **Consistency**: Does the requirement avoid internal contradiction or obvious conflict with the case context?
- **Feasibility**: Is the requirement realistic under normal engineering assumptions for the stated system?

## Scale

- **1**: Very poor; unusable without major rewriting.
- **2**: Weak; substantial ambiguity, missing information, or feasibility concerns.
- **3**: Acceptable but needs clarification or refinement.
- **4**: Good; mostly clear and useful with only minor issues.
- **5**: Excellent; clear, complete, testable, consistent, and feasible.
""",
        encoding="utf-8",
    )


def write_readme(
    path: Path,
    *,
    input_csv: Path,
    pairs: list[PairedRequirement],
    shortfalls: list[dict[str, int | str]],
) -> None:
    by_case: dict[str, int] = defaultdict(int)
    for pair in pairs:
        by_case[pair.case_study] += 1

    lines = [
        "# Blind Requirement Quality Evaluation Package (Proxy Draft)",
        "",
        "This package contains source-blinded final requirement texts for human scoring.",
        "",
        "## Files",
        "",
        "- `blind_eval_sheet.csv`: evaluator-facing sheet. Share this file with annotators.",
        "- `blind_eval_rubric.md`: scoring rubric to share with annotators.",
        "- `private_reveal_mapping.csv`: private source mapping. Do not share with annotators.",
        "- `private_reveal_mapping.json`: private mapping plus metadata.",
        "",
        "## Source",
        "",
        f"- Input CSV: `{input_csv}`",
        "- ArgRE proxy: `quare` / `negotiation_integration_verification`",
        "- ArgRE-NoAF proxy: `quare` / `multi_agent_without_negotiation`",
        "",
        "Important: this package uses available QUARE settings as proxies. If true AF-grounded",
        "or AF-preferred artifacts are regenerated, rebuild the package from those artifacts before",
        "claiming the rows are ArgRE(AF) outputs.",
        "",
        "Validation note: with the current OpenRE export, many paired cleaned texts are identical",
        "between the two proxy settings. Do not use this proxy package as the final human-scoring",
        "instrument unless you explicitly intend to evaluate duplicate-text agreement.",
        "",
        "## Pair Counts",
        "",
    ]
    for case_study in DEFAULT_CASE_TARGETS:
        lines.append(
            f"- {case_study}: {by_case.get(case_study, 0)} pairs "
            f"(target {DEFAULT_CASE_TARGETS[case_study]})"
        )
    lines.extend(["", f"- Total pairs: {len(pairs)}", f"- Evaluator rows: {len(pairs) * 2}"])
    if shortfalls:
        lines.extend(["", "## Shortfalls", ""])
        for item in shortfalls:
            lines.append(
                f"- {item['case_study']}: requested {item['requested']}, "
                f"available {item['available']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalized_text_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def to_int(value: object, *, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
