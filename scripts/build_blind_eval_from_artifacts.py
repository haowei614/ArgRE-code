#!/usr/bin/env python3
"""Build blind requirement-quality sheets from two artifact trees.

Example:
    python3 scripts/build_blind_eval_from_artifacts.py \
      --argre-runs-dir report/blind_requirement_quality_runs/argre_af_preferred/runs \
      --noaf-runs-dir report/blind_requirement_quality_runs/argre_noaf_original/runs \
      --output-dir report/blind_requirement_quality_formal
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_blind_pairwise import write_package_compare_script  # noqa: E402
from blind_pairwise_lib import (  # noqa: E402
    DEFAULT_SEED,
    PAIRWISE_COLUMNS,
    RequirementRecord,
    build_pair_diagnostics,
    build_pairs,
    build_pairwise_rows,
    collect_records,
    copy_run_evidence,
    write_csv,
    write_pairwise_readme,
)

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
    "System",
    "Setting",
    "Seed",
    "Run_ID",
    "Source_Requirement_ID",
    "Source_File",
    "Raw_Requirement_Text",
    "Cleaned_Requirement_Text",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build blind final-requirement quality sheets from ArgRE/NoAF artifacts."
    )
    parser.add_argument("--argre-runs-dir", required=True)
    parser.add_argument("--noaf-runs-dir", required=True)
    parser.add_argument("--output-dir", default="report/blind_requirement_quality_formal")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-copy-runs", action="store_true")
    args = parser.parse_args()

    argre_dir = Path(args.argre_runs_dir)
    noaf_dir = Path(args.noaf_runs_dir)
    if not argre_dir.exists():
        raise FileNotFoundError(f"ArgRE runs dir not found: {argre_dir}")
    if not noaf_dir.exists():
        raise FileNotFoundError(f"NoAF runs dir not found: {noaf_dir}")

    output_dir = Path(args.output_dir)
    if not args.skip_copy_runs:
        copy_run_evidence(
            argre_runs_dir=argre_dir,
            noaf_runs_dir=noaf_dir,
            output_runs_dir=output_dir / "runs",
        )

    argre_records = collect_records(argre_dir, method="ArgRE")
    noaf_records = collect_records(noaf_dir, method="ArgRE-NoAF")
    pairs, shortfalls = build_pairs(argre_records, noaf_records, seed=args.seed)
    pair_diagnostics = build_pair_diagnostics(pairs)
    blind_rows, mapping_rows = blind_pairs(pairs, seed=args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "blind_eval_sheet.csv", blind_rows, BLIND_COLUMNS)
    write_csv(output_dir / "private_reveal_mapping.csv", mapping_rows, MAPPING_COLUMNS)
    write_csv(
        output_dir / "pairwise_comparison.csv",
        build_pairwise_rows(pairs),
        PAIRWISE_COLUMNS,
    )
    write_json(
        output_dir / "private_reveal_mapping.json",
        {
            "metadata": {
                "argre_runs_dir": str(argre_dir),
                "noaf_runs_dir": str(noaf_dir),
                "random_seed": args.seed,
                "pair_count": len(pairs),
                "evaluator_row_count": len(blind_rows),
                "pair_diagnostics": pair_diagnostics,
            },
            "shortfalls": shortfalls,
            "mapping": mapping_rows,
        },
    )
    write_rubric(output_dir / "blind_eval_rubric.md")
    write_pairwise_readme(
        output_dir / "README.md",
        pairs=pairs,
        pair_diagnostics=pair_diagnostics,
    )
    write_package_compare_script(output_dir / "compare.py")
    shutil.copy2(SCRIPT_DIR / "blind_pairwise_lib.py", output_dir / "blind_pairwise_lib.py")

    print(f"Wrote formal blind package to {output_dir}")
    print(f"Pairs: {len(pairs)}; evaluator rows: {len(blind_rows)}")
    print(
        "Identical: "
        f"{pair_diagnostics['identical_pair_count']}; "
        f"different: {pair_diagnostics['different_pair_count']}"
    )
    if shortfalls:
        for item in shortfalls:
            print(
                f"Shortfall: {item['case_study']} requested {item['requested']} "
                f"available {item['available']}"
            )
    return 0


def blind_pairs(
    pairs: list[tuple[str, RequirementRecord, RequirementRecord]],
    *,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_rows: list[dict[str, str]] = []
    for pair_id, argre, noaf in pairs:
        source_rows.append(mapping_row(pair_id=pair_id, record=argre))
        source_rows.append(mapping_row(pair_id=pair_id, record=noaf))

    rng = random.Random(seed)
    rng.shuffle(source_rows)

    blind_rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []
    for index, row in enumerate(source_rows, start=1):
        blind_id = f"R{index:03d}"
        mapping_rows.append({"Blind_ID": blind_id, **row})
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


def mapping_row(*, pair_id: str, record: RequirementRecord) -> dict[str, str]:
    return {
        "Pair_ID": pair_id,
        "Case_Study": record.case_study,
        "Method": record.method,
        "System": record.system,
        "Setting": record.setting,
        "Seed": str(record.seed),
        "Run_ID": record.run_id,
        "Source_Requirement_ID": record.source_requirement_id,
        "Source_File": record.source_file,
        "Raw_Requirement_Text": record.raw_requirement_text,
        "Cleaned_Requirement_Text": record.cleaned_requirement_text,
    }


def write_rubric(path: Path) -> None:
    path.write_text(
        """# Blind Final Requirement Quality Rubric

Score each requirement text independently from 1 (poor) to 5 (excellent). Use only
the visible text. Do not infer the method that produced it.

- **Clarity**: understandable and free from avoidable ambiguity.
- **Completeness**: enough actor, condition, object, and expected behavior to be useful.
- **Verifiability**: testable, measurable, or checkable through observable evidence.
- **Consistency**: no internal contradiction or obvious conflict with the case context.
- **Feasibility**: realistic under normal engineering assumptions.
""",
        encoding="utf-8",
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
