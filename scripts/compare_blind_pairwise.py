#!/usr/bin/env python3
"""Generate pairwise ArgRE vs NoAF requirement text comparison.

Example:
    python3 scripts/compare_blind_pairwise.py \
      --argre-runs-dir report/blind_requirement_quality_formal/runs/argre_af_preferred \
      --noaf-runs-dir report/blind_requirement_quality_formal/runs/argre_noaf_original \
      --output report/blind_requirement_quality_formal/pairwise_comparison.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blind_pairwise_lib import (  # noqa: E402
    DEFAULT_SEED,
    PAIRWISE_COLUMNS,
    build_pair_diagnostics,
    build_pairs,
    build_pairwise_rows,
    collect_records,
    copy_run_evidence,
    write_csv,
    write_pairwise_readme,
)

COMPARE_SCRIPT_SOURCE = Path(__file__).resolve()
LIB_SOURCE = SCRIPT_DIR / "blind_pairwise_lib.py"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare ArgRE and ArgRE-NoAF final requirement texts pairwise."
    )
    parser.add_argument(
        "--argre-runs-dir",
        default="report/blind_requirement_quality_formal/runs/argre_af_preferred",
    )
    parser.add_argument(
        "--noaf-runs-dir",
        default="report/blind_requirement_quality_formal/runs/argre_noaf_original",
    )
    parser.add_argument(
        "--output",
        default="report/blind_requirement_quality_formal/pairwise_comparison.csv",
    )
    parser.add_argument(
        "--package-dir",
        default="report/blind_requirement_quality_formal",
        help="Write README.md and compare.py into this directory.",
    )
    parser.add_argument(
        "--source-argre-runs-dir",
        default="report/blind_requirement_quality_runs/argre_af_preferred/runs",
        help="Copy Phase 3 evidence from this tree into package runs/.",
    )
    parser.add_argument(
        "--source-noaf-runs-dir",
        default="report/blind_requirement_quality_runs/argre_noaf_original/runs",
        help="Copy Phase 3 evidence from this tree into package runs/.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-copy-runs", action="store_true")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    argre_dir = Path(args.argre_runs_dir)
    noaf_dir = Path(args.noaf_runs_dir)
    output_path = Path(args.output)

    if not args.skip_copy_runs:
        copy_run_evidence(
            argre_runs_dir=Path(args.source_argre_runs_dir),
            noaf_runs_dir=Path(args.source_noaf_runs_dir),
            output_runs_dir=package_dir / "runs",
        )
        argre_dir = package_dir / "runs" / "argre_af_preferred"
        noaf_dir = package_dir / "runs" / "argre_noaf_original"

    if not argre_dir.exists():
        raise FileNotFoundError(f"ArgRE runs dir not found: {argre_dir}")
    if not noaf_dir.exists():
        raise FileNotFoundError(f"NoAF runs dir not found: {noaf_dir}")

    argre_records = collect_records(argre_dir, method="ArgRE")
    noaf_records = collect_records(noaf_dir, method="ArgRE-NoAF")
    pairs, shortfalls = build_pairs(argre_records, noaf_records, seed=args.seed)
    pair_diagnostics = build_pair_diagnostics(pairs)
    pairwise_rows = build_pairwise_rows(pairs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, pairwise_rows, PAIRWISE_COLUMNS)
    write_pairwise_readme(
        package_dir / "README.md",
        pairs=pairs,
        pair_diagnostics=pair_diagnostics,
    )
    write_package_compare_script(package_dir / "compare.py")
    shutil.copy2(LIB_SOURCE, package_dir / "blind_pairwise_lib.py")

    print(f"Wrote pairwise comparison to {output_path}")
    print(f"Pairs: {len(pairs)}")
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


def write_package_compare_script(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
\"\"\"Reproduce pairwise ArgRE vs ArgRE-NoAF text comparison from local runs/.\"\"\"

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from blind_pairwise_lib import (  # noqa: E402
    DEFAULT_SEED,
    PAIRWISE_COLUMNS,
    build_pair_diagnostics,
    build_pairs,
    build_pairwise_rows,
    collect_records,
    write_csv,
    write_pairwise_readme,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ArgRE and ArgRE-NoAF requirement texts.")
    parser.add_argument(
        "--argre-runs-dir",
        default=str(PACKAGE_DIR / "runs" / "argre_af_preferred"),
    )
    parser.add_argument(
        "--noaf-runs-dir",
        default=str(PACKAGE_DIR / "runs" / "argre_noaf_original"),
    )
    parser.add_argument(
        "--output",
        default=str(PACKAGE_DIR / "pairwise_comparison.csv"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    argre_dir = Path(args.argre_runs_dir)
    noaf_dir = Path(args.noaf_runs_dir)
    if not argre_dir.exists():
        raise FileNotFoundError(f"ArgRE runs dir not found: {argre_dir}")
    if not noaf_dir.exists():
        raise FileNotFoundError(f"NoAF runs dir not found: {noaf_dir}")

    argre_records = collect_records(argre_dir, method="ArgRE")
    noaf_records = collect_records(noaf_dir, method="ArgRE-NoAF")
    pairs, shortfalls = build_pairs(argre_records, noaf_records, seed=args.seed)
    pair_diagnostics = build_pair_diagnostics(pairs)
    pairwise_rows = build_pairwise_rows(pairs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, pairwise_rows, PAIRWISE_COLUMNS)
    write_pairwise_readme(
        PACKAGE_DIR / "README.md",
        pairs=pairs,
        pair_diagnostics=pair_diagnostics,
    )

    print(f"Wrote pairwise comparison to {output_path}")
    print(f"Pairs: {len(pairs)}")
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


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


if __name__ == "__main__":
    raise SystemExit(main())
