#!/usr/bin/env python3
"""Reproduce pairwise ArgRE vs ArgRE-NoAF text comparison from local runs/."""

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
