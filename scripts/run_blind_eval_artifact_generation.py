#!/usr/bin/env python3
"""Generate ArgRE(AF) and ArgRE-NoAF artifacts for blind human evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from openre_bench.pipeline import PipelineConfig
from openre_bench.pipeline import run_case_pipeline


CASES = ("AD", "ATM", "Bookkeeping")
SEEDS = (101, 202, 303)
SETTING = "negotiation_integration_verification"
SYSTEM = "quare"


@dataclass(frozen=True)
class MethodConfig:
    label: str
    resolution_mode: str


METHODS = (
    MethodConfig(label="argre_af_preferred", resolution_mode="af_preferred"),
    MethodConfig(label="argre_noaf_original", resolution_mode="original"),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ArgRE(AF)/NoAF artifacts and build the blind eval package."
    )
    parser.add_argument("--cases-dir", default="data/case_studies")
    parser.add_argument("--output-root", default="report/blind_requirement_quality_runs")
    parser.add_argument("--blind-output-dir", default="report/blind_requirement_quality_formal")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--round-cap", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--rag-corpus-dir", default="data/knowledge_base")
    parser.add_argument("--attack-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--attack-llm-confidence-floor", type=float, default=0.85)
    parser.add_argument(
        "--paper-tools",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable paper-faithful BERT/Chroma/LLM entailment tools.",
    )
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        method_root = output_root / method.label
        for case_name in CASES:
            case_input = cases_dir / f"{case_name}_input.json"
            if not case_input.exists():
                raise FileNotFoundError(f"Missing case input: {case_input}")
            for seed in SEEDS:
                run_id = f"{method.label}-{case_name.lower()}-s{seed}"
                artifacts_dir = method_root / "runs" / run_id
                run_record_path = artifacts_dir / "run_record.json"
                if run_record_path.exists():
                    print(f"skip existing {run_id}")
                    continue
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                print(f"running {run_id}")
                run_case_pipeline(
                    PipelineConfig(
                        case_input=case_input,
                        artifacts_dir=artifacts_dir,
                        run_record_path=run_record_path,
                        run_id=run_id,
                        setting=SETTING,
                        seed=seed,
                        model=args.model,
                        temperature=args.temperature,
                        round_cap=args.round_cap,
                        max_tokens=args.max_tokens,
                        system=SYSTEM,
                        resolution_mode=method.resolution_mode,
                        rag_enabled=True,
                        rag_backend="local_tfidf",
                        rag_corpus_dir=Path(args.rag_corpus_dir),
                        attack_confidence_threshold=args.attack_confidence_threshold,
                        attack_llm_confidence_floor=args.attack_llm_confidence_floor,
                        paper_bert_conflict_prescreen=bool(args.paper_tools),
                        paper_chroma_hallucination_layer=bool(args.paper_tools),
                        paper_llm_compliance_entailment=bool(args.paper_tools),
                        paper_phase2_llm_pair_classification=bool(args.paper_tools),
                    )
                )

    build_blind_package(
        argre_runs_dir=output_root / "argre_af_preferred" / "runs",
        noaf_runs_dir=output_root / "argre_noaf_original" / "runs",
        output_dir=Path(args.blind_output_dir),
    )
    return 0


def build_blind_package(*, argre_runs_dir: Path, noaf_runs_dir: Path, output_dir: Path) -> None:
    command = [
        sys.executable,
        "scripts/build_blind_eval_from_artifacts.py",
        "--argre-runs-dir",
        str(argre_runs_dir),
        "--noaf-runs-dir",
        str(noaf_runs_dir),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
