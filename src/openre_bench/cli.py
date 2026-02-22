"""CLI entrypoint for OpenRE-Bench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openre_bench.auto_report import AutoReportConfig
from openre_bench.auto_report import run_auto_report
from openre_bench.comparison_harness import MatrixConfig
from openre_bench.comparison_harness import export_trace_audit
from openre_bench.comparison_harness import parse_seeds
from openre_bench.comparison_harness import parse_settings
from openre_bench.comparison_harness import prepare_blind_evaluation
from openre_bench.comparison_harness import run_comparison_matrix
from openre_bench.comparison_validator import validate_case_input
from openre_bench.comparison_validator import validate_phase_artifacts
from openre_bench.comparison_validator import validate_run_record
from openre_bench.comparison_validator import validate_system_behavior_contract
from openre_bench.schemas import load_json_file
from openre_bench.pipeline import PipelineConfig
from openre_bench.pipeline import default_run_id
from openre_bench.pipeline import run_case_pipeline
from openre_bench.llm_client import LLMClient
from openre_bench.schemas import SUPPORTED_SYSTEMS
from openre_bench.settings import MissingAPIKeyError
from openre_bench.settings import load_openai_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openre_bench",
        description="OpenRE-Bench CLI",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    parser.add_argument(
        "--check-openai",
        action="store_true",
        help="Validate OpenAI API key configuration and exit.",
    )
    parser.add_argument(
        "--llm-ping",
        action="store_true",
        help="Run a minimal LLM call to verify end-to-end inference connectivity.",
    )
    parser.add_argument(
        "--validate-comparison",
        action="store_true",
        help="Validate preflight/postrun comparison artifacts against protocol anchors.",
    )
    parser.add_argument(
        "--case-input",
        type=str,
        help="Path to case input JSON (expects case_name/case_description/requirement).",
    )
    parser.add_argument(
        "--run-record",
        type=str,
        help="Path to a run-record JSON for controlled-setting checks.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        help="Path to an artifacts directory containing phase1-4 JSON files.",
    )
    parser.add_argument(
        "--run-case",
        action="store_true",
        help="Run a deterministic OpenRE-Bench parity scaffold and emit phase1-4 artifacts.",
    )
    parser.add_argument(
        "--run-comparison-matrix",
        action="store_true",
        help="Run matrix harness and generate protocol deliverables.",
    )
    parser.add_argument(
        "--export-trace-audit",
        action="store_true",
        help="Export markdown trace-sanity audit from an existing matrix output directory.",
    )
    parser.add_argument(
        "--blind-eval-prepare",
        action="store_true",
        help="Prepare blinded matrix artifacts for strict blind evaluation.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run strict controlled MARE+QUARE matrix workflow and generate report/proofs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Explicit run id for scaffold execution.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="multi_agent_with_negotiation",
        help="Experiment setting label stored in run metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Run seed stored in run metadata.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model label stored in run metadata.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature stored in run metadata.",
    )
    parser.add_argument(
        "--round-cap",
        type=int,
        default=3,
        help="Negotiation round cap stored in run metadata.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Token budget stored in run metadata.",
    )
    parser.add_argument(
        "--rag-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable protocol RAG mode (default: enabled).",
    )
    parser.add_argument(
        "--rag-backend",
        type=str,
        default="local_tfidf",
        help="RAG backend label stored in run metadata.",
    )
    parser.add_argument(
        "--rag-corpus-dir",
        type=str,
        default="../OpenRE-Bench/data/knowledge_base",
        help="Path to knowledge corpus directory used for retrieval parity.",
    )
    parser.add_argument(
        "--cases-dir",
        type=str,
        default="../OpenRE-Bench/data/case_studies",
        help="Directory containing case-study JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_outputs",
        help="Directory where matrix outputs are written.",
    )
    parser.add_argument(
        "--matrix-output-dir",
        type=str,
        default="comparison_outputs",
        help="Existing matrix output directory used by audit/blind commands.",
    )
    parser.add_argument(
        "--blind-output-dir",
        type=str,
        default="comparison_blind_outputs",
        help="Directory where blinded evaluation artifacts are written.",
    )
    parser.add_argument(
        "--matrix-seeds",
        type=str,
        default="101",
        help="Comma-separated seed list for matrix runs.",
    )
    parser.add_argument(
        "--matrix-settings",
        type=str,
        default="",
        help="Comma-separated settings; defaults to protocol four settings.",
    )
    parser.add_argument(
        "--system",
        type=str,
        choices=list(SUPPORTED_SYSTEMS),
        default="mare",
        help="Execution system identity label for runtime routing (mare or quare).",
    )
    parser.add_argument(
        "--judge-script",
        type=str,
        default="src/openre_bench/comparison_harness.py",
        help="Path to judge pipeline script used for sha256 provenance hashing.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="report",
        help="Directory for /auto outputs (logs, runs, analysis, proofs).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    auto_requested = bool(args.auto) or str(args.command).strip().lower() in {"auto", "/auto"}

    if args.version:
        from openre_bench import __version__

        print(__version__)
        return 0

    if args.check_openai:
        try:
            settings = load_openai_settings()
        except MissingAPIKeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        base_url_text = settings.base_url or "(default)"
        print("OpenAI configuration is valid.")
        print(f"Model: {settings.model}")
        print(f"Base URL: {base_url_text}")
        return 0

    if args.llm_ping:
        try:
            settings = load_openai_settings()
        except MissingAPIKeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        client = LLMClient(settings)
        try:
            response = client.chat(
                [
                    {
                        "role": "user",
                        "content": "Reply with exactly: pong",
                    }
                ],
                temperature=0,
                max_tokens=16,
            )
        except Exception as exc:
            print(f"LLM ping failed: {exc}", file=sys.stderr)
            return 1

        print(response)
        return 0

    if auto_requested:
        try:
            auto_result = run_auto_report(
                AutoReportConfig(
                    report_dir=Path(args.report_dir),
                    cases_dir=Path(args.cases_dir),
                    seeds=parse_seeds(args.matrix_seeds),
                    settings=parse_settings(args.matrix_settings),
                    model=args.model,
                    temperature=args.temperature,
                    round_cap=args.round_cap,
                    max_tokens=args.max_tokens,
                    rag_enabled=bool(args.rag_enabled),
                    rag_backend=args.rag_backend,
                    rag_corpus_dir=Path(args.rag_corpus_dir),
                    judge_pipeline_path=Path(args.judge_script),
                )
            )
        except Exception as exc:
            print(f"/auto workflow failed: {exc}", file=sys.stderr)
            return 1

        if auto_result.warnings:
            print("WARNINGS:")
            for item in auto_result.warnings:
                print(f"- {item}")

        if auto_result.hard_failures:
            print("ERRORS:", file=sys.stderr)
            for item in auto_result.hard_failures:
                print(f"- {item}", file=sys.stderr)
            print(f"Run Key: {auto_result.run_key}")
            print(f"Run Dir: {auto_result.run_dir}")
            print(f"Logs Dir: {auto_result.logs_dir}")
            return 1

        print("/auto workflow completed.")
        print(f"Run Key: {auto_result.run_key}")
        print(f"Run Dir: {auto_result.run_dir}")
        print(f"Logs Dir: {auto_result.logs_dir}")
        print(f"Report README: {auto_result.report_readme}")
        print(f"Analysis: {auto_result.report_analysis}")
        print(f"Verdict JSON: {auto_result.verdict_path}")
        return 0

    if args.validate_comparison:
        errors: list[str] = []
        warnings: list[str] = []

        if not args.case_input and not args.run_record and not args.artifacts_dir:
            print(
                "Validation requires at least one of --case-input, --run-record, or --artifacts-dir.",
                file=sys.stderr,
            )
            return 2

        if args.case_input:
            report = validate_case_input(Path(args.case_input))
            errors.extend(report.errors)
            warnings.extend(report.warnings)

        if args.run_record:
            report = validate_run_record(Path(args.run_record))
            errors.extend(report.errors)
            warnings.extend(report.warnings)

        if args.artifacts_dir:
            report = validate_phase_artifacts(Path(args.artifacts_dir))
            errors.extend(report.errors)
            warnings.extend(report.warnings)

        if args.run_record and args.artifacts_dir:
            try:
                run_payload = load_json_file(Path(args.run_record))
            except (OSError, ValueError) as exc:
                warnings.append(
                    "Skipping system behavior validation because run record could "
                    f"not be parsed: {exc}"
                )
            else:
                if isinstance(run_payload, dict):
                    behavior_report = validate_system_behavior_contract(
                        system=str(run_payload.get("system", "")),
                        artifacts_dir=Path(args.artifacts_dir),
                    )
                    errors.extend(behavior_report.errors)
                    warnings.extend(behavior_report.warnings)

        if warnings:
            print("WARNINGS:")
            for item in warnings:
                print(f"- {item}")

        if errors:
            print("ERRORS:", file=sys.stderr)
            for item in errors:
                print(f"- {item}", file=sys.stderr)
            return 1

        print("Comparison validation passed.")
        return 0

    if args.run_case:
        if not args.case_input:
            print("--run-case requires --case-input.", file=sys.stderr)
            return 2

        case_path = Path(args.case_input)
        if not case_path.exists():
            print(f"Case input not found: {case_path}", file=sys.stderr)
            return 1

        run_id = args.run_id or default_run_id(case_path.stem, args.seed)
        artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path("artifacts") / run_id
        run_record_path = Path(args.run_record) if args.run_record else artifacts_dir / "run_record.json"

        pipeline_config = PipelineConfig(
            case_input=case_path,
            artifacts_dir=artifacts_dir,
            run_record_path=run_record_path,
            run_id=run_id,
            setting=args.setting,
            seed=args.seed,
            model=args.model,
            temperature=args.temperature,
            round_cap=args.round_cap,
            max_tokens=args.max_tokens,
            system=args.system,
            rag_enabled=bool(args.rag_enabled),
            rag_backend=args.rag_backend,
            rag_corpus_dir=Path(args.rag_corpus_dir),
        )
        run_record = run_case_pipeline(pipeline_config)

        case_report = validate_case_input(case_path)
        run_report = validate_run_record(run_record_path)
        artifacts_report = validate_phase_artifacts(artifacts_dir)
        behavior_report = validate_system_behavior_contract(
            system=run_record.system,
            artifacts_dir=artifacts_dir,
        )

        warnings = (
            case_report.warnings
            + run_report.warnings
            + artifacts_report.warnings
            + behavior_report.warnings
        )
        errors = (
            case_report.errors
            + run_report.errors
            + artifacts_report.errors
            + behavior_report.errors
        )

        if warnings:
            print("WARNINGS:")
            for item in warnings:
                print(f"- {item}")

        if errors:
            print("ERRORS:", file=sys.stderr)
            for item in errors:
                print(f"- {item}", file=sys.stderr)
            return 1

        print("OpenRE-Bench scaffold run completed.")
        print(f"Run ID: {run_record.run_id}")
        print(f"Case ID: {run_record.case_id}")
        print(f"Artifacts Dir: {artifacts_dir}")
        print(f"Run Record: {run_record_path}")
        print("Validation: passed")
        return 0

    if args.run_comparison_matrix:
        matrix_config = MatrixConfig(
            cases_dir=Path(args.cases_dir),
            output_dir=Path(args.output_dir),
            seeds=parse_seeds(args.matrix_seeds),
            settings=parse_settings(args.matrix_settings),
            model=args.model,
            temperature=args.temperature,
            round_cap=args.round_cap,
            max_tokens=args.max_tokens,
            system=args.system,
            rag_enabled=bool(args.rag_enabled),
            rag_backend=args.rag_backend,
            rag_corpus_dir=Path(args.rag_corpus_dir),
            judge_pipeline_path=Path(args.judge_script),
        )
        try:
            matrix_result = run_comparison_matrix(matrix_config)
        except Exception as exc:
            print(f"Matrix run failed: {exc}", file=sys.stderr)
            return 1

        if matrix_result.warnings:
            print("WARNINGS:")
            for item in matrix_result.warnings:
                print(f"- {item}")

        if matrix_result.errors:
            print("ERRORS:", file=sys.stderr)
            for item in matrix_result.errors:
                print(f"- {item}", file=sys.stderr)
            return 1

        print("Comparison matrix run completed.")
        print(f"Total runs: {matrix_result.total_runs} / Expected: {matrix_result.expected_runs}")
        print(f"Runs JSONL: {matrix_result.runs_jsonl}")
        print(f"By-case CSV: {matrix_result.by_case_csv}")
        print(f"Summary CSV: {matrix_result.summary_csv}")
        print(f"Ablation CSV: {matrix_result.ablation_csv}")
        print(f"Validity Log: {matrix_result.validity_md}")
        return 0

    if args.export_trace_audit:
        matrix_output_dir = Path(args.matrix_output_dir)
        try:
            trace_result = export_trace_audit(matrix_output_dir=matrix_output_dir)
        except Exception as exc:
            print(f"Trace audit export failed: {exc}", file=sys.stderr)
            return 1

        print("Trace audit export completed.")
        print(f"Output: {trace_result.output_path}")
        print(f"Total runs: {trace_result.total_runs}")
        print(f"Runs with loops: {trace_result.runs_with_loops}")
        print(f"Runs with conflicts: {trace_result.runs_with_conflicts}")
        return 0

    if args.blind_eval_prepare:
        matrix_output_dir = Path(args.matrix_output_dir)
        blind_output_dir = Path(args.blind_output_dir)
        judge_script = Path(args.judge_script)
        try:
            blind_result = prepare_blind_evaluation(
                matrix_output_dir=matrix_output_dir,
                blind_output_dir=blind_output_dir,
                judge_pipeline_path=judge_script,
            )
        except Exception as exc:
            print(f"Blind evaluation preparation failed: {exc}", file=sys.stderr)
            return 1

        print("Blind evaluation preparation completed.")
        print(f"Output Dir: {blind_result.output_dir}")
        print(f"Blinded Runs JSONL: {blind_result.blinded_runs_jsonl}")
        print(f"Blinded By-case CSV: {blind_result.blinded_by_case_csv}")
        print(f"Mapping JSON: {blind_result.mapping_json}")
        print(f"Protocol MD: {blind_result.protocol_md}")
        print(f"Judge Pipeline Hash: {blind_result.judge_pipeline_hash}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
