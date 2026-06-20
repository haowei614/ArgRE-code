#!/usr/bin/env python3
"""Run the AD seed-101 SafetyAgent scope sensitivity pilot for ArgRE."""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

quare_prompts: Any | None = None


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_quare_prompts() -> Any:
    global quare_prompts
    if quare_prompts is None:
        from openre_bench.pipeline import quare as imported_quare_prompts

        quare_prompts = imported_quare_prompts
    return quare_prompts


SETTING = "negotiation_integration_verification"
SYSTEM = "quare"
SEED = 101
TEMPERATURE = 0.7
ROUND_CAP = 3
MAX_TOKENS = 4000
ATTACK_CONFIDENCE_THRESHOLD = 0.7
ATTACK_LLM_CONFIDENCE_FLOOR = 0.85
MODEL = "gpt-4o-mini-2024-07-18"

SAFETY_SCOPE_A = (
    "Focus only on safety concerns: hazard prevention, fault tolerance, risk "
    "mitigation, unsafe states, fail-safe behavior, and safety validation."
)

SAFETY_SCOPE_B = (
    "Concentrate exclusively on system safety aspects: risk mitigation strategies, "
    "fault-tolerant design, hazard identification and prevention, fail-safe mechanisms, "
    "unsafe state detection, and safety verification."
)

SAFETY_SCOPE_C = (
    "Your sole responsibility is evaluating safety-related requirements. This includes "
    "assessing hazard scenarios, ensuring fault tolerance and fail-safe behavior, "
    "identifying unsafe system states, validating risk mitigation measures, and "
    "confirming safety compliance."
)


@dataclass(frozen=True)
class Variant:
    variant: str
    label: str
    model: str
    prompt_variant: str


class ProgressLLMClient:
    """Thin logging wrapper around the repository LLM client."""

    def __init__(self, client: Any, *, label: str, hard_timeout_seconds: float) -> None:
        self._client = client
        self._label = label
        self._hard_timeout_seconds = max(5.0, float(hard_timeout_seconds))
        self._calls = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        self._calls += 1
        message_chars = sum(len(str(item.get("content", ""))) for item in messages)
        print(
            f"{self._label}: llm call {self._calls} "
            f"(temperature={temperature}, max_tokens={max_tokens}, seed={seed}, chars={message_chars})",
            flush=True,
        )
        import queue
        import threading

        from openre_bench.llm import LLMClientError

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def _invoke() -> None:
            try:
                result = self._client.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            except Exception as exc:  # pragma: no cover - operational logging wrapper.
                result_queue.put(("error", exc))
                return
            result_queue.put(("ok", result))

        worker = threading.Thread(target=_invoke, daemon=True)
        worker.start()
        try:
            status, value = result_queue.get(timeout=self._hard_timeout_seconds)
        except queue.Empty as exc:
            raise LLMClientError(
                f"LLM call exceeded hard timeout of {self._hard_timeout_seconds:.1f}s"
            ) from exc
        if status == "error":
            raise value
        return str(value)


VARIANTS: tuple[Variant, ...] = (
    Variant("A", "A (Original)", MODEL, "scope_a"),
    Variant("B", "B (Synonym+Reorder)", MODEL, "scope_b"),
    Variant("C", "C (Formal)", MODEL, "scope_c"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-input", default="data/case_studies/AD_input.json")
    parser.add_argument("--output-dir", default="report/prompt_sensitivity_pilot")
    parser.add_argument("--rag-corpus-dir", default="data/knowledge_base")
    parser.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    parser.add_argument(
        "--attack-detection-mode",
        choices=("full", "rule_only", "llm_only"),
        default="full",
        help="Attack builder mode; use rule_only as a constrained-runtime fallback.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed run metrics when present (default: enabled).",
    )
    parser.add_argument("--force", action="store_true", help="Delete existing output dir first.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replication_dir = output_dir / "replication_package"
    prompts_dir = replication_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    write_prompt_dumps(prompts_dir)
    print(f"prompt dumps written: {prompts_dir}", flush=True)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"running {variant.variant} with {variant.model}", flush=True)
        row = run_one_variant(
            variant=variant,
            case_input=Path(args.case_input),
            output_dir=output_dir,
            rag_corpus_dir=Path(args.rag_corpus_dir),
            llm_timeout_seconds=float(args.llm_timeout_seconds),
            resume=bool(args.resume),
            attack_detection_mode=str(args.attack_detection_mode),
        )
        rows.append(row)
        write_json_file(output_dir / "results_partial.json", rows)
        print(f"finished {variant.variant}: {row.get('status')}", flush=True)

    successful_rows = [row for row in rows if row.get("status") == "ok"]
    markdown = build_markdown_summary(rows)
    latex = build_latex_table(rows)
    (output_dir / "prompt_sensitivity_summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "prompt_sensitivity_table.tex").write_text(latex, encoding="utf-8")
    write_json_file(output_dir / "results.json", rows)
    write_json_file(output_dir / "summary_ranges.json", summarize_ranges(successful_rows))
    write_sanity_note(output_dir)

    print(markdown)
    print(f"Wrote results: {output_dir / 'results.json'}")
    print(f"Wrote Markdown table: {output_dir / 'prompt_sensitivity_summary.md'}")
    print(f"Wrote LaTeX table: {output_dir / 'prompt_sensitivity_table.tex'}")
    print(f"Wrote prompts: {prompts_dir}")
    return 0 if successful_rows else 1


def run_one_variant(
    *,
    variant: Variant,
    case_input: Path,
    output_dir: Path,
    rag_corpus_dir: Path,
    llm_timeout_seconds: float,
    resume: bool,
    attack_detection_mode: str,
) -> dict[str, Any]:
    from openre_bench.comparison_harness import _compute_run_metrics
    from openre_bench.pipeline import PipelineConfig
    from openre_bench.pipeline import run_case_pipeline

    ensure_quare_prompts()

    run_id = f"prompt_sensitivity-{variant.variant.lower()}-ad-s{SEED}-{safe_model_name(variant.model)}"
    artifacts_dir = output_dir / "runs" / run_id
    run_record_path = artifacts_dir / "run_record.json"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "metrics.json"

    if resume and metrics_path.exists():
        existing = load_json_file(metrics_path)
        if isinstance(existing, dict) and existing.get("status") == "ok":
            print(f"{variant.variant}: reuse completed metrics from {metrics_path}", flush=True)
            return existing

    row: dict[str, Any] = {
        "variant": variant.label,
        "variant_id": variant.variant,
        "prompt_variant": variant.prompt_variant,
        "safety_scope": safety_scope_for_variant(variant.prompt_variant),
        "model": variant.model,
        "run_id": run_id,
        "artifacts_dir": str(artifacts_dir),
        "seed": SEED,
        "temperature": TEMPERATURE,
        "theta": ATTACK_CONFIDENCE_THRESHOLD,
        "theta_floor": ATTACK_LLM_CONFIDENCE_FLOOR,
        "theta_eff": max(ATTACK_CONFIDENCE_THRESHOLD, ATTACK_LLM_CONFIDENCE_FLOOR),
        "attack_detection_mode": attack_detection_mode,
    }
    try:
        llm_client = make_progress_llm_client(
            model=variant.model,
            timeout_seconds=llm_timeout_seconds,
            label=variant.variant,
        )
        with patched_prompt_variant(variant.prompt_variant):
            run_case_pipeline(
                PipelineConfig(
                    case_input=case_input,
                    artifacts_dir=artifacts_dir,
                    run_record_path=run_record_path,
                    run_id=run_id,
                    setting=SETTING,
                    seed=SEED,
                    model=variant.model,
                    temperature=TEMPERATURE,
                    round_cap=ROUND_CAP,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM,
                    parallel=False,
                    llm_max_concurrency=1,
                    llm_client=llm_client,
                    resolution_mode="af_preferred",
                    rag_enabled=True,
                    rag_backend="local_tfidf",
                    rag_corpus_dir=rag_corpus_dir,
                    attack_confidence_threshold=ATTACK_CONFIDENCE_THRESHOLD,
                    attack_llm_confidence_floor=ATTACK_LLM_CONFIDENCE_FLOOR,
                    attack_detection_mode=attack_detection_mode,
                    paper_bert_conflict_prescreen=False,
                    paper_chroma_hallucination_layer=False,
                    paper_llm_compliance_entailment=False,
                    paper_phase2_llm_pair_classification=False,
                )
            )
        metrics = _compute_run_metrics(artifacts_dir)
        graph_metrics = extract_graph_metrics(artifacts_dir / "argumentation_graph.json")
        row.update(metrics)
        row.update(graph_metrics)
        row["status"] = "ok"
        write_json_file(artifacts_dir / "metrics.json", row)
    except Exception as exc:  # pragma: no cover - pilot runner records operational failures.
        row["status"] = "failed"
        row["error"] = str(exc)
        row["traceback"] = traceback.format_exc()
        write_json_file(artifacts_dir / "failure.json", row)
    return row


def make_progress_llm_client(*, model: str, timeout_seconds: float, label: str) -> Any | None:
    from openre_bench.llm import LLMClient
    from openre_bench.llm import MissingAPIKeyError
    from openre_bench.llm import load_openai_settings

    try:
        settings = load_openai_settings()
    except MissingAPIKeyError as exc:
        print(f"{label}: missing API key; pipeline will use deterministic fallback ({exc})", flush=True)
        return None
    settings.model = model
    settings.timeout_seconds = max(5.0, float(timeout_seconds))
    settings.request_retries = 0
    return ProgressLLMClient(
        LLMClient(settings),
        label=label,
        hard_timeout_seconds=settings.timeout_seconds,
    )


def safety_scope_for_variant(prompt_variant: str) -> str:
    return {
        "scope_a": SAFETY_SCOPE_A,
        "scope_b": SAFETY_SCOPE_B,
        "scope_c": SAFETY_SCOPE_C,
    }[prompt_variant]


@contextlib.contextmanager
def patched_prompt_variant(prompt_variant: str) -> Iterator[None]:
    prompt_module = ensure_quare_prompts()
    original_safety_scope = prompt_module.QUARE_AGENT_SYSTEM_SCOPES["SafetyAgent"]
    try:
        prompt_module.QUARE_AGENT_SYSTEM_SCOPES["SafetyAgent"] = safety_scope_for_variant(prompt_variant)
        yield
    finally:
        prompt_module.QUARE_AGENT_SYSTEM_SCOPES["SafetyAgent"] = original_safety_scope


def write_prompt_dumps(prompts_dir: Path) -> None:
    prompt_dump = {
        "variants": {
            "A": {
                "label": "A (Original)",
                "safety_scope": SAFETY_SCOPE_A,
            },
            "B": {
                "label": "B (Synonym+Reorder)",
                "safety_scope": SAFETY_SCOPE_B,
            },
            "C": {
                "label": "C (Formal)",
                "safety_scope": SAFETY_SCOPE_C,
            },
        },
        "patched_field": "QUARE_AGENT_SYSTEM_SCOPES['SafetyAgent']",
        "unchanged_fields": [
            "all non-SafetyAgent scopes",
            "_build_quare_llm_messages template",
            "user payload JSON schema",
            "temperature",
            "theta_eff",
            "round_cap",
        ],
        "system_prompt_template": (
            "You are {reviewer_agent} in QUARE Phase-2 dialectic negotiation. Review the focus "
            "model only through your assigned quality scope. {reviewer_agent_scope} When "
            "identifying conflicts, prioritize this scope and avoid optimizing other quality "
            "attributes unless they directly affect it. Return exactly one JSON object. Do not "
            "emit markdown fences or prose."
        ),
    }
    write_json_file(prompts_dir / "safety_scope_variants.json", prompt_dump)
    (prompts_dir / "README.md").write_text(
        "\n".join(
            [
                "# Safety Scope Sensitivity Pilot Prompts",
                "",
                "This directory records the SafetyAgent scope variants used for A/B/C.",
                "Only `QUARE_AGENT_SYSTEM_SCOPES['SafetyAgent']` is patched per variant.",
                "All other agent scopes, the prompt template, and JSON output schema are unchanged.",
                "",
                "- `safety_scope_variants.json`: complete scope text and shared prompt template.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def extract_graph_metrics(graph_path: Path) -> dict[str, Any]:
    from openre_bench.evaluation.af_metrics import compute_af_metrics

    graph = load_json_file(graph_path)
    arguments = graph.get("arguments", []) if isinstance(graph, dict) else []
    attacks = graph.get("attacks", []) if isinstance(graph, dict) else []
    grounded = graph.get("grounded_extension", []) if isinstance(graph, dict) else []
    meta = graph.get("meta", {}) if isinstance(graph, dict) else {}
    attack_meta = meta.get("attack_build", {}) if isinstance(meta, dict) else {}
    rule_counts = attack_meta.get("rule_pattern_counts", {}) if isinstance(attack_meta, dict) else {}

    p1 = _pattern_count(rule_counts, attacks, "critique_direct")
    p2 = _pattern_count(rule_counts, attacks, "refinement_prev_version")
    p3 = _pattern_count(rule_counts, attacks, "refinement_counter_critique")
    p_other = 0
    if isinstance(attacks, list):
        p_other = max(0, len(attacks) - p1 - p2 - p3)

    af_metrics = compute_af_metrics(graph_path.parent)
    return {
        "num_arguments": len(arguments) if isinstance(arguments, list) else 0,
        "num_attacks": len(attacks) if isinstance(attacks, list) else 0,
        "grounded_extension_size": len(grounded) if isinstance(grounded, list) else 0,
        "p1_count": p1,
        "p2_count": p2,
        "p3_count": p3,
        "p_other_count": p_other,
        "tc_percent": round(100.0 * float(af_metrics.get("tc", 0.0)), 1),
    }


def _pattern_count(rule_counts: Any, attacks: Any, pattern: str) -> int:
    if isinstance(rule_counts, dict) and pattern in rule_counts:
        try:
            return int(rule_counts.get(pattern, 0))
        except (TypeError, ValueError):
            return 0
    if not isinstance(attacks, list):
        return 0
    expected = f"rule:{pattern}"
    return sum(
        1
        for item in attacks
        if isinstance(item, dict) and str(item.get("source", "")).strip() == expected
    )


def build_markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | |A| | |R_att| | P1/P2/P3 | |E_g| | BERTScore | TC(%) |",
        "|---------|-----|---------|----------|-------|-----------|-------|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"| {row.get('variant', '')} | failed | failed | failed | failed | failed | failed |"
            )
            continue
        lines.append(
            "| "
            f"{row.get('variant', '')} | "
            f"{int(row.get('num_arguments', 0))} | "
            f"{int(row.get('num_attacks', 0))} | "
            f"{int(row.get('p1_count', 0))}/{int(row.get('p2_count', 0))}/"
            f"{int(row.get('p3_count', 0))} | "
            f"{int(row.get('grounded_extension_size', 0))} | "
            f"{float(row.get('semantic_preservation_f1', 0.0)):.3f} | "
            f"{_tc_percent(row):.1f} |"
        )
    delta_row = delta_max_row([row for row in rows if row.get("status") == "ok"])
    lines.append(
        "| Delta_max | "
        f"{delta_row['num_arguments']:.0f} | "
        f"{delta_row['num_attacks']:.0f} | "
        "- | "
        f"{delta_row['grounded_extension_size']:.0f} | "
        f"{delta_row['semantic_preservation_f1']:.3f} | "
        f"{delta_row['tc_percent']:.1f} |"
    )
    return "\n".join(lines) + "\n"


def delta_max_row(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "num_arguments",
        "num_attacks",
        "grounded_extension_size",
        "semantic_preservation_f1",
        "tc_percent",
    )
    return {key: numeric_delta(rows, key) for key in keys}


def numeric_delta(rows: list[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        value = _tc_percent(row) if key == "tc_percent" else row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return round(max(values) - min(values), 6)


def _tc_percent(row: dict[str, Any]) -> float:
    if row.get("tc_percent") is not None:
        return float(row.get("tc_percent", 0.0))
    return 100.0 * float(row.get("tc", 0.0))


def build_latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Prompt and Model Sensitivity (AD case, seed 101)}",
        r"\label{tab:prompt-sensitivity}",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Model} & $|\mathcal{A}|$ & $|\mathcal{R}_{att}|$ & $|\mathcal{E}_g|$ & \textbf{BERT} & \textbf{TC} & \textbf{Compl.} & \textbf{CRR} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(format_latex_row(row))
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def format_latex_row(row: dict[str, Any]) -> str:
    variant = latex_escape(str(row.get("variant", "")))
    model = latex_escape(display_model(str(row.get("model", ""))))
    if row.get("status") != "ok":
        reason = latex_escape("failed")
        return f"{variant} & {model} & \\multicolumn{{7}}{{c}}{{{reason}}} \\\\"
    return (
        f"{variant:<16} & {model:<7} & "
        f"{int(row.get('num_arguments', 0))} & "
        f"{int(row.get('num_attacks', 0))} & "
        f"{int(row.get('grounded_extension_size', 0))} & "
        f"{float(row.get('semantic_preservation_f1', 0.0)):.3f} & "
        f"{100.0 * float(row.get('tc', 0.0)):.1f} & "
        f"{100.0 * float(row.get('compliance_coverage', 0.0)):.1f} & "
        f"{100.0 * float(row.get('conflict_resolution_rate', 0.0)):.1f} \\\\"
    )


def summarize_ranges(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bert_percent_range": numeric_range(rows, "semantic_preservation_f1", multiplier=100.0),
        "num_arguments_range": numeric_range(rows, "num_arguments"),
        "num_attacks_range": numeric_range(rows, "num_attacks"),
        "p1_count_range": numeric_range(rows, "p1_count"),
        "p2_count_range": numeric_range(rows, "p2_count"),
        "p3_count_range": numeric_range(rows, "p3_count"),
        "grounded_extension_size_range": numeric_range(rows, "grounded_extension_size"),
        "tc_percent_range": numeric_range(rows, "tc", multiplier=100.0),
        "delta_max": delta_max_row(rows),
        "compliance_percent_range": numeric_range(rows, "compliance_coverage", multiplier=100.0),
        "crr_percent_range": numeric_range(rows, "conflict_resolution_rate", multiplier=100.0),
    }


def numeric_range(rows: list[dict[str, Any]], key: str, multiplier: float = 1.0) -> dict[str, float] | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            values.append(float(value) * multiplier)
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return {"min": round(min(values), 6), "max": round(max(values), 6)}


def write_sanity_note(output_dir: Path) -> None:
    existing = (
        REPO_ROOT
        / "report/blind_requirement_quality_formal/runs/argre_af_preferred/"
        / "argre_af_preferred-ad-s101/run_record.json"
    )
    note_lines = [
        "# Sanity Check Note",
        "",
        "Variant A is run with the original QUARE SafetyAgent scope, AD case, seed 101, "
        f"{MODEL}, temperature 0.7, and theta_eff 0.85.",
        "Variants B/C patch only `QUARE_AGENT_SYSTEM_SCOPES['SafetyAgent']`; the prompt "
        "template and JSON schema are unchanged.",
        "",
    ]
    if existing.exists():
        note_lines.append(f"Existing formal mirror run record found: `{existing}`.")
        try:
            payload = load_json_file(existing)
            note_lines.append(
                "Existing run metadata: "
                f"model={payload.get('model')}, setting={payload.get('setting')}, "
                f"system={payload.get('system')}, seed={payload.get('seed')}."
            )
        except Exception as exc:  # pragma: no cover
            note_lines.append(f"Could not parse existing run record: {exc}.")
        note_lines.append(
            "The complete original artifact directory referenced by that formal mirror is not "
            "present in this checkout, so metric-level equality could not be checked against it."
        )
    else:
        note_lines.append("No prior AD seed-101 formal mirror run record was found in this checkout.")
    (output_dir / "sanity_check.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def display_model(model: str) -> str:
    if model in {"gpt-4o-mini", "gpt-4o-mini-2024-07-18"}:
        return "4o-mini"
    if model == "gpt-4o":
        return "4o"
    return model


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


if __name__ == "__main__":
    raise SystemExit(main())
