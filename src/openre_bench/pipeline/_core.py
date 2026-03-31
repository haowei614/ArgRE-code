"""Protocol-parity pipeline for phase artifact generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field


def _paper_tools_default() -> bool:
    from openre_bench.paper_env import paper_tools_enabled

    return paper_tools_enabled()


def _default_rag_backend_field() -> str:
    from openre_bench.paper_env import default_rag_backend

    return default_rag_backend()
from pathlib import Path
from typing import Any

from openre_bench import __version__
from openre_bench.llm import LLMClient
from openre_bench.llm import LLMClientError
from openre_bench.llm import LLMContract as Phase2LLMClient  # backward compat
from openre_bench.llm import MissingAPIKeyError
from openre_bench.llm import chat_with_optional_seed as _chat_with_optional_seed
from openre_bench.llm import load_openai_settings
from openre_bench.schemas import CaseInput
from openre_bench.schemas import DEFAULT_AGENT_QUALITY_ATTRIBUTES
from openre_bench.schemas import GSNConnection
from openre_bench.schemas import GSNElement
from openre_bench.schemas import KAOSElement
from openre_bench.schemas import MARE_ACTIONS
from openre_bench.schemas import MARE_AGENT_ROLES
from openre_bench.schemas import MARE_ROLE_ACTIONS
from openre_bench.schemas import non_comparable_reasons_for_setting
from openre_bench.schemas import NegotiationHistory
from openre_bench.schemas import NegotiationStep
from openre_bench.schemas import PHASE1_FILENAME
from openre_bench.schemas import PHASE0_FILENAME
from openre_bench.schemas import PHASE2_FILENAME
from openre_bench.schemas import PHASE25_FILENAME
from openre_bench.schemas import PHASE3_FILENAME
from openre_bench.schemas import PHASE4_FILENAME
from openre_bench.schemas import PHASE5_FILENAME
from openre_bench.schemas import Phase2Artifact
from openre_bench.schemas import Phase3Artifact
from openre_bench.schemas import Phase4Artifact
from openre_bench.schemas import RunComparability
from openre_bench.schemas import RunExecutionFlags
from openre_bench.schemas import RunProvenance
from openre_bench.schemas import RunRecord
from openre_bench.schemas import RunSystemIdentity
from openre_bench.schemas import SETTING_MULTI_AGENT_WITH_NEGOTIATION
from openre_bench.schemas import SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION
from openre_bench.schemas import SETTING_NEGOTIATION_INTEGRATION_VERIFICATION
from openre_bench.schemas import SETTING_SINGLE_AGENT
from openre_bench.schemas import SYSTEM_IREDEV
from openre_bench.schemas import SYSTEM_MARE
from openre_bench.schemas import SYSTEM_QUARE
from openre_bench.schemas import SUPPORTED_SYSTEMS
from openre_bench.schemas import load_json_file
from openre_bench.schemas import utc_timestamp
from openre_bench.schemas import write_json_file
from openre_bench.argumentation import build_attack_relations
from openre_bench.argumentation import parse_phase2_arguments
from openre_bench.argumentation import solve_argumentation_framework
from openre_bench.pipeline._utils import _extract_requirement_fragments
# settings and llm_client are consolidated into openre_bench.llm (imported above)


@dataclass
class PipelineConfig:
    """Run configuration used to generate parity artifacts."""

    case_input: Path
    artifacts_dir: Path
    run_record_path: Path
    run_id: str
    setting: str
    seed: int
    model: str
    temperature: float
    round_cap: int
    max_tokens: int
    system: str = SYSTEM_MARE
    parallel: bool = True
    llm_max_concurrency: int = 10
    llm_backoff_cap_seconds: float = 60.0
    llm_backoff_base_seconds: float = 1.0
    resolution_mode: str = "original"
    parser_temperature: float = 0.7
    attack_detection_temperature: float = 1.0
    attack_confidence_threshold: float = 0.7
    """Nominal LLM confidence threshold for cross-pair attack classification (paper: $\\theta$)."""

    attack_llm_confidence_floor: float = 0.85
    """Lower bound applied with attack_confidence_threshold to form $\\theta_{\\mathrm{eff}}$ for LLM edges (default matches main experiments). Set to 0.0 for sensitivity runs so $\\theta_{\\mathrm{eff}}=\\theta$."""

    attack_detection_mode: str = "full"
    judge_temperature: float = 0.0
    priority_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Safety": 0.2,
            "Efficiency": 0.2,
            "Green": 0.2,
            "Trustworthiness": 0.2,
            "Responsibility": 0.2,
        }
    )
    rag_enabled: bool = True
    rag_backend: str = field(default_factory=_default_rag_backend_field)
    rag_corpus_dir: Path | None = None
    llm_client: Phase2LLMClient | None = None
    paper_bert_conflict_prescreen: bool = field(default_factory=_paper_tools_default)
    paper_bert_similarity_tau: float = 0.85
    paper_chroma_hallucination_layer: bool = field(default_factory=_paper_tools_default)
    paper_hallucination_similarity_floor: float = 0.60
    paper_llm_compliance_entailment: bool = field(default_factory=_paper_tools_default)
    paper_phase2_llm_pair_classification: bool = field(default_factory=_paper_tools_default)
    paper_pair_classification_temperature: float = 0.7
    paper_chroma_persist_dir: Path | None = None


@dataclass
class Phase2ExecutionMeta:
    """Execution metadata from phase 2 used for strict run-record tainting."""

    llm_enabled: bool = False
    llm_turns: int = 0
    llm_fallback_turns: int = 0
    llm_retry_count: int = 0
    llm_parse_recoveries: int = 0
    llm_seed_applied_turns: int = 0
    llm_source: str = "disabled"


@dataclass
class MareRuntimeExecutionMeta:
    """Execution metadata for the MARE runtime 5-agent/9-action workflow."""

    llm_enabled: bool = False
    llm_turns: int = 0
    llm_fallback_turns: int = 0
    llm_retry_count: int = 0
    llm_parse_recoveries: int = 0
    llm_seed_applied_turns: int = 0
    llm_source: str = "disabled"
    execution_mode: str = "deterministic_emulation"


_RAG_CHUNK_CACHE: dict[str, list[dict[str, Any]]] = {}
_CORPUS_HASH_CACHE: dict[str, str] = {}
PHASE2_LLM_RETRY_LIMIT = 1
MARE_RUNTIME_SEMANTICS_MODE = "mare_paper_workflow_v1"
MARE_RUNTIME_TRACE_VERSION = "1"
IREDEV_RUNTIME_SEMANTICS_MODE = "iredev_knowledge_driven_v1"
IREDEV_RUNTIME_TRACE_VERSION = "1"
ARGUMENTATION_GRAPH_FILENAME = "argumentation_graph.json"

# NOTE: MARE paper (Jin et al., ASE 2024) does not define quality attributes
# per role. These are OpenRE-Bench design decisions to enable comparable KAOS
# element generation across all three systems, using QUARE's five quality
# dimensions distributed across MARE's 5 roles.
MARE_ROLE_QUALITY_ATTRIBUTES: dict[str, str] = {
    "Stakeholders": "Responsibility",
    "Collector": "Efficiency",
    "Modeler": "Trustworthiness",
    "Checker": "Safety",
    "Documenter": "Sustainability",
}

# NOTE: iReDev paper (Jin et al., TOSEM 2025) does not define quality
# attributes per role. Same rationale as MARE_ROLE_QUALITY_ATTRIBUTES above.
IREDEV_ROLE_QUALITY_ATTRIBUTES: dict[str, str] = {
    "Interviewer": "Responsibility",
    "EndUser": "Efficiency",
    "Deployer": "Safety",
    "Analyst": "Trustworthiness",
    "Archivist": "Sustainability",
    "Reviewer": "Integrated",
}

QUALITY_LENS_CUES: dict[str, tuple[str, ...]] = {
    "Safety": (
        "hazard prevention",
        "fault tolerance",
        "risk mitigation",
    ),
    "Efficiency": (
        "latency optimization",
        "throughput stability",
        "resource utilization",
    ),
    "Sustainability": (
        "energy footprint reduction",
        "resource lifecycle control",
        "environmental impact awareness",
    ),
    "Trustworthiness": (
        "security assurance",
        "auditability",
        "integrity guarantees",
    ),
    "Responsibility": (
        "regulatory accountability",
        "stakeholder transparency",
        "ethical compliance",
    ),
    "Integrated": (
        "cross-quality balance",
        "holistic requirement coherence",
        "end-to-end requirement traceability",
    ),
}


# Phase2LLMClient and _chat_with_optional_seed are imported from openre_bench.llm



def run_case_pipeline(config: PipelineConfig) -> RunRecord:
    """Generate deterministic phase artifacts and run record for one case."""

    if config.system == SYSTEM_QUARE:
        return _run_quare_pipeline(config)
    if config.system == SYSTEM_MARE:
        return _run_mare_pipeline(config)
    if config.system == SYSTEM_IREDEV:
        return _run_iredev_pipeline(config)
    raise ValueError(f"Unsupported system '{config.system}'. Supported systems: {SUPPORTED_SYSTEMS}")


def _run_mare_pipeline(config: PipelineConfig) -> RunRecord:
    """Run the baseline MARE scaffold pipeline."""

    return _run_pipeline_for_system(config=config, system=SYSTEM_MARE)


def _run_quare_pipeline(config: PipelineConfig) -> RunRecord:
    """Run the QUARE scaffold pipeline.

    MVP behavior is intentionally equivalent to MARE generation logic while preserving
    explicit system identity and reproducibility metadata for strict comparability checks.
    """

    return _run_pipeline_for_system(config=config, system=SYSTEM_QUARE)


def _run_iredev_pipeline(config: PipelineConfig) -> RunRecord:
    """Run the iReDev knowledge-driven multi-agent pipeline.

    Implements the 6-agent knowledge-driven workflow from Jin et al. (TOSEM 2025)
    with Interviewer, EndUser, Deployer, Analyst, Archivist, and Reviewer agents.
    """

    return _run_pipeline_for_system(config=config, system=SYSTEM_IREDEV)


def _run_pipeline_for_system(*, config: PipelineConfig, system: str) -> RunRecord:
    """Shared deterministic pipeline implementation for a selected system identity."""

    case_payload = load_json_file(config.case_input)
    case = CaseInput.model_validate(case_payload)

    started_at = time.perf_counter()
    start_timestamp = utc_timestamp()

    chroma_persist_early = config.paper_chroma_persist_dir
    if chroma_persist_early is None:
        chroma_persist_early = config.artifacts_dir.parent / ".chroma_openre_bench"

    rag_context = _prepare_rag_context(
        rag_enabled=config.rag_enabled,
        rag_backend=config.rag_backend,
        rag_corpus_dir=config.rag_corpus_dir,
        chroma_persist_dir=chroma_persist_early,
    )

    llm_client, llm_source = _resolve_phase2_llm_client(config=config, system=system)
    mare_llm_client, mare_llm_source = _resolve_runtime_llm_client(
        config=config,
        system=system,
    )

    mare_runtime_semantics: dict[str, Any] | None = None
    mare_runtime_meta = MareRuntimeExecutionMeta(llm_source=mare_llm_source)
    if system == SYSTEM_MARE and config.setting != SETTING_SINGLE_AGENT:
        phase1_payload, mare_runtime_semantics, mare_runtime_meta = _build_phase1_mare_semantics(
            case=case,
            seed=config.seed,
            setting=config.setting,
            rag_context=rag_context,
            llm_client=mare_llm_client,
            llm_source=mare_llm_source,
            llm_temperature=config.temperature,
            llm_max_tokens=config.max_tokens,
            llm_seed=config.seed,
        )
    elif system == SYSTEM_IREDEV and config.setting != SETTING_SINGLE_AGENT:
        phase1_payload, mare_runtime_semantics, mare_runtime_meta = _build_phase1_iredev_semantics(
            case=case,
            seed=config.seed,
            setting=config.setting,
            rag_context=rag_context,
            llm_client=mare_llm_client,
            llm_source=mare_llm_source,
            llm_temperature=config.temperature,
            llm_max_tokens=config.max_tokens,
            llm_seed=config.seed,
        )
    else:
        phase1_payload = _build_phase1(case, config.seed, config.setting, rag_context)
    phase2_payload, phase2_meta, argumentation_graph_payload = _build_phase2(
        run_id=config.run_id,
        phase1=phase1_payload,
        setting=config.setting,
        requirement=case.requirement,
        system=system,
        round_cap=config.round_cap,
        llm_client=llm_client,
        llm_source=llm_source,
        llm_model=config.model,
        llm_temperature=config.temperature,
        llm_max_tokens=config.max_tokens,
        llm_seed=config.seed,
        parallel=config.parallel,
        llm_max_concurrency=config.llm_max_concurrency,
        llm_backoff_cap_seconds=config.llm_backoff_cap_seconds,
        llm_backoff_base_seconds=config.llm_backoff_base_seconds,
        resolution_mode=config.resolution_mode,
        parser_temperature=config.parser_temperature,
        attack_detection_temperature=config.attack_detection_temperature,
        attack_confidence_threshold=config.attack_confidence_threshold,
        attack_llm_confidence_floor=config.attack_llm_confidence_floor,
        attack_detection_mode=config.attack_detection_mode,
        priority_weights=config.priority_weights,
        paper_bert_conflict_prescreen=config.paper_bert_conflict_prescreen,
        paper_bert_similarity_tau=config.paper_bert_similarity_tau,
        paper_phase2_llm_pair_classification=config.paper_phase2_llm_pair_classification,
        paper_pair_classification_temperature=config.paper_pair_classification_temperature,
    )
    phase3_payload = _build_phase3(
        run_id=config.run_id,
        case=case,
        phase1=phase1_payload,
        phase2=phase2_payload,
        setting=config.setting,
    )
    entailment_client = llm_client if llm_client is not None else mare_llm_client

    phase4_payload = _build_phase4(
        phase3_payload=phase3_payload,
        requirement=case.requirement,
        setting=config.setting,
        rag_corpus_dir=config.rag_corpus_dir,
        chroma_persist_dir=chroma_persist_early,
        paper_chroma_hallucination_layer=config.paper_chroma_hallucination_layer,
        paper_hallucination_tau_h=config.paper_hallucination_similarity_floor,
        paper_llm_compliance_entailment=config.paper_llm_compliance_entailment,
        llm_client=entailment_client,
        llm_model=config.model,
    )

    quare_optional_artifacts: dict[str, dict[str, Any]] = {}
    if system == SYSTEM_QUARE:
        quare_optional_artifacts = _build_quare_optional_artifacts(
            case=case,
            phase2=phase2_payload,
            phase3=phase3_payload,
            phase4=phase4_payload,
            setting=config.setting,
            round_cap=config.round_cap,
            rag_context=rag_context,
        )

    artifact_paths = {
        PHASE1_FILENAME: str(config.artifacts_dir / PHASE1_FILENAME),
        PHASE2_FILENAME: str(config.artifacts_dir / PHASE2_FILENAME),
        PHASE3_FILENAME: str(config.artifacts_dir / PHASE3_FILENAME),
        PHASE4_FILENAME: str(config.artifacts_dir / PHASE4_FILENAME),
    }
    if quare_optional_artifacts:
        artifact_paths.update(
            {
                PHASE0_FILENAME: str(config.artifacts_dir / PHASE0_FILENAME),
                PHASE25_FILENAME: str(config.artifacts_dir / PHASE25_FILENAME),
                PHASE5_FILENAME: str(config.artifacts_dir / PHASE5_FILENAME),
            }
        )
    if argumentation_graph_payload is not None:
        artifact_paths[ARGUMENTATION_GRAPH_FILENAME] = str(
            config.artifacts_dir / ARGUMENTATION_GRAPH_FILENAME
        )

    write_json_file(Path(artifact_paths[PHASE1_FILENAME]), phase1_payload)
    write_json_file(Path(artifact_paths[PHASE2_FILENAME]), phase2_payload)
    write_json_file(Path(artifact_paths[PHASE3_FILENAME]), phase3_payload)
    write_json_file(Path(artifact_paths[PHASE4_FILENAME]), phase4_payload)
    if argumentation_graph_payload is not None:
        write_json_file(Path(artifact_paths[ARGUMENTATION_GRAPH_FILENAME]), argumentation_graph_payload)
    if quare_optional_artifacts:
        write_json_file(Path(artifact_paths[PHASE0_FILENAME]), quare_optional_artifacts[PHASE0_FILENAME])
        write_json_file(Path(artifact_paths[PHASE25_FILENAME]), quare_optional_artifacts[PHASE25_FILENAME])
        write_json_file(Path(artifact_paths[PHASE5_FILENAME]), quare_optional_artifacts[PHASE5_FILENAME])

    end_timestamp = utc_timestamp()
    runtime_seconds = round(time.perf_counter() - started_at, 6)
    non_comparable_reasons = non_comparable_reasons_for_setting(config.setting)
    prompt_hash = _prompt_contract_hash(
        system=system,
        setting=config.setting,
        round_cap=config.round_cap,
        max_tokens=config.max_tokens,
    )
    llm_retry_count = int(phase2_meta.llm_retry_count) + int(mare_runtime_meta.llm_retry_count)
    llm_fallback_used = bool(
        phase2_meta.llm_fallback_turns > 0 or mare_runtime_meta.llm_fallback_turns > 0
    )
    retry_used = llm_retry_count > 0
    fallback_tainted = bool(rag_context["fallback_used"] or llm_fallback_used)
    runtime_semantics_mode = _runtime_semantics_mode(system=system, setting=config.setting)
    runtime_semantics_notes: dict[str, Any] = {
        "mode": runtime_semantics_mode,
    }
    if isinstance(mare_runtime_semantics, dict):
        runtime_semantics_notes.update(mare_runtime_semantics)

    run_record = RunRecord(
        run_id=config.run_id,
        case_id=case.case_name,
        system=system,
        setting=config.setting,
        seed=config.seed,
        model=config.model,
        temperature=config.temperature,
        round_cap=config.round_cap,
        system_identity=RunSystemIdentity(
            system_name=system,
            implementation="deterministic-parity-pipeline",
            implementation_version=__version__,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            machine=platform.machine(),
        ),
        provenance=RunProvenance(
            model=config.model,
            temperature=config.temperature,
            seed=config.seed,
            prompt_hash=prompt_hash,
            corpus_hash=str(rag_context["corpus_hash"]),
            corpus_path=str(rag_context["rag_corpus_dir"]),
        ),
        execution_flags=RunExecutionFlags(
            rag_fallback_used=bool(rag_context["fallback_used"]),
            llm_fallback_used=llm_fallback_used,
            fallback_tainted=fallback_tainted,
            retry_used=retry_used,
            retry_count=llm_retry_count,
        ),
        comparability=RunComparability(
            is_comparable=not non_comparable_reasons,
            non_comparable_reasons=non_comparable_reasons,
        ),
        max_tokens=config.max_tokens,
        rag_enabled=rag_context["rag_enabled"],
        rag_backend=rag_context["rag_backend"],
        rag_fallback_used=bool(rag_context["fallback_used"]),
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        runtime_seconds=runtime_seconds,
        artifacts_dir=str(config.artifacts_dir),
        artifact_paths=artifact_paths,
        artifact_blinded=False,
        blinding_scheme_version="",
        blind_eval_run_id="",
        judge_pipeline_hash="",
        trace_audit_path="",
        notes={
            "phase_a_b": True,
            "generation_mode": "deterministic-parity-pipeline",
            "runtime_semantics": runtime_semantics_notes,
            "resolution_mode": config.resolution_mode,
            "negotiation_temperature": config.temperature,
            "parser_temperature": config.parser_temperature,
            "attack_detection_temperature": config.attack_detection_temperature,
            "attack_confidence_threshold": config.attack_confidence_threshold,
            "attack_llm_confidence_floor": config.attack_llm_confidence_floor,
            "attack_detection_mode": config.attack_detection_mode,
            "judge_temperature": config.judge_temperature,
            "case_description": case.case_description,
            "system": system,
            "setting": config.setting,
            "negotiation_enabled": _negotiation_enabled(config.setting),
            "verification_executed": _verification_executed(config.setting),
            "quare_only_behaviors": {
                "phase0_external_spec_processing": system == SYSTEM_QUARE,
                "dynamic_round_control": system == SYSTEM_QUARE,
                "phase25_conflict_map_generation": system == SYSTEM_QUARE,
                "phase5_software_materials_generation": system == SYSTEM_QUARE,
                "phase2_dialectic_negotiation": system == SYSTEM_QUARE
                and _negotiation_enabled(config.setting),
            },
            "quare_optional_artifacts": {
                name: artifact_paths[name]
                for name in (PHASE0_FILENAME, PHASE25_FILENAME, PHASE5_FILENAME)
                if name in artifact_paths
            },
            "argumentation_artifacts": {
                ARGUMENTATION_GRAPH_FILENAME: artifact_paths[ARGUMENTATION_GRAPH_FILENAME]
            }
            if ARGUMENTATION_GRAPH_FILENAME in artifact_paths
            else {},
            "phase2_llm": {
                "source": phase2_meta.llm_source,
                "enabled": phase2_meta.llm_enabled,
                "seed": config.seed,
                "seed_applied_turns": phase2_meta.llm_seed_applied_turns,
                "turns": phase2_meta.llm_turns,
                "fallback_turns": phase2_meta.llm_fallback_turns,
                "retry_count": phase2_meta.llm_retry_count,
                "parse_recoveries": phase2_meta.llm_parse_recoveries,
            },
            "mare_runtime_llm": {
                "source": mare_runtime_meta.llm_source,
                "enabled": mare_runtime_meta.llm_enabled,
                "seed": config.seed,
                "seed_applied_turns": mare_runtime_meta.llm_seed_applied_turns,
                "turns": mare_runtime_meta.llm_turns,
                "fallback_turns": mare_runtime_meta.llm_fallback_turns,
                "retry_count": mare_runtime_meta.llm_retry_count,
                "parse_recoveries": mare_runtime_meta.llm_parse_recoveries,
                "execution_mode": mare_runtime_meta.execution_mode,
            },
            "rag_corpus_dir": rag_context["rag_corpus_dir"],
            "rag_corpus_hash": rag_context["corpus_hash"],
            "rag_chunk_count": rag_context["chunk_count"],
            "prompt_contract_hash": prompt_hash,
            "paper_verification_tools": {
                "bert_conflict_prescreen": config.paper_bert_conflict_prescreen,
                "bert_similarity_tau": config.paper_bert_similarity_tau,
                "chroma_hallucination_layer": config.paper_chroma_hallucination_layer,
                "hallucination_tau_h": config.paper_hallucination_similarity_floor,
                "llm_compliance_entailment": config.paper_llm_compliance_entailment,
                "phase2_llm_pair_classification": config.paper_phase2_llm_pair_classification,
                "pair_classification_temperature": config.paper_pair_classification_temperature,
                "chroma_embedding_model": "text-embedding-ada-002",
                "chroma_persist_dir": str(chroma_persist_early.resolve()),
            },
        },
    )

    write_json_file(config.run_record_path, run_record.model_dump(mode="json"))
    return run_record


def _resolve_phase2_llm_client(
    *,
    config: PipelineConfig,
    system: str,
) -> tuple[Phase2LLMClient | None, str]:
    """Resolve the client for QUARE phase2; missing client marks non-comparable fallback."""

    if system != SYSTEM_QUARE or not _negotiation_enabled(config.setting):
        return None, "disabled"

    if config.llm_client is not None:
        return config.llm_client, "injected"

    try:
        settings = load_openai_settings()
    except MissingAPIKeyError:
        return None, "missing_api_key"

    settings.model = config.model
    return LLMClient(settings), "openai"


def _resolve_mare_runtime_llm_client(
    *,
    config: PipelineConfig,
    system: str,
) -> tuple[Phase2LLMClient | None, str]:
    """Resolve the client for MARE paper workflow actions."""

    return _resolve_runtime_llm_client(config=config, system=system)


def _resolve_runtime_llm_client(
    *,
    config: PipelineConfig,
    system: str,
) -> tuple[Phase2LLMClient | None, str]:
    """Resolve the LLM client for MARE or iReDev paper workflow actions."""

    if system not in (SYSTEM_MARE, SYSTEM_IREDEV) or config.setting == SETTING_SINGLE_AGENT:
        return None, "disabled"

    if config.llm_client is not None:
        return config.llm_client, "injected"

    try:
        settings = load_openai_settings()
    except MissingAPIKeyError:
        return None, "missing_api_key"

    settings.model = config.model
    return LLMClient(settings), "openai"


def _prepare_rag_context(
    *,
    rag_enabled: bool,
    rag_backend: str,
    rag_corpus_dir: Path | None,
    chroma_persist_dir: Path | None = None,
) -> dict[str, Any]:
    """Prepare deterministic RAG context used during generation."""

    if not rag_enabled:
        return {
            "rag_enabled": False,
            "rag_backend": "none",
            "rag_corpus_dir": "",
            "corpus_hash": "",
            "chunks": [],
            "chunk_count": 0,
            "fallback_used": False,
            "chroma_rag": {"skipped": True, "skip_reason": "rag_disabled"},
        }

    corpus_dir = (rag_corpus_dir or Path("../OpenRE-Bench/data/knowledge_base")).resolve()
    corpus_hash = _hash_corpus_dir(corpus_dir)
    backend_raw = (rag_backend.strip() or "local_tfidf").lower()
    use_chroma = backend_raw in ("chroma_ada002", "chroma", "openai_ada_chroma")

    if use_chroma:
        persist = (chroma_persist_dir or Path(".chroma_openre_bench")).resolve()
        try:
            settings = load_openai_settings()
            api_key = settings.resolved_api_key
        except MissingAPIKeyError:
            chunks = _load_rag_chunks(corpus_dir)
            return {
                "rag_enabled": True,
                "rag_backend": "local_tfidf",
                "rag_corpus_dir": str(corpus_dir),
                "corpus_hash": corpus_hash,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "fallback_used": len(chunks) == 0 or True,
                "chroma_rag": {"skipped": True, "skip_reason": "missing_openai_api_key"},
            }

        from openre_bench.verification.chroma_corpus_index import ensure_chroma_embedded_corpus

        chroma_meta = ensure_chroma_embedded_corpus(
            corpus_dir=corpus_dir,
            persist_root=persist,
            openai_api_key=api_key,
            namespace="phase1_rag",
        )
        if chroma_meta.get("skipped"):
            chunks = _load_rag_chunks(corpus_dir)
            return {
                "rag_enabled": True,
                "rag_backend": "local_tfidf",
                "rag_corpus_dir": str(corpus_dir),
                "corpus_hash": corpus_hash,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "fallback_used": True,
                "chroma_rag": chroma_meta,
            }

        return {
            "rag_enabled": True,
            "rag_backend": "chroma_ada002",
            "rag_corpus_dir": str(corpus_dir),
            "corpus_hash": corpus_hash,
            "chunks": [],
            "chunk_count": int(chroma_meta.get("chunk_count", 0)),
            "fallback_used": False,
            "chroma_rag": chroma_meta,
        }

    chunks = _load_rag_chunks(corpus_dir)
    fallback_used = len(chunks) == 0
    backend = rag_backend.strip() or "local_tfidf"
    return {
        "rag_enabled": True,
        "rag_backend": backend,
        "rag_corpus_dir": str(corpus_dir),
        "corpus_hash": corpus_hash,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "fallback_used": fallback_used,
        "chroma_rag": {"skipped": True, "skip_reason": "lexical_rag_backend"},
    }


def _load_rag_chunks(corpus_dir: Path) -> list[dict[str, Any]]:
    """Load and cache RAG chunks from the configured corpus directory."""

    cache_key = str(corpus_dir)
    if cache_key in _RAG_CHUNK_CACHE:
        return _RAG_CHUNK_CACHE[cache_key]

    if not corpus_dir.exists() or not corpus_dir.is_dir():
        _RAG_CHUNK_CACHE[cache_key] = []
        return []

    chunks: list[dict[str, Any]] = []
    allowed_suffixes = {".md", ".txt", ".json", ".py"}
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel_path = path.relative_to(corpus_dir)
        file_chunks = _chunk_text(text)
        for index, chunk_text in enumerate(file_chunks, start=1):
            tokens = set(_tokens(chunk_text.lower()))
            if not tokens:
                continue
            chunks.append(
                {
                    "chunk_id": f"{rel_path}:{index}",
                    "document": str(rel_path),
                    "text": chunk_text,
                    "tokens": tokens,
                }
            )
            if len(chunks) >= 5000:
                break
        if len(chunks) >= 5000:
            break

    _RAG_CHUNK_CACHE[cache_key] = chunks
    return chunks


def _hash_corpus_dir(corpus_dir: Path) -> str:
    """Hash corpus files to make retrieval provenance machine-checkable."""

    cache_key = str(corpus_dir)
    if cache_key in _CORPUS_HASH_CACHE:
        return _CORPUS_HASH_CACHE[cache_key]

    if not corpus_dir.exists() or not corpus_dir.is_dir():
        _CORPUS_HASH_CACHE[cache_key] = ""
        return ""

    digest = hashlib.sha256()
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(corpus_dir).as_posix()
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            _CORPUS_HASH_CACHE[cache_key] = ""
            return ""

    corpus_hash = digest.hexdigest()
    _CORPUS_HASH_CACHE[cache_key] = corpus_hash
    return corpus_hash


def _prompt_contract_hash(*, system: str, setting: str, round_cap: int, max_tokens: int) -> str:
    """Hash deterministic generation contract to detect configuration drift."""

    runtime_mode = _runtime_semantics_mode(system=system, setting=setting)
    contract = {
        "generator": "openre_bench-deterministic-parity-pipeline",
        "system": system,
        "setting": setting,
        "round_cap": round_cap,
        "max_tokens": max_tokens,
        "runtime_semantics_mode": runtime_mode,
        "agent_quality_mapping": _agent_quality_mapping_for_setting(setting),
    }
    if runtime_mode == MARE_RUNTIME_SEMANTICS_MODE:
        contract["agent_quality_mapping"] = {
            role: MARE_ROLE_QUALITY_ATTRIBUTES.get(role, "Integrated")
            for role in MARE_AGENT_ROLES
        }
        contract["mare_roles"] = list(MARE_AGENT_ROLES)
        contract["mare_actions"] = list(MARE_ACTIONS)
    elif runtime_mode == IREDEV_RUNTIME_SEMANTICS_MODE:
        from openre_bench.schemas import IREDEV_ACTIONS
        from openre_bench.schemas import IREDEV_AGENT_ROLES

        contract["agent_quality_mapping"] = {
            role: IREDEV_ROLE_QUALITY_ATTRIBUTES.get(role, "Integrated")
            for role in IREDEV_AGENT_ROLES
        }
        contract["iredev_roles"] = list(IREDEV_AGENT_ROLES)
        contract["iredev_actions"] = list(IREDEV_ACTIONS)
    serialized = json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _runtime_semantics_mode(*, system: str, setting: str) -> str:
    """Return machine-checkable runtime semantics mode label."""

    if system == SYSTEM_MARE and setting != SETTING_SINGLE_AGENT:
        return MARE_RUNTIME_SEMANTICS_MODE
    if system == SYSTEM_MARE:
        return "mare_single_agent_baseline"
    if system == SYSTEM_IREDEV and setting != SETTING_SINGLE_AGENT:
        return IREDEV_RUNTIME_SEMANTICS_MODE
    if system == SYSTEM_IREDEV:
        return "iredev_single_agent_baseline"
    return "quare_dialectic_scaffold_v1"


def _chunk_text(text: str) -> list[str]:
    """Split a source document into paragraph-like retrieval chunks."""

    normalized = text.replace("\r", "\n")
    raw_parts = re.split(r"\n\s*\n+", normalized)
    chunks: list[str] = []
    for part in raw_parts:
        compact = re.sub(r"\s+", " ", part).strip()
        if len(compact) < 80:
            continue
        if len(compact) > 800:
            for start in range(0, len(compact), 800):
                piece = compact[start : start + 800].strip()
                if len(piece) >= 80:
                    chunks.append(piece)
            continue
        chunks.append(compact)
    return chunks


def _rag_payload(*, query: str, rag_context: dict[str, Any]) -> dict[str, Any]:
    """Return RAG source metadata for one generated element."""

    if not bool(rag_context.get("rag_enabled", False)):
        return {
            "source": "openre_bench.phase1",
            "source_chunk_id": None,
            "source_document": None,
            "retrieved_chunks": [],
        }

    backend = str(rag_context.get("rag_backend", "")).lower()
    chroma_meta = rag_context.get("chroma_rag") or {}
    if backend in ("chroma_ada002", "chroma") and not chroma_meta.get("skipped") and chroma_meta.get(
        "collection_name"
    ):
        try:
            settings = load_openai_settings()
            api_key = settings.resolved_api_key
        except MissingAPIKeyError:
            return _rag_payload_tfidf_fallback(query=query, rag_context=rag_context)

        from openre_bench.verification.chroma_corpus_index import query_chroma_top_documents

        hits, _sims = query_chroma_top_documents(
            query=query,
            persist_path=str(chroma_meta["persist_path"]),
            collection_name=str(chroma_meta["collection_name"]),
            openai_api_key=api_key,
            top_k=3,
        )
        if not hits:
            return _rag_payload_tfidf_fallback(query=query, rag_context=rag_context)

        retrieved = [
            {
                "chunk_id": h["chunk_id"],
                "document": h["document"],
                "score": round(float(h.get("similarity", 0.0)), 6),
                "content": _summarize_text(h.get("text", ""), 280),
            }
            for h in hits
        ]
        best = hits[0]
        return {
            "source": "openre_bench.phase1.rag_chroma_ada002",
            "source_chunk_id": best["chunk_id"],
            "source_document": best["document"],
            "retrieved_chunks": retrieved,
        }

    return _rag_payload_tfidf_fallback(query=query, rag_context=rag_context)


def _rag_payload_tfidf_fallback(*, query: str, rag_context: dict[str, Any]) -> dict[str, Any]:
    chunks = list(rag_context.get("chunks", []))
    if not chunks:
        crs = rag_context.get("rag_corpus_dir")
        if crs:
            corpus_dir = Path(str(crs))
            if corpus_dir.exists():
                chunks = _load_rag_chunks(corpus_dir)
    if not chunks:
        return {
            "source": "openre_bench.phase1.rag_fallback",
            "source_chunk_id": None,
            "source_document": None,
            "retrieved_chunks": [],
        }

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for item in chunks:
        score = _chunk_overlap_score(query=query, chunk_tokens=item["tokens"])
        scored.append((score, item["chunk_id"], item))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    top = scored[:3]

    retrieved = [
        {
            "chunk_id": item["chunk_id"],
            "document": item["document"],
            "score": round(score, 6),
            "content": _summarize_text(item["text"], 280),
        }
        for score, _, item in top
    ]
    best_item = top[0][2]
    return {
        "source": "openre_bench.phase1.rag",
        "source_chunk_id": best_item["chunk_id"],
        "source_document": best_item["document"],
        "retrieved_chunks": retrieved,
    }



def _chunk_overlap_score(*, query: str, chunk_tokens: set[str]) -> float:
    """Compute deterministic lexical overlap score for retrieval ranking."""

    query_tokens = set(_tokens(query.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    return overlap / len(query_tokens)


# ---------------------------------------------------------------------------
# Shared LLM action runner (used by mare.py AND iredev.py)
# ---------------------------------------------------------------------------


@dataclass
class ActionRunResult:
    """Result from one LLM-first action with deterministic fallback."""

    items: list[str]
    summary: str
    execution_mode: str       # "llm" | "fallback"
    llm_generated: bool
    fallback_reason: str


def _run_llm_action_with_fallback(
    *,
    llm_client: Phase2LLMClient | None,
    messages: list[dict[str, str]],
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_source: str,
    llm_meta: MareRuntimeExecutionMeta,
    coerce_items_fn: Any,
    max_items: int,
    fallback_items: list[str],
) -> ActionRunResult:
    """Attempt an LLM call with retry; fall back to deterministic items on failure.

    Centralises the retry → parse → coerce → fallback loop previously duplicated
    inside the ``mare.py`` and ``iredev.py`` ``_run_action`` closures.
    """

    llm_items: list[str] = []
    action_summary = ""
    llm_generated = False
    fallback_reason = ""

    if llm_client is not None:
        max_attempts = max(1, PHASE2_LLM_RETRY_LIMIT + 1)
        for attempt in range(max_attempts):
            try:
                raw_response, seed_applied = _chat_with_optional_seed(
                    llm_client=llm_client,
                    messages=messages,
                    temperature=max(0.0, min(float(llm_temperature), 1.0)),
                    max_tokens=max(256, min(int(llm_max_tokens), 1200)),
                    seed=llm_seed,
                )
            except (LLMClientError, RuntimeError, ValueError) as exc:
                fallback_reason = f"request_failed: {exc}"
                continue

            try:
                payload, recovered = _parse_quare_llm_payload(raw_response)
            except ValueError as exc:
                fallback_reason = f"parse_failed: {exc}"
                if attempt < max_attempts - 1:
                    llm_meta.llm_retry_count += 1
                continue

            if recovered:
                llm_meta.llm_parse_recoveries += 1

            llm_items = coerce_items_fn(payload.get("items"), limit=max_items)
            action_summary = _coerce_non_empty_text(
                payload.get("summary"),
                fallback="",
            )
            if not llm_items:
                fallback_reason = "empty_items"
                if attempt < max_attempts - 1:
                    llm_meta.llm_retry_count += 1
                continue

            llm_generated = True
            llm_meta.llm_turns += 1
            if seed_applied:
                llm_meta.llm_seed_applied_turns += 1
            break

    if llm_generated:
        items = llm_items
        if not action_summary:
            action_summary = _summarize_text(" ".join(items), 160)
        execution_mode = "llm"
    else:
        llm_meta.llm_fallback_turns += 1
        items = fallback_items
        action_summary = _summarize_text(" ".join(items), 160)
        execution_mode = "fallback"

    return ActionRunResult(
        items=items,
        summary=action_summary,
        execution_mode=execution_mode,
        llm_generated=llm_generated,
        fallback_reason=fallback_reason,
    )


# ---------------------------------------------------------------------------
# MARE functions — delegated to openre_bench.pipeline.mare
# ---------------------------------------------------------------------------


def _build_phase1_mare_semantics(
    *,
    case: CaseInput,
    seed: int,
    setting: str,
    rag_context: dict[str, Any],
    llm_client: Phase2LLMClient | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], MareRuntimeExecutionMeta]:
    """Delegate to mare.py — MARE 5-agent/9-action workflow."""

    from openre_bench.pipeline.mare import build_phase1_mare_semantics as _impl

    return _impl(
        case=case,
        seed=seed,
        setting=setting,
        rag_context=rag_context,
        llm_client=llm_client,
        llm_source=llm_source,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
    )


def _run_mare_action_workflow(
    *,
    case_name: str,
    fragments: list[str],
    llm_client: Phase2LLMClient | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, Any], MareRuntimeExecutionMeta]:
    """Delegate to mare.py — MARE action workflow execution."""

    from openre_bench.pipeline.mare import run_mare_action_workflow as _impl

    return _impl(
        case_name=case_name,
        fragments=fragments,
        llm_client=llm_client,
        llm_source=llm_source,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
    )

# ---------------------------------------------------------------------------
# iReDev functions — delegated to openre_bench.pipeline.iredev
# ---------------------------------------------------------------------------


def _build_phase1_iredev_semantics(
    *,
    case: CaseInput,
    seed: int,
    setting: str,
    rag_context: dict[str, Any],
    llm_client: Phase2LLMClient | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], MareRuntimeExecutionMeta]:
    """Delegate to iredev.py — iReDev 6-agent/17-action workflow."""

    from openre_bench.pipeline.iredev import build_phase1_iredev_semantics as _impl

    return _impl(
        case=case,
        seed=seed,
        setting=setting,
        rag_context=rag_context,
        llm_client=llm_client,
        llm_source=llm_source,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
    )


def _run_iredev_action_workflow(
    *,
    case_name: str,
    fragments: list[str],
    llm_client: Phase2LLMClient | None,
    llm_source: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
) -> tuple[dict[str, Any], MareRuntimeExecutionMeta]:
    """Delegate to iredev.py — iReDev action workflow execution."""

    from openre_bench.pipeline.iredev import run_iredev_action_workflow as _impl

    return _impl(
        case_name=case_name,
        fragments=fragments,
        llm_client=llm_client,
        llm_source=llm_source,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
    )


def _sha256_payload(payload: dict[str, Any]) -> str:
    """Return deterministic SHA256 digest for JSON-like payload."""

    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_phase1(
    case: CaseInput,
    seed: int,
    setting: str,
    rag_context: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build Phase 1 initial requirement models."""

    fragments = _extract_requirement_fragments(case.requirement)
    if not fragments:
        fragments = [case.requirement.strip() or "Requirement text unavailable"]

    rotated = _rotate_fragments(fragments, seed)
    phase1: dict[str, list[dict[str, Any]]] = {}
    agent_quality_mapping = _agent_quality_mapping_for_setting(setting)

    if setting == SETTING_SINGLE_AGENT:
        agent_name = "SingleAgent"
        quality_attribute = "Integrated"
        root_id = "INT-L1-001"
        root_rag_payload = _rag_payload(
            query=" ".join(rotated),
            rag_context=rag_context,
        )
        root = KAOSElement(
            id=root_id,
            name="Integrated Goal",
            description=(
                f"Integrated single-agent objective for {case.case_name}: "
                f"{_summarize_text(' '.join(rotated), 220)}"
            ),
            element_type="Goal",
            quality_attribute=quality_attribute,
            hierarchy_level=1,
            stakeholder=agent_name,
            measurable_criteria="Integrated coverage across all requirement clauses",
            source="openre_bench.phase1.rag" if rag_context["rag_enabled"] else "openre_bench.phase1",
            source_chunk_id=root_rag_payload["source_chunk_id"],
            source_document=root_rag_payload["source_document"],
            retrieved_chunks=root_rag_payload["retrieved_chunks"],
            validation_status="pending",
        )
        single_elements: list[dict[str, Any]] = [root.model_dump(mode="json")]
        for leaf_index, fragment in enumerate(rotated[:6], start=1):
            rag_payload = _rag_payload(query=fragment, rag_context=rag_context)
            leaf = KAOSElement(
                id=f"INT-L2-{leaf_index:03d}",
                name=f"Integrated Requirement {leaf_index}",
                description=f"The system shall satisfy: {_summarize_text(fragment, 220)}",
                element_type="Task",
                quality_attribute=quality_attribute,
                hierarchy_level=2,
                parent_goal_id=root_id,
                stakeholder=agent_name,
                measurable_criteria="Clause-level integrated requirement",
                source=rag_payload["source"],
                source_chunk_id=rag_payload["source_chunk_id"],
                source_document=rag_payload["source_document"],
                retrieved_chunks=rag_payload["retrieved_chunks"],
                validation_status="pending",
            )
            single_elements.append(leaf.model_dump(mode="json"))
        phase1[agent_name] = single_elements
        return phase1

    agent_items = list(agent_quality_mapping.items())
    base_multi_agent_leaf_count = 6
    if len(agent_items) == 5:
        # Keep the same total volume (30 leaves + 5 roots = 35) while avoiding
        # fixed per-axis 7-count outputs in QUARE multi-agent settings.
        delta_pattern = [1, 1, 0, -1, -1]
        shift = seed % len(delta_pattern)
        leaf_deltas = delta_pattern[shift:] + delta_pattern[:shift]
    else:
        leaf_deltas = [0] * len(agent_items)
    for agent_index, (agent_name, quality_attribute) in enumerate(agent_items, start=1):
        root_fragment_index = _agent_fragment_start_index(
            fragment_count=len(rotated),
            agent_index=agent_index,
            total_agents=len(agent_items),
        )
        root_rag_payload = _rag_payload(
            query=rotated[root_fragment_index],
            rag_context=rag_context,
        )
        root_id = f"{quality_attribute[:3].upper()}-L1-{agent_index:03d}"
        root = KAOSElement(
            id=root_id,
            name=f"{quality_attribute} Goal",
            description=(
                f"{quality_attribute} objective for {case.case_name}: "
                f"{_summarize_text(rotated[root_fragment_index], 220)}"
            ),
            element_type="Goal",
            quality_attribute=quality_attribute,
            hierarchy_level=1,
            stakeholder=agent_name,
            measurable_criteria="Quality-attribute specific requirement analysis",
            source=root_rag_payload["source"],
            source_chunk_id=root_rag_payload["source_chunk_id"],
            source_document=root_rag_payload["source_document"],
            retrieved_chunks=root_rag_payload["retrieved_chunks"],
            validation_status="pending",
        )

        multi_agent_leaf_count = max(
            1,
            base_multi_agent_leaf_count + leaf_deltas[agent_index - 1],
        )
        assigned_fragments = _agent_fragment_window(
            rotated=rotated,
            agent_index=agent_index,
            total_agents=len(agent_items),
            leaf_count=multi_agent_leaf_count,
        )
        elements: list[dict[str, Any]] = [root.model_dump(mode="json")]
        for leaf_index, fragment in enumerate(assigned_fragments[:multi_agent_leaf_count], start=1):
            lens_phrase = _quality_lens_phrase(
                quality_attribute=quality_attribute,
                leaf_index=leaf_index,
            )
            rag_payload = _rag_payload(query=fragment, rag_context=rag_context)
            leaf = KAOSElement(
                id=f"{quality_attribute[:3].upper()}-L2-{agent_index:03d}-{leaf_index:02d}",
                name=f"{quality_attribute} Requirement {leaf_index}",
                description=(
                    f"The system shall ensure {quality_attribute.lower()} ({lens_phrase}): "
                    f"{_summarize_text(fragment, 220)}"
                ),
                element_type="Task",
                quality_attribute=quality_attribute,
                hierarchy_level=2,
                parent_goal_id=root_id,
                stakeholder=agent_name,
                measurable_criteria=f"Clause-level quality requirement for {lens_phrase}",
                source=rag_payload["source"],
                source_chunk_id=rag_payload["source_chunk_id"],
                source_document=rag_payload["source_document"],
                retrieved_chunks=rag_payload["retrieved_chunks"],
                validation_status="pending",
            )
            elements.append(leaf.model_dump(mode="json"))

        phase1[agent_name] = elements

    return phase1


def _build_phase2(
    *,
    run_id: str,
    phase1: dict[str, list[dict[str, Any]]],
    setting: str,
    requirement: str,
    system: str,
    round_cap: int,
    llm_client: Phase2LLMClient | None,
    llm_source: str,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    parallel: bool = True,
    llm_max_concurrency: int = 10,
    llm_backoff_cap_seconds: float = 60.0,
    llm_backoff_base_seconds: float = 1.0,
    resolution_mode: str = "original",
    parser_temperature: float = 0.7,
    attack_detection_temperature: float = 1.0,
    attack_confidence_threshold: float = 0.7,
    attack_llm_confidence_floor: float = 0.85,
    attack_detection_mode: str = "full",
    priority_weights: dict[str, float] | None = None,
    paper_bert_conflict_prescreen: bool = False,
    paper_bert_similarity_tau: float = 0.85,
    paper_phase2_llm_pair_classification: bool = False,
    paper_pair_classification_temperature: float = 0.7,
) -> tuple[dict[str, Any], Phase2ExecutionMeta, dict[str, Any] | None]:
    """Build Phase 2 negotiation trace with system-specific behavior."""

    agents = list(phase1.keys())
    if not _negotiation_enabled(setting):
        artifact = Phase2Artifact(
            total_negotiations=0,
            negotiations={},
            summary_stats={
                "total_steps": 0,
                "successful_consensus": 0,
                "average_rounds": 0.0,
                "detected_conflicts": 0,
                "resolved_conflicts": 0,
                "llm_enabled": False,
                "llm_turns": 0,
                "llm_fallback_turns": 0,
                "llm_retry_count": 0,
                "llm_parse_recoveries": 0,
                "llm_seed_applied_turns": 0,
                "llm_source": "disabled",
            },
        )
        return artifact.model_dump(mode="json"), Phase2ExecutionMeta(), None

    negotiation_map: dict[str, NegotiationHistory] = {}
    step_counter = 1
    detected_conflicts = 0
    resolved_conflicts = 0
    llm_turns = 0
    llm_fallback_turns = 0
    llm_retry_count = 0
    llm_parse_recoveries = 0
    llm_seed_applied_turns = 0
    round_cap_hits = 0

    pair_jobs: list[dict[str, Any]] = []
    for index, focus_agent in enumerate(agents):
        reviewer_agent = agents[(index + 1) % len(agents)]
        pair_jobs.append(
            {
                "focus_agent": focus_agent,
                "reviewer_agent": reviewer_agent,
                "pair_key": f"{focus_agent}_{reviewer_agent}",
                "focus_elements": [dict(item) for item in phase1[focus_agent]],
                "reviewer_elements": [dict(item) for item in phase1[reviewer_agent]],
            }
        )

    pair_results: list[tuple[str, _NegotiationBuildResult]] = []
    if system == SYSTEM_QUARE and parallel and pair_jobs:
        pair_results = _run_parallel_quare_negotiations(
            run_id=run_id,
            pair_jobs=pair_jobs,
            requirement=requirement,
            setting=setting,
            round_cap=round_cap,
            llm_client=llm_client,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_max_concurrency=llm_max_concurrency,
            llm_backoff_cap_seconds=llm_backoff_cap_seconds,
            llm_backoff_base_seconds=llm_backoff_base_seconds,
            paper_bert_conflict_prescreen=paper_bert_conflict_prescreen,
            paper_bert_similarity_tau=paper_bert_similarity_tau,
            paper_phase2_llm_pair_classification=paper_phase2_llm_pair_classification,
            paper_pair_classification_temperature=paper_pair_classification_temperature,
        )
    else:
        for job in pair_jobs:
            if system == SYSTEM_QUARE:
                result = _build_quare_negotiation_history(
                    run_id=run_id,
                    pair_key=job["pair_key"],
                    focus_agent=job["focus_agent"],
                    reviewer_agent=job["reviewer_agent"],
                    focus_elements=job["focus_elements"],
                    reviewer_elements=job["reviewer_elements"],
                    requirement=requirement,
                    setting=setting,
                    round_cap=round_cap,
                    step_counter=step_counter,
                    llm_client=llm_client,
                    llm_model=llm_model,
                    llm_temperature=llm_temperature,
                    llm_max_tokens=llm_max_tokens,
                    llm_seed=llm_seed,
                    llm_max_concurrency=llm_max_concurrency,
                    llm_backoff_cap_seconds=llm_backoff_cap_seconds,
                    llm_backoff_base_seconds=llm_backoff_base_seconds,
                    paper_bert_conflict_prescreen=paper_bert_conflict_prescreen,
                    paper_bert_similarity_tau=paper_bert_similarity_tau,
                    paper_phase2_llm_pair_classification=paper_phase2_llm_pair_classification,
                    paper_pair_classification_temperature=paper_pair_classification_temperature,
                )
            else:
                # iReDev (Jin et al., TOSEM 2025) and MARE share baseline single-round
                # negotiation.  iReDev's paper focuses on elicitation and specification
                # and does not define its own negotiation protocol, so we reuse the
                # MARE negotiation scaffold for Phase 2 comparability.
                result = _build_mare_negotiation_history(
                    run_id=run_id,
                    pair_key=job["pair_key"],
                    focus_agent=job["focus_agent"],
                    reviewer_agent=job["reviewer_agent"],
                    focus_elements=job["focus_elements"],
                    reviewer_elements=job["reviewer_elements"],
                    requirement=requirement,
                    step_counter=step_counter,
                    paper_bert_conflict_prescreen=paper_bert_conflict_prescreen,
                    paper_bert_similarity_tau=paper_bert_similarity_tau,
                )
            pair_results.append((job["pair_key"], result))
            step_counter = result.next_step_id

    for pair_key, result in pair_results:
        negotiation_map[pair_key] = result.history
        step_counter = result.next_step_id
        if result.conflict_detected:
            detected_conflicts += 1
        if result.conflict_resolved:
            resolved_conflicts += 1
        llm_turns += result.llm_turns
        llm_fallback_turns += result.llm_fallback_turns
        llm_retry_count += result.llm_retry_count
        llm_parse_recoveries += result.llm_parse_recoveries
        llm_seed_applied_turns += result.llm_seed_applied_turns
        round_cap_hits += result.round_cap_hits

    argumentation_graph_payload: dict[str, Any] | None = None
    if (
        system == SYSTEM_QUARE
        and _negotiation_enabled(setting)
        and resolution_mode in {"af_grounded", "af_preferred"}
    ):
        af_result = _run_phase2_argumentation_layer(
            negotiation_map=negotiation_map,
            llm_client=llm_client,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_max_concurrency=llm_max_concurrency,
            llm_backoff_cap_seconds=llm_backoff_cap_seconds,
            llm_backoff_base_seconds=llm_backoff_base_seconds,
            resolution_mode=resolution_mode,
            parser_temperature=parser_temperature,
            attack_detection_temperature=attack_detection_temperature,
            attack_confidence_threshold=attack_confidence_threshold,
            attack_llm_confidence_floor=attack_llm_confidence_floor,
            attack_detection_mode=attack_detection_mode,
            priority_weights=priority_weights or {},
        )
        if af_result["selected_elements"]:
            negotiation_map = _inject_argumentation_selection_into_negotiation_map(
                negotiation_map=negotiation_map,
                selected_elements=af_result["selected_elements"],
            )
        argumentation_graph_payload = af_result["graph_payload"]

    round_counts = [item.total_rounds for item in negotiation_map.values()]
    average_rounds = round(sum(round_counts) / len(round_counts), 3) if round_counts else 0.0
    llm_enabled = system == SYSTEM_QUARE

    artifact = Phase2Artifact(
        total_negotiations=len(negotiation_map),
        negotiations=negotiation_map,
        summary_stats={
            "total_steps": sum(len(item.steps) for item in negotiation_map.values()),
            "successful_consensus": sum(
                1 for item in negotiation_map.values() if item.final_consensus
            ),
            "average_rounds": average_rounds,
            "detected_conflicts": detected_conflicts,
            "resolved_conflicts": resolved_conflicts,
            "llm_enabled": llm_enabled,
            "llm_turns": llm_turns,
            "llm_fallback_turns": llm_fallback_turns,
            "llm_retry_count": llm_retry_count,
            "llm_parse_recoveries": llm_parse_recoveries,
            "llm_seed_applied_turns": llm_seed_applied_turns,
            "round_cap_hits": round_cap_hits,
            "llm_source": llm_source if llm_enabled else "disabled",
        },
    )
    return (
        artifact.model_dump(mode="json"),
        Phase2ExecutionMeta(
            llm_enabled=llm_enabled,
            llm_turns=llm_turns,
            llm_fallback_turns=llm_fallback_turns,
            llm_retry_count=llm_retry_count,
            llm_parse_recoveries=llm_parse_recoveries,
            llm_seed_applied_turns=llm_seed_applied_turns,
            llm_source=llm_source if llm_enabled else "disabled",
        ),
        argumentation_graph_payload,
    )


def _run_phase2_argumentation_layer(
    *,
    negotiation_map: dict[str, NegotiationHistory],
    llm_client: Phase2LLMClient | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_max_concurrency: int,
    llm_backoff_cap_seconds: float,
    llm_backoff_base_seconds: float,
    resolution_mode: str,
    parser_temperature: float,
    attack_detection_temperature: float,
    attack_confidence_threshold: float,
    attack_llm_confidence_floor: float,
    attack_detection_mode: str,
    priority_weights: dict[str, float],
) -> dict[str, Any]:
    """Run argument extraction + attack construction + AF solving for phase2."""

    arguments, parse_meta = parse_phase2_arguments(
        negotiation_map=negotiation_map,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_temperature=parser_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
        llm_max_concurrency=llm_max_concurrency,
        llm_backoff_cap_seconds=llm_backoff_cap_seconds,
        llm_backoff_base_seconds=llm_backoff_base_seconds,
    )
    attacks, attack_meta = build_attack_relations(
        arguments=arguments,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_temperature=attack_detection_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
        llm_max_concurrency=llm_max_concurrency,
        llm_backoff_cap_seconds=llm_backoff_cap_seconds,
        llm_backoff_base_seconds=llm_backoff_base_seconds,
        confidence_threshold=attack_confidence_threshold,
        attack_detection_mode=attack_detection_mode,
        llm_confidence_floor=attack_llm_confidence_floor,
    )
    strategy = "grounded" if resolution_mode == "af_grounded" else "preferred_priority"
    af_solution = solve_argumentation_framework(
        arguments=arguments,
        attacks=attacks,
        priority_weights=priority_weights,
        strategy=strategy,
    )
    selected_ids = set(af_solution.selected_extension)
    selected_elements = _selected_elements_from_extension(
        arguments=arguments,
        selected_ids=selected_ids,
    )
    _assert_phase3_input_compatible_elements(selected_elements)
    return {
        "selected_elements": selected_elements,
        "graph_payload": {
            "arguments": [asdict(item) for item in arguments],
            "attacks": [asdict(item) for item in attacks],
            "grounded_extension": list(af_solution.grounded_extension),
            "preferred_extensions": [list(item) for item in af_solution.preferred_extensions],
            "selected_extension": list(af_solution.selected_extension),
            "meta": {
                "argument_parse": asdict(parse_meta),
                "attack_build": asdict(attack_meta),
                "strategy": af_solution.selection_strategy,
            },
        },
    }


def _selected_elements_from_extension(
    *,
    arguments: list[Any],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    merged: dict[str, tuple[int, dict[str, Any]]] = {}
    for argument in sorted(arguments, key=lambda item: (item.step_id, item.argument_id)):
        if argument.argument_id not in selected_ids:
            continue
        if argument.turn_type == "critique":
            continue
        for element in argument.kaos_elements:
            if not isinstance(element, dict):
                continue
            element_id = str(element.get("id", "")).strip()
            if not element_id:
                continue
            candidate = (int(argument.step_id), dict(element))
            existing = merged.get(element_id)
            if existing is None or candidate[0] >= existing[0]:
                merged[element_id] = candidate
    return [merged[item_id][1] for item_id in sorted(merged)]


def _inject_argumentation_selection_into_negotiation_map(
    *,
    negotiation_map: dict[str, NegotiationHistory],
    selected_elements: list[dict[str, Any]],
) -> dict[str, NegotiationHistory]:
    """Inject AF-selected elements as latest backward step payload for phase3 extraction."""

    if not selected_elements:
        return negotiation_map
    updated: dict[str, NegotiationHistory] = {}
    for pair_key, history in negotiation_map.items():
        dumped = history.model_dump(mode="json")
        steps = dumped.get("steps", [])
        if isinstance(steps, list) and steps:
            for idx in range(len(steps) - 1, -1, -1):
                step = steps[idx]
                if isinstance(step, dict) and step.get("message_type") == "backward":
                    step["kaos_elements"] = [dict(item) for item in selected_elements]
                    steps[idx] = step
                    break
        updated[pair_key] = NegotiationHistory.model_validate(dumped)
    return updated


def _assert_phase3_input_compatible_elements(elements: list[dict[str, Any]]) -> None:
    """Ensure AF-selected elements keep the schema required by phase3 builders."""

    required_keys = {"id", "name", "description"}
    for element in elements:
        missing = [key for key in required_keys if key not in element or not str(element.get(key, "")).strip()]
        if missing:
            raise ValueError(
                "AF-selected elements are not phase3-compatible; missing keys: "
                f"{missing} in element {element}"
            )


def _run_parallel_quare_negotiations(
    *,
    run_id: str,
    pair_jobs: list[dict[str, Any]],
    requirement: str,
    setting: str,
    round_cap: int,
    llm_client: Phase2LLMClient | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_max_concurrency: int,
    llm_backoff_cap_seconds: float,
    llm_backoff_base_seconds: float,
    paper_bert_conflict_prescreen: bool = False,
    paper_bert_similarity_tau: float = 0.85,
    paper_phase2_llm_pair_classification: bool = False,
    paper_pair_classification_temperature: float = 0.7,
) -> list[tuple[str, _NegotiationBuildResult]]:
    """Run QUARE pairwise negotiations concurrently and keep deterministic order."""

    async def _runner() -> list[tuple[str, _NegotiationBuildResult]]:
        semaphore = asyncio.Semaphore(max(1, int(llm_max_concurrency)))

        async def _build_one(job: dict[str, Any]) -> tuple[str, _NegotiationBuildResult]:
            async with semaphore:
                result = await asyncio.to_thread(
                    _build_quare_negotiation_history,
                    run_id=run_id,
                    pair_key=job["pair_key"],
                    focus_agent=job["focus_agent"],
                    reviewer_agent=job["reviewer_agent"],
                    focus_elements=job["focus_elements"],
                    reviewer_elements=job["reviewer_elements"],
                    requirement=requirement,
                    setting=setting,
                    round_cap=round_cap,
                    step_counter=1,
                    llm_client=llm_client,
                    llm_model=llm_model,
                    llm_temperature=llm_temperature,
                    llm_max_tokens=llm_max_tokens,
                    llm_seed=llm_seed,
                    llm_max_concurrency=llm_max_concurrency,
                    llm_backoff_cap_seconds=llm_backoff_cap_seconds,
                    llm_backoff_base_seconds=llm_backoff_base_seconds,
                    paper_bert_conflict_prescreen=paper_bert_conflict_prescreen,
                    paper_bert_similarity_tau=paper_bert_similarity_tau,
                    paper_phase2_llm_pair_classification=paper_phase2_llm_pair_classification,
                    paper_pair_classification_temperature=paper_pair_classification_temperature,
                )
            return job["pair_key"], result

        tasks = [_build_one(job) for job in pair_jobs]
        gathered = await asyncio.gather(*tasks)
        order = {job["pair_key"]: idx for idx, job in enumerate(pair_jobs)}
        return sorted(gathered, key=lambda item: order[item[0]])

    return asyncio.run(_runner())


@dataclass
class _NegotiationBuildResult:
    """Internal return type used by system-specific phase2 builders."""

    history: NegotiationHistory
    next_step_id: int
    conflict_detected: bool
    conflict_resolved: bool
    llm_turns: int = 0
    llm_fallback_turns: int = 0
    llm_retry_count: int = 0
    llm_parse_recoveries: int = 0
    llm_seed_applied_turns: int = 0
    round_cap_hits: int = 0


def _build_mare_negotiation_history(
    *,
    run_id: str,
    pair_key: str,
    focus_agent: str,
    reviewer_agent: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    requirement: str,
    step_counter: int,
    paper_bert_conflict_prescreen: bool = False,
    paper_bert_similarity_tau: float = 0.85,
) -> _NegotiationBuildResult:
    """Build baseline-faithful MARE single-turn negotiation history."""

    conflict_detected = _detect_conflict(
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        focus_elements=focus_elements,
        reviewer_elements=reviewer_elements,
        requirement=requirement,
        paper_bert_conflict_prescreen=paper_bert_conflict_prescreen,
        paper_bert_similarity_tau=paper_bert_similarity_tau,
    )
    negotiated_elements = _apply_negotiation_adjustments(
        elements=focus_elements,
        reviewer_agent=reviewer_agent,
        conflict_detected=conflict_detected,
    )
    conflict_resolved = conflict_detected

    forward_text = f"{focus_agent} proposes initial model for peer review."
    forward_step = NegotiationStep(
        step_id=step_counter,
        timestamp=utc_timestamp(),
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        round_number=1,
        message_type="forward",
        kaos_elements=focus_elements,
        analysis_text=forward_text,
        analysis=forward_text,
        conflict_detected=conflict_detected,
    )
    step_counter += 1

    backward_text = _backward_analysis_text(reviewer_agent, conflict_detected)
    backward_step = NegotiationStep(
        step_id=step_counter,
        timestamp=utc_timestamp(),
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        round_number=1,
        message_type="backward",
        kaos_elements=negotiated_elements,
        analysis_text=backward_text,
        analysis=backward_text,
        feedback=_backward_feedback(conflict_detected),
        conflict_detected=conflict_detected,
    )
    step_counter += 1

    history = NegotiationHistory(
        negotiation_id=f"neg_{pair_key}_{run_id}",
        focus_agent=focus_agent,
        reviewer_agents=[reviewer_agent],
        start_timestamp=utc_timestamp(),
        end_timestamp=utc_timestamp(),
        steps=[forward_step, backward_step],
        final_consensus=conflict_resolved or not conflict_detected,
        total_rounds=1,
    )

    return _NegotiationBuildResult(
        history=history,
        next_step_id=step_counter,
        conflict_detected=conflict_detected,
        conflict_resolved=conflict_resolved,
    )



# ---------------------------------------------------------------------------
# QUARE functions — delegated to openre_bench.pipeline.quare
# ---------------------------------------------------------------------------


def _build_quare_negotiation_history(
    *,
    run_id: str,
    pair_key: str,
    focus_agent: str,
    reviewer_agent: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    requirement: str,
    setting: str,
    round_cap: int,
    step_counter: int,
    llm_client: Phase2LLMClient | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_max_concurrency: int = 10,
    llm_backoff_cap_seconds: float = 60.0,
    llm_backoff_base_seconds: float = 1.0,
    paper_bert_conflict_prescreen: bool = False,
    paper_bert_similarity_tau: float = 0.85,
    paper_phase2_llm_pair_classification: bool = False,
    paper_pair_classification_temperature: float = 0.7,
) -> _NegotiationBuildResult:
    """Delegate to quare.py — QUARE multi-turn dialectic negotiation."""

    from openre_bench.pipeline.quare import build_quare_negotiation_history as _impl

    return _impl(
        run_id=run_id,
        pair_key=pair_key,
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        focus_elements=focus_elements,
        reviewer_elements=reviewer_elements,
        requirement=requirement,
        setting=setting,
        round_cap=round_cap,
        step_counter=step_counter,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_seed=llm_seed,
        llm_max_concurrency=llm_max_concurrency,
        llm_backoff_cap_seconds=llm_backoff_cap_seconds,
        llm_backoff_base_seconds=llm_backoff_base_seconds,
        paper_bert_conflict_prescreen=paper_bert_conflict_prescreen,
        paper_bert_similarity_tau=paper_bert_similarity_tau,
        paper_phase2_llm_pair_classification=paper_phase2_llm_pair_classification,
        paper_pair_classification_temperature=paper_pair_classification_temperature,
    )


def _build_phase3(
    *,
    run_id: str,
    case: CaseInput,
    phase1: dict[str, list[dict[str, Any]]],
    phase2: dict[str, Any],
    setting: str,
) -> dict[str, Any]:
    """Build Phase 3 integrated model from Phase 1/2 artifacts."""

    source_elements = _phase3_source_elements(phase1=phase1, phase2=phase2, setting=setting)

    gsn_elements: list[GSNElement] = []
    for element in source_elements:
        gsn_elements.append(
            GSNElement(
                id=element["id"],
                name=element["name"],
                description=element["description"],
                gsn_type="Goal" if element.get("element_type") == "Goal" else "Strategy",
                quality_attribute=element.get("quality_attribute", "Integrated"),
                priority=int(element.get("priority", 1)),
                stakeholder=element.get("stakeholder"),
                measurable_criteria=element.get("measurable_criteria"),
                hierarchy_level=int(element.get("hierarchy_level", 1)),
                parent_goal_id=element.get("parent_goal_id"),
                properties={
                    "original_kaos_type": element.get("element_type", "Goal"),
                    "source": element.get("source"),
                    "source_document": element.get("source_document"),
                    "conflict_resolved_by": element.get("conflict_resolved_by"),
                    "validation_status": element.get("validation_status"),
                    "citation_required": element.get("citation_required", True),
                },
            )
        )

    gsn_connections: list[GSNConnection] = []
    for element in source_elements:
        parent_id = element.get("parent_goal_id")
        if not parent_id:
            continue
        gsn_connections.append(
            GSNConnection(
                id=f"REL-{len(gsn_connections) + 1:03d}",
                source_id=parent_id,
                target_id=element["id"],
                connection_type="SupportedBy",
                description="Derived from parent-child refinement",
                properties={
                    "original_kaos_relation": "AND-refinement",
                },
            )
        )

    topology_status = _compute_topology_status([item.model_dump(mode="json") for item in gsn_elements])

    hierarchy_structure = {
        "level_1": [
            _hierarchy_view(item)
            for item in source_elements
            if int(item.get("hierarchy_level", 1)) == 1
        ],
        "level_2": [
            _hierarchy_view(item)
            for item in source_elements
            if int(item.get("hierarchy_level", 1)) == 2
        ],
        "level_3": [
            _hierarchy_view(item)
            for item in source_elements
            if int(item.get("hierarchy_level", 1)) >= 3
        ],
        "parent_child_mappings": _parent_child_mappings(source_elements),
    }

    phase3 = Phase3Artifact(
        gsn_elements=gsn_elements,
        gsn_connections=gsn_connections,
        model_metadata={
            "model_id": f"model_{run_id}",
            "name": f"OpenRE-Bench Integrated KAOS for {case.case_name}",
            "description": "Deterministic protocol-aligned integrated GSN model.",
            "created_timestamp": utc_timestamp(),
            "requirement_source": case.requirement,
            "model_type": "GSN",
            "quality_attributes": sorted(
                {item.get("quality_attribute", "Integrated") for item in source_elements}
            ),
            "hierarchy_levels": sorted(
                {int(item.get("hierarchy_level", 1)) for item in source_elements}
            ),
            "setting": setting,
        },
        hierarchy_structure=hierarchy_structure,
        topology_status=topology_status,
        model_id=f"model_{run_id}",
        created_timestamp=utc_timestamp(),
        total_elements=len(gsn_elements),
        total_relations=len(gsn_connections),
    )
    return phase3.model_dump(mode="json")


def _build_phase4(
    *,
    phase3_payload: dict[str, Any],
    requirement: str,
    setting: str,
    rag_corpus_dir: Path | None = None,
    chroma_persist_dir: Path | None = None,
    paper_chroma_hallucination_layer: bool = False,
    paper_hallucination_tau_h: float = 0.60,
    paper_llm_compliance_entailment: bool = False,
    llm_client: Phase2LLMClient | None = None,
    llm_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Build Phase 4 verification report."""

    gsn_elements = phase3_payload.get("gsn_elements", [])
    topology_status = phase3_payload.get("topology_status", {})

    logical = _logical_consistency(gsn_elements)
    terminology = _terminology_consistency(gsn_elements)
    paper_phase4: dict[str, Any] = {
        "chroma_layer2": None,
        "compliance_method": "text_overlap",
    }
    if paper_llm_compliance_entailment and llm_client is not None:
        from openre_bench.verification.llm_entailment import compliance_coverage_llm_entailment

        clauses = _extract_requirement_fragments(requirement)
        if not clauses:
            clauses = [requirement.strip()] if requirement.strip() else []
        req_texts = [
            f"{item.get('name', '')}. {item.get('description', '')}".strip()
            for item in gsn_elements
            if isinstance(item, dict)
        ]
        coverage = compliance_coverage_llm_entailment(
            clauses=clauses,
            requirement_texts=req_texts,
            llm_client=llm_client,
            llm_model=llm_model,
            temperature=0.0,
            max_tokens=256,
        )
        paper_phase4["compliance_method"] = "llm_entailment"
    else:
        coverage = _compliance_coverage(gsn_elements, requirement)
        if paper_llm_compliance_entailment and llm_client is None:
            paper_phase4["compliance_note"] = "llm_entailment_requested_but_no_client_fallback_overlap"

    fact_checking = _fact_checking(gsn_elements)
    if paper_chroma_hallucination_layer:
        from openre_bench.verification.chroma_hallucination import chroma_hallucination_pass

        corpus = (rag_corpus_dir or Path("../OpenRE-Bench/data/knowledge_base")).resolve()
        persist = (chroma_persist_dir or Path(".chroma_openre_bench")).resolve()
        api_key = ""
        try:
            settings = load_openai_settings()
            api_key = settings.resolved_api_key
        except MissingAPIKeyError:
            api_key = ""
        if api_key:
            chroma_report = chroma_hallucination_pass(
                gsn_elements=gsn_elements,
                corpus_dir=corpus,
                persist_root=persist,
                openai_api_key=api_key,
                tau_h=float(paper_hallucination_tau_h),
            )
            paper_phase4["chroma_layer2"] = chroma_report
            flagged = [
                entry
                for entry in chroma_report.get("per_goal", [])
                if isinstance(entry, dict) and entry.get("flagged_low_support")
            ]
            existing = list(fact_checking.get("hallucination_reports", []))
            for entry in flagged:
                existing.append(
                    {
                        "element_id": entry.get("element_id"),
                        "issue": "low_corpus_support_chroma",
                        "best_similarity": entry.get("best_similarity"),
                        "tau_h": entry.get("tau_h"),
                    }
                )
            fact_checking["hallucination_reports"] = existing
            fact_checking["chroma_hallucination_layer"] = chroma_report
        else:
            paper_phase4["chroma_layer2"] = {
                "enabled": True,
                "skipped": True,
                "skip_reason": "missing_openai_api_key",
            }
    deterministic_validation = _build_deterministic_validation(
        topology_status=topology_status,
        logical=logical,
        verification_executed=_verification_executed(setting),
    )
    universal_verification = {
        "verification_summary": {
            "total_constraints": len(coverage["clauses"]),
            "total_conflicts": len(logical["contradictions"]),
            "critical_conflicts": len(logical["contradictions"]),
            "quality_score": round(100 * logical["normalized_score"], 2),
            "verification_passed": bool(deterministic_validation.get("is_valid", False)),
        },
        "constraint_analysis": {
            "extracted_constraints": coverage["clauses"],
        },
        "conflict_analysis": {
            "detected_conflicts": logical["contradictions"],
        },
        "resolutions": {
            "applied": [],
        },
        "recommendations": logical["recommendations"],
        "session_log": {
            "status": "passed" if _verification_executed(setting) else "not_executed",
        },
    }

    verification_results = {
        "fact_checking": fact_checking,
        "deterministic_validation": deterministic_validation,
        "topology_status": topology_status,
        "consistency_verification": {
            "status": "passed" if _verification_executed(setting) else "not_executed",
            "issues": logical["contradictions"],
            "logical_score": logical["score"],
            "logical_score_normalized": logical["normalized_score"],
            "terminology_score": terminology["score"],
            "terminology_consistency_ratio": terminology["consistency_ratio"],
        },
        "universal_verification": universal_verification,
        "compliance_coverage": coverage,
        "terminology_consistency": terminology,
        "s_logic": logical["normalized_score"],
        "s_term": terminology["consistency_ratio"],
        "paper_phase4": paper_phase4,
    }

    phase4 = Phase4Artifact(
        verification_results=verification_results,
        fact_checking=fact_checking,
        deterministic_validation=deterministic_validation,
        topology_status=topology_status,
        consistency_verification=verification_results["consistency_verification"],
        universal_verification=universal_verification,
        correction_summary={
            "total_corrections": len(logical["contradictions"]),
            "corrections": [],
        },
    )
    return phase4.model_dump(mode="json")



def _build_quare_optional_artifacts(
    *,
    case: CaseInput,
    phase2: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    setting: str,
    round_cap: int,
    rag_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Delegate to quare.py — QUARE optional artifacts (phase 0, 2.5, 5)."""

    from openre_bench.pipeline.quare import build_quare_optional_artifacts as _impl

    return _impl(
        case=case,
        phase2=phase2,
        phase3=phase3,
        phase4=phase4,
        setting=setting,
        round_cap=round_cap,
        rag_context=rag_context,
    )



def default_run_id(case_name: str, seed: int) -> str:
    """Build a readable run id for one-case scaffold runs."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", case_name.strip().lower()).strip("-")
    return f"{normalized}-{utc_timestamp().replace(':', '').replace('-', '')}-s{seed:03d}"


def _rotate_fragments(fragments: list[str], seed: int) -> list[str]:
    """Apply deterministic rotation without changing fragment content."""

    if not fragments:
        return []
    offset = seed % len(fragments)
    return fragments[offset:] + fragments[:offset]


def _agent_fragment_window(
    *,
    rotated: list[str],
    agent_index: int,
    total_agents: int,
    leaf_count: int,
) -> list[str]:
    """Return deterministic per-agent fragment windows with fixed leaf count."""

    if not rotated:
        return []
    fragment_count = len(rotated)
    window_size = max(1, leaf_count)
    start = _agent_fragment_start_index(
        fragment_count=fragment_count,
        agent_index=agent_index,
        total_agents=total_agents,
    )
    selected: list[str] = []
    for offset in range(window_size):
        cursor = (start + offset) % fragment_count
        selected.append(rotated[cursor])
    return selected


def _agent_fragment_start_index(*, fragment_count: int, agent_index: int, total_agents: int) -> int:
    """Compute an evenly distributed start index for one agent fragment window."""

    if fragment_count <= 0:
        return 0
    agent_slot = max(0, agent_index - 1)
    safe_total_agents = max(1, total_agents)
    return ((agent_slot * fragment_count) // safe_total_agents) % fragment_count


def _quality_lens_phrase(*, quality_attribute: str, leaf_index: int) -> str:
    """Return deterministic quality-lens phrasing to diversify multi-agent outputs."""

    cues = QUALITY_LENS_CUES.get(quality_attribute, QUALITY_LENS_CUES["Integrated"])
    index = max(0, leaf_index - 1) % len(cues)
    return cues[index]


def _hierarchy_view(item: dict[str, Any]) -> dict[str, Any]:
    """Return hierarchy view shape used by phase 3 export."""

    return {
        "id": item["id"],
        "name": item["name"],
        "quality_attribute": item.get("quality_attribute"),
        "parent_goal_id": item.get("parent_goal_id"),
    }


def _parent_child_mappings(elements: list[dict[str, Any]]) -> dict[str, list[str]]:
    mappings: dict[str, list[str]] = {}
    for element in elements:
        parent_id = element.get("parent_goal_id")
        if not parent_id:
            continue
        mappings.setdefault(parent_id, []).append(element["id"])
    return mappings


def _agent_quality_mapping_for_setting(setting: str) -> dict[str, str]:
    """Return agent-quality mapping for a given experimental setting."""

    if setting == SETTING_SINGLE_AGENT:
        return {"SingleAgent": "Integrated"}
    return DEFAULT_AGENT_QUALITY_ATTRIBUTES


def _negotiation_enabled(setting: str) -> bool:
    """Whether phase 2 negotiation should run for the setting."""

    return setting in {
        SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    }


def _verification_executed(setting: str) -> bool:
    """Whether phase 4 verification should run for the setting."""

    return setting == SETTING_NEGOTIATION_INTEGRATION_VERIFICATION


_NO_AF_LEAF_PREFIX_RE = re.compile(
    r"^The system shall ensure [a-z]+ \([^)]+\):\s*",
    re.IGNORECASE,
)


def _no_af_leaf_body_key(el: dict[str, Any]) -> str:
    """Normalize QUARE multi-agent leaf description for cross-agent deduplication."""

    desc = str(el.get("description", "")).strip()
    m = _NO_AF_LEAF_PREFIX_RE.match(desc)
    body = desc[m.end() :] if m else desc
    return re.sub(r"\s+", " ", body).strip().lower()


def _dedupe_no_af_phase3_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate leaf requirements across agents (same underlying clause body)."""

    if not elements:
        return []
    roots = [e for e in elements if int(e.get("hierarchy_level", 1) or 1) == 1]
    leaves = [e for e in elements if int(e.get("hierarchy_level", 1) or 1) >= 2]
    seen_body: set[str] = set()
    kept_leaves: list[dict[str, Any]] = []
    for leaf in sorted(leaves, key=lambda e: str(e.get("id", ""))):
        key = _no_af_leaf_body_key(leaf)
        if not key:
            kept_leaves.append(leaf)
            continue
        if key in seen_body:
            continue
        seen_body.add(key)
        kept_leaves.append(leaf)
    parents_with_children = {
        str(l.get("parent_goal_id")) for l in kept_leaves if l.get("parent_goal_id")
    }
    kept_roots = [
        r
        for r in sorted(roots, key=lambda e: str(e.get("id", "")))
        if str(r.get("id")) in parents_with_children
    ]
    merged = kept_roots + kept_leaves
    return sorted(merged, key=lambda e: str(e.get("id", "")))


def _phase3_source_elements(
    *,
    phase1: dict[str, list[dict[str, Any]]],
    phase2: dict[str, Any],
    setting: str,
) -> list[dict[str, Any]]:
    """Select phase3 source elements according to ablation setting semantics."""

    if setting == SETTING_MULTI_AGENT_WITHOUT_NEGOTIATION:
        flat = [item for elements in phase1.values() for item in elements]
        return _dedupe_no_af_phase3_elements(flat)

    if setting in {
        SETTING_MULTI_AGENT_WITH_NEGOTIATION,
        SETTING_NEGOTIATION_INTEGRATION_VERIFICATION,
    }:
        negotiated = _latest_backward_elements(phase2)
        if negotiated:
            if setting == SETTING_MULTI_AGENT_WITH_NEGOTIATION:
                return _compress_negotiated_elements_for_multi_setting(negotiated)
            return negotiated

    return [item for elements in phase1.values() for item in elements]


def _latest_backward_elements(phase2: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract latest backward-step elements from negotiation trace."""

    merged: dict[str, tuple[int, dict[str, Any]]] = {}
    negotiations = phase2.get("negotiations", {})
    if not isinstance(negotiations, dict):
        return []

    for pair_key in sorted(negotiations):
        negotiation = negotiations[pair_key]
        if not isinstance(negotiation, dict):
            continue
        steps = negotiation.get("steps", [])
        if not isinstance(steps, list):
            continue
        backward_steps = [step for step in steps if step.get("message_type") == "backward"]
        if backward_steps:
            source = backward_steps[-1].get("kaos_elements", [])
            step_id = _to_int(backward_steps[-1].get("step_id"), default=0)
        elif steps:
            source = steps[-1].get("kaos_elements", [])
            step_id = _to_int(steps[-1].get("step_id"), default=0)
        else:
            source = []
            step_id = 0
        if not isinstance(source, list):
            continue
        for element in source:
            if not isinstance(element, dict):
                continue
            element_id = str(element.get("id", "")).strip()
            if not element_id:
                continue
            candidate = (step_id, dict(element))
            existing = merged.get(element_id)
            if existing is None or candidate[0] >= existing[0]:
                merged[element_id] = candidate
    return [merged[element_id][1] for element_id in sorted(merged)]


def _compress_negotiated_elements_for_multi_setting(
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain a compact negotiated subset for the multi-agent-with-negotiation setting.

    The paper reports substantial post-negotiation filtering for this setting.
    We keep a deterministic representative subset: one integrated root plus a
    quality-diverse slice of negotiated leaf requirements.
    """

    if not elements:
        return []

    copied = [dict(item) for item in elements if isinstance(item, dict)]
    roots = [item for item in copied if int(item.get("hierarchy_level", 1)) == 1]
    leaves = [item for item in copied if int(item.get("hierarchy_level", 1)) >= 2]
    if not roots or not leaves:
        return copied

    primary_root = dict(sorted(roots, key=lambda item: str(item.get("id", "")))[0])
    primary_root["name"] = "Negotiated Integrated Goal"
    primary_root["description"] = _summarize_text(
        str(primary_root.get("description", "Negotiated integrated objective")),
        220,
    )
    primary_root["quality_attribute"] = "Integrated"
    primary_root["hierarchy_level"] = 1
    primary_root["parent_goal_id"] = None

    by_quality: dict[str, list[dict[str, Any]]] = {}
    for leaf in leaves:
        quality = str(leaf.get("quality_attribute", "Integrated")).strip() or "Integrated"
        by_quality.setdefault(quality, []).append(leaf)

    # Keep filtering substantial enough to match paper behavior while preserving
    # one representative item per quality lens.
    target_leaf_count = min(
        len(leaves),
        max(len(by_quality), 4, round(len(leaves) * 0.2)),
    )

    def _leaf_score(item: dict[str, Any]) -> tuple[int, int, str]:
        status = str(item.get("validation_status", "")).strip().lower()
        status_rank = 2 if status == "resolved" else 1 if status in {"accepted", "candidate_resolution"} else 0
        resolved_rank = 1 if str(item.get("conflict_resolved_by", "")).strip() else 0
        return (
            status_rank,
            resolved_rank,
            str(item.get("id", "")),
        )

    selected: list[dict[str, Any]] = []
    for quality in sorted(by_quality):
        candidates = sorted(by_quality[quality], key=_leaf_score, reverse=True)
        if candidates and len(selected) < target_leaf_count:
            selected.append(dict(candidates[0]))

    if len(selected) < target_leaf_count:
        selected_ids = {str(item.get("id", "")) for item in selected}
        remainder = [item for item in leaves if str(item.get("id", "")) not in selected_ids]
        for leaf in sorted(remainder, key=_leaf_score, reverse=True):
            selected.append(dict(leaf))
            if len(selected) >= target_leaf_count:
                break

    for leaf in selected:
        leaf["parent_goal_id"] = primary_root.get("id")

    return [primary_root, *selected]


def _detect_conflict_heuristic(
    *,
    focus_agent: str,
    reviewer_agent: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    requirement: str,
) -> bool:
    """Detect negotiation conflicts using deterministic rule heuristics."""

    statuses = {str(item.get("validation_status", "")).strip().lower() for item in focus_elements}
    if "candidate_resolution" in statuses or "resolved" in statuses:
        return False

    quality_pair = (_quality_axis_for_agent(focus_agent), _quality_axis_for_agent(reviewer_agent))
    known_conflict_pairs = {
        ("Safety", "Efficiency"),
        ("Efficiency", "Safety"),
        ("Sustainability", "Efficiency"),
        ("Responsibility", "Efficiency"),
    }
    trigger_terms = {"conflict", "tradeoff", "trade-off", "violate", "relax"}
    requirement_tokens = set(_tokens(requirement.lower()))

    lexical_trigger = bool(trigger_terms & requirement_tokens)
    if quality_pair in known_conflict_pairs and lexical_trigger:
        return True

    focus_text = " ".join(item.get("description", "") for item in focus_elements)
    reviewer_text = " ".join(item.get("description", "") for item in reviewer_elements)
    overlap = set(_tokens(focus_text.lower())) & set(_tokens(reviewer_text.lower()))

    has_positive_modal = any(token in focus_text.lower() for token in ("shall", "must", "should"))
    has_negative_modal = "not" in reviewer_text.lower() or "never" in reviewer_text.lower()
    return len(overlap) >= 3 and has_positive_modal and has_negative_modal


def _detect_conflict(
    *,
    focus_agent: str,
    reviewer_agent: str,
    focus_elements: list[dict[str, Any]],
    reviewer_elements: list[dict[str, Any]],
    requirement: str,
    paper_bert_conflict_prescreen: bool = False,
    paper_bert_similarity_tau: float = 0.85,
    precalc_bert_cosine: float | None = None,
) -> bool:
    """Conflict gate: legacy heuristics plus optional BERT cosine prescreen (paper Stage~1)."""

    heur = _detect_conflict_heuristic(
        focus_agent=focus_agent,
        reviewer_agent=reviewer_agent,
        focus_elements=focus_elements,
        reviewer_elements=reviewer_elements,
        requirement=requirement,
    )
    if not paper_bert_conflict_prescreen:
        return heur

    focus_text = " ".join(str(item.get("description", "")) for item in focus_elements).strip()
    reviewer_text = " ".join(str(item.get("description", "")) for item in reviewer_elements).strip()
    if len(focus_text) < 8 or len(reviewer_text) < 8:
        return heur

    sim = precalc_bert_cosine
    if sim is None:
        try:
            from openre_bench.verification.bert_pair_similarity import (
                pairwise_cosine_similarity_bert_uncased,
            )

            sim = pairwise_cosine_similarity_bert_uncased(focus_text, reviewer_text)
        except Exception:
            return heur

    bert_flag = float(sim) >= float(paper_bert_similarity_tau)
    return bool(heur or bert_flag)


def _quality_axis_for_agent(agent_name: str) -> str:
    """Normalize QUARE and MARE runtime role names onto the paper quality axes."""

    normalized = str(agent_name).strip()
    aliases = {
        "SafetyAgent": "Safety",
        "EfficiencyAgent": "Efficiency",
        "GreenAgent": "Sustainability",
        "TrustworthinessAgent": "Trustworthiness",
        "ResponsibilityAgent": "Responsibility",
        "Stakeholders": "Responsibility",
        "Collector": "Efficiency",
        "Modeler": "Trustworthiness",
        "Checker": "Safety",
        "Documenter": "Sustainability",
    }
    return aliases.get(normalized, normalized)


def _apply_negotiation_adjustments(
    *,
    elements: list[dict[str, Any]],
    reviewer_agent: str,
    conflict_detected: bool,
) -> list[dict[str, Any]]:
    """Adjust forward elements into negotiated backward elements."""

    updated = [dict(item) for item in elements]
    if not conflict_detected:
        return updated

    for element in updated:
        if int(element.get("hierarchy_level", 1)) != 2:
            continue
        element["description"] = (
            f"{element['description']} Negotiated trade-off accepted by {reviewer_agent}."
        )
        element["conflict_resolved_by"] = reviewer_agent
        element["validation_status"] = "resolved"
    return updated


def _backward_analysis_text(reviewer_agent: str, conflict_detected: bool) -> str:
    """Build backward analysis text with conflict context."""

    if conflict_detected:
        return f"{reviewer_agent} resolves detected conflict through negotiated refinement."
    return f"{reviewer_agent} accepts with no blocking conflicts."


def _backward_feedback(conflict_detected: bool) -> str:
    """Build backward feedback text."""

    if conflict_detected:
        return "Conflict detected and resolved through wording refinement."
    return "No blocking conflict found in this negotiation pair."


def _compute_topology_status(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute DAG topology validity from GSN elements."""

    ids = {item["id"] for item in elements}
    children: dict[str, list[str]] = {item["id"]: [] for item in elements}
    orphan_elements: list[str] = []

    for item in elements:
        parent_id = item.get("parent_goal_id")
        if not parent_id:
            if item.get("gsn_type") != "Goal":
                orphan_elements.append(item["id"])
            continue
        if parent_id not in ids:
            orphan_elements.append(item["id"])
            continue
        children[parent_id].append(item["id"])

    cycles: list[list[str]] = []
    visited: set[str] = set()
    active: set[str] = set()

    def walk(node: str, path: list[str]) -> None:
        if node in active:
            start = path.index(node) if node in path else 0
            cycles.append(path[start:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        active.add(node)
        for child in children.get(node, []):
            walk(child, path + [child])
        active.remove(node)

    for node_id in children:
        walk(node_id, [node_id])

    invalid_leaves = [
        item["id"]
        for item in elements
        if not children.get(item["id"], []) and item.get("gsn_type") == "Goal"
    ]

    is_valid = not orphan_elements and not cycles and not invalid_leaves
    return {
        "status": "Valid" if is_valid else "Invalid",
        "is_valid": is_valid,
        "is_dag": not cycles,
        "orphan_count": len(orphan_elements),
        "cycle_count": len(cycles),
        "invalid_leaf_count": len(invalid_leaves),
        "orphan_elements": orphan_elements,
        "cycles": cycles,
        "invalid_leaves": invalid_leaves,
    }


def _logical_consistency(gsn_elements: list[dict[str, Any]]) -> dict[str, Any]:
    """ISO29148-style contradiction checks for logical consistency."""

    constraints: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []

    for element in gsn_elements:
        description = str(element.get("description", ""))
        element_id = str(element.get("id", ""))
        for var, op, value in re.findall(r"([a-zA-Z_]+)\s*(≤|≥|<|>|=)\s*(\d+\.?\d*)", description):
            constraints.append(
                {
                    "element_id": element_id,
                    "variable": var,
                    "operator": op,
                    "value": float(value),
                }
            )

        forbidden = re.search(r"\b(must|shall|should)\s+not\s+(\w+)", description, re.IGNORECASE)
        if forbidden:
            constraints.append(
                {
                    "element_id": element_id,
                    "variable": forbidden.group(2).lower(),
                    "operator": "forbidden",
                    "value": None,
                }
            )

    for index, left in enumerate(constraints):
        for right in constraints[index + 1 :]:
            if left["variable"].lower() != right["variable"].lower():
                continue
            if left["operator"] == "forbidden" and right["operator"] != "forbidden":
                contradictions.append(
                    {
                        "type": "forbidden_conflict",
                        "variable": left["variable"],
                        "left": left,
                        "right": right,
                    }
                )
                continue
            if right["operator"] == "forbidden" and left["operator"] != "forbidden":
                contradictions.append(
                    {
                        "type": "forbidden_conflict",
                        "variable": left["variable"],
                        "left": left,
                        "right": right,
                    }
                )
                continue

            if left["value"] is None or right["value"] is None:
                continue
            if left["operator"] in {"≤", "<", "="} and right["operator"] in {"≥", ">", "="}:
                if left["operator"] == "≤" and right["operator"] == "≥" and left["value"] < right["value"]:
                    contradictions.append(
                        {
                            "type": "numerical_contradiction",
                            "variable": left["variable"],
                            "left": left,
                            "right": right,
                        }
                    )
                if left["operator"] == "=" and right["operator"] == "=" and left["value"] != right["value"]:
                    contradictions.append(
                        {
                            "type": "numerical_contradiction",
                            "variable": left["variable"],
                            "left": left,
                            "right": right,
                        }
                    )

    total_requirements = max(1, len(gsn_elements))
    contradiction_rate = len(contradictions) / total_requirements
    if contradiction_rate == 0:
        score = 5
    elif contradiction_rate <= 0.05:
        score = 4
    elif contradiction_rate <= 0.10:
        score = 3
    elif contradiction_rate <= 0.20:
        score = 2
    else:
        score = 1

    return {
        "score": score,
        "normalized_score": round(score / 5.0, 6),
        "contradiction_rate": round(contradiction_rate, 6),
        "contradictions": contradictions,
        "recommendations": [
            "Resolve contradictory numeric bounds" if contradictions else "No contradiction detected"
        ],
    }


def _terminology_consistency(gsn_elements: list[dict[str, Any]]) -> dict[str, Any]:
    """ISO29148-style terminology consistency checks."""

    synonym_groups = {
        "user": ["user", "client", "customer", "member"],
        "system": ["system", "application", "platform", "service"],
        "account": ["account", "checking-account", "saving-account"],
        "transaction": ["transaction", "operation", "action"],
        "data": ["data", "information", "record"],
        "access": ["access", "gain access", "obtain access"],
        "verify": ["verify", "validate", "check", "authenticate"],
    }

    terminology_map: dict[str, list[tuple[str, str]]] = {}
    for element in gsn_elements:
        description = str(element.get("description", "")).lower()
        element_id = str(element.get("id", ""))
        for concept, synonyms in synonym_groups.items():
            for term in synonyms:
                if term in description:
                    terminology_map.setdefault(concept, []).append((term, element_id))

    conflicts: list[dict[str, Any]] = []
    for concept, usage in terminology_map.items():
        terms = {term for term, _ in usage}
        if len(terms) > 1:
            conflicts.append(
                {
                    "concept": concept,
                    "terms_used": sorted(terms),
                    "requirements": sorted({element_id for _, element_id in usage}),
                }
            )

    total_concepts = len(terminology_map)
    consistent = total_concepts - len(conflicts)
    consistency_ratio = consistent / total_concepts if total_concepts else 1.0
    if consistency_ratio >= 0.95:
        score = 5
    elif consistency_ratio >= 0.85:
        score = 4
    elif consistency_ratio >= 0.70:
        score = 3
    elif consistency_ratio >= 0.50:
        score = 2
    else:
        score = 1

    return {
        "score": score,
        "consistency_ratio": round(consistency_ratio, 6),
        "conflicts": conflicts,
        "total_concepts": total_concepts,
        "consistent_concepts": consistent,
    }


def _compliance_coverage(gsn_elements: list[dict[str, Any]], requirement: str) -> dict[str, Any]:
    """Compute compliance coverage as satisfied clauses / applicable clauses."""

    clauses = _extract_requirement_fragments(requirement)
    if not clauses:
        clauses = [requirement.strip()] if requirement.strip() else []

    requirement_texts = [
        f"{item.get('name', '')}. {item.get('description', '')}".strip()
        for item in gsn_elements
    ]

    satisfied = 0
    clause_results: list[dict[str, Any]] = []
    for clause in clauses:
        best = 0.0
        for generated in requirement_texts:
            score = _text_overlap_score(clause, generated)
            if score > best:
                best = score
        matched = best >= 0.45
        if matched:
            satisfied += 1
        clause_results.append(
            {
                "clause": _summarize_text(clause, 120),
                "best_overlap": round(best, 6),
                "matched": matched,
            }
        )

    total = len(clauses)
    ratio = satisfied / total if total else 0.0
    return {
        "satisfied_applicable_clauses": satisfied,
        "total_applicable_clauses": total,
        "coverage_ratio": round(ratio, 6),
        "clauses": clause_results,
    }


def _fact_checking(gsn_elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Basic source-evidence checks for generated elements."""

    flagged: list[dict[str, Any]] = []
    for item in gsn_elements:
        properties = item.get("properties", {})
        if properties.get("source"):
            continue
        flagged.append(
            {
                "element_id": item.get("id"),
                "issue": "missing_source",
            }
        )
    return {
        "hallucination_reports": flagged,
        "flagged_elements": [entry["element_id"] for entry in flagged],
        "total_checked": len(gsn_elements),
    }


def _build_deterministic_validation(
    *,
    topology_status: dict[str, Any],
    logical: dict[str, Any],
    verification_executed: bool,
) -> dict[str, Any]:
    """Construct deterministic validation payload used by phase4 artifact."""

    violations: list[dict[str, Any]] = []
    for contradiction in logical["contradictions"]:
        violations.append(
            {
                "rule_id": "logical_consistency",
                "rule_name": contradiction.get("type", "logical_consistency"),
                "element_id": contradiction.get("left", {}).get("element_id", "N/A"),
                "element_name": "N/A",
                "severity": "medium",
                "message": "Detected contradiction between requirement constraints",
            }
        )

    if topology_status.get("cycle_count", 0) > 0:
        violations.append(
            {
                "rule_id": "topology_cycle",
                "rule_name": "cycle_detection",
                "element_id": "N/A",
                "element_name": "N/A",
                "severity": "high",
                "message": "Detected cycle in integrated topology",
            }
        )

    passed_rules = ["topology_validity", "traceability"]
    if not violations:
        passed_rules.append("logical_consistency")

    if not verification_executed:
        return {
            "is_valid": False,
            "violations": [],
            "passed_rules": [],
            "status": "not_executed",
        }

    return {
        "is_valid": bool(topology_status.get("is_valid", False) and not violations),
        "violations": violations,
        "passed_rules": passed_rules,
    }


def _text_overlap_score(left: str, right: str) -> float:
    """Compute token overlap F1 for two text strings."""

    left_tokens = set(_tokens(left.lower()))
    right_tokens = set(_tokens(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tokens(text: str) -> list[str]:
    """Tokenize to alphanumeric lowercase words."""

    return re.findall(r"[a-z0-9_]+", text)


def _to_int(value: Any, default: int = 0) -> int:
    """Parse integer-like values safely for optional artifact accounting."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    """Parse float-like values safely for optional artifact accounting."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summarize_text(text: str, limit: int) -> str:
    """Trim text to a bounded length preserving readability."""

    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    budget = max(0, limit - 3)
    cut = compact[:budget].rstrip()
    if " " in cut and len(cut) == budget:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return cut + "..."


def _coerce_non_empty_text(value: Any, *, fallback: str) -> str:
    """Return normalized text when available; otherwise use fallback."""

    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return fallback


def _parse_quare_llm_payload(raw_response: str) -> tuple[dict[str, Any], bool]:
    """Parse one JSON object from LLM output with fence/substring recovery."""

    text = raw_response.strip()
    if not text:
        raise ValueError("empty response")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload, False

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for candidate in fenced:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, True

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response did not contain a JSON object")

    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid recovered JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("recovered payload is not an object")
    return payload, True
