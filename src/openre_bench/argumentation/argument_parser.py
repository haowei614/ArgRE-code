"""Argument extraction and turn classification from phase-2 negotiations."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from openre_bench.llm import LLMContract as Phase2LLMClient
from openre_bench.llm import chat_with_optional_seed_with_backoff_async

TURN_TYPES = {"proposal", "critique", "refinement"}


@dataclass(frozen=True)
class Argument:
    """Normalized argument item extracted from one phase-2 negotiation turn."""

    argument_id: str
    pair_key: str
    negotiation_id: str
    step_id: int
    round_number: int
    focus_agent: str
    reviewer_agent: str
    message_type: str
    analysis_text: str
    feedback: str
    turn_type: str
    turn_type_confidence: float
    llm_classified: bool
    quality_attribute: str
    kaos_elements: list[dict[str, Any]]
    source_timestamp: str


@dataclass(frozen=True)
class ArgumentParseMeta:
    """Execution metadata for argument extraction/classification."""

    total_turns: int
    llm_turns: int
    llm_fallback_turns: int
    llm_retry_count: int


@dataclass(frozen=True)
class _TurnJob:
    pair_key: str
    negotiation_id: str
    step: dict[str, Any]


def parse_phase2_arguments(
    *,
    negotiation_map: dict[str, Any],
    llm_client: Phase2LLMClient | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_max_concurrency: int = 10,
    llm_backoff_cap_seconds: float = 60.0,
    llm_backoff_base_seconds: float = 1.0,
    llm_retry_limit: int = 1,
) -> tuple[list[Argument], ArgumentParseMeta]:
    """Extract arguments from phase-2 turns and classify turn type in parallel."""

    return asyncio.run(
        _parse_phase2_arguments_async(
            negotiation_map=negotiation_map,
            llm_client=llm_client,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_max_concurrency=llm_max_concurrency,
            llm_backoff_cap_seconds=llm_backoff_cap_seconds,
            llm_backoff_base_seconds=llm_backoff_base_seconds,
            llm_retry_limit=llm_retry_limit,
        )
    )


async def _parse_phase2_arguments_async(
    *,
    negotiation_map: dict[str, Any],
    llm_client: Phase2LLMClient | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_max_concurrency: int,
    llm_backoff_cap_seconds: float,
    llm_backoff_base_seconds: float,
    llm_retry_limit: int,
) -> tuple[list[Argument], ArgumentParseMeta]:
    jobs = _collect_turn_jobs(negotiation_map)
    if not jobs:
        return [], ArgumentParseMeta(0, 0, 0, 0)

    semaphore = asyncio.Semaphore(max(1, int(llm_max_concurrency)))
    tasks = [
        _build_argument_for_turn(
            job=job,
            llm_client=llm_client,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_retry_limit=llm_retry_limit,
            semaphore=semaphore,
            llm_backoff_cap_seconds=llm_backoff_cap_seconds,
            llm_backoff_base_seconds=llm_backoff_base_seconds,
        )
        for job in jobs
    ]
    results = await asyncio.gather(*tasks)

    arguments = [item[0] for item in results]
    llm_turns = sum(1 for item in results if item[0].llm_classified)
    llm_retry_count = sum(item[1] for item in results)
    llm_fallback_turns = max(0, len(arguments) - llm_turns)
    return arguments, ArgumentParseMeta(
        total_turns=len(arguments),
        llm_turns=llm_turns,
        llm_fallback_turns=llm_fallback_turns,
        llm_retry_count=llm_retry_count,
    )


async def _build_argument_for_turn(
    *,
    job: _TurnJob,
    llm_client: Phase2LLMClient | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_retry_limit: int,
    semaphore: asyncio.Semaphore,
    llm_backoff_cap_seconds: float,
    llm_backoff_base_seconds: float,
) -> tuple[Argument, int]:
    step = job.step
    fallback_turn_type = _heuristic_turn_type(step)
    retry_count = 0
    llm_classified = False
    turn_type = fallback_turn_type
    confidence = 0.5

    if llm_client is not None:
        response_tuple, retries, _ = await chat_with_optional_seed_with_backoff_async(
            llm_client=llm_client,
            messages=_build_turn_classification_messages(
                llm_model=llm_model,
                pair_key=job.pair_key,
                negotiation_id=job.negotiation_id,
                step=step,
            ),
            temperature=max(0.0, min(float(llm_temperature), 1.0)),
            max_tokens=max(128, min(int(llm_max_tokens), 600)),
            seed=llm_seed,
            retry_limit=max(0, int(llm_retry_limit)),
            semaphore=semaphore,
            backoff_cap_seconds=llm_backoff_cap_seconds,
            backoff_base_seconds=llm_backoff_base_seconds,
        )
        retry_count = retries
        if response_tuple is not None:
            parsed = _parse_turn_type_response(response_tuple[0])
            if parsed is not None:
                turn_type, confidence = parsed
                llm_classified = True

    argument = Argument(
        argument_id=f"{job.pair_key}-s{_to_int(step.get('step_id', 0))}",
        pair_key=job.pair_key,
        negotiation_id=job.negotiation_id,
        step_id=_to_int(step.get("step_id", 0)),
        round_number=_to_int(step.get("round_number", 1), default=1),
        focus_agent=str(step.get("focus_agent", "")).strip(),
        reviewer_agent=str(step.get("reviewer_agent", "")).strip(),
        message_type=str(step.get("message_type", "")).strip(),
        analysis_text=_coerce_text(step.get("analysis_text") or step.get("analysis")),
        feedback=_coerce_text(step.get("feedback")),
        turn_type=turn_type,
        turn_type_confidence=confidence,
        llm_classified=llm_classified,
        quality_attribute=_infer_quality_attribute(step),
        kaos_elements=_coerce_elements(step.get("kaos_elements")),
        source_timestamp=str(step.get("timestamp", "")).strip(),
    )
    return argument, retry_count


def _collect_turn_jobs(negotiation_map: dict[str, Any]) -> list[_TurnJob]:
    jobs: list[_TurnJob] = []
    for pair_key in sorted(negotiation_map):
        negotiation = _to_dict(negotiation_map[pair_key])
        if not negotiation:
            continue
        negotiation_id = str(negotiation.get("negotiation_id", pair_key)).strip()
        steps_raw = negotiation.get("steps", [])
        if not isinstance(steps_raw, list):
            continue
        for step_raw in steps_raw:
            step = _to_dict(step_raw)
            if not step:
                continue
            jobs.append(
                _TurnJob(
                    pair_key=str(pair_key),
                    negotiation_id=negotiation_id,
                    step=step,
                )
            )
    return jobs


def _build_turn_classification_messages(
    *,
    llm_model: str,
    pair_key: str,
    negotiation_id: str,
    step: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "task": "Classify negotiation turn type",
        "model": llm_model,
        "pair_key": pair_key,
        "negotiation_id": negotiation_id,
        "step": {
            "step_id": step.get("step_id"),
            "round_number": step.get("round_number"),
            "focus_agent": step.get("focus_agent"),
            "reviewer_agent": step.get("reviewer_agent"),
            "message_type": step.get("message_type"),
            "analysis_text": _coerce_text(step.get("analysis_text") or step.get("analysis")),
            "feedback": _coerce_text(step.get("feedback")),
            "resolution_state": step.get("resolution_state"),
            "requires_refinement": step.get("requires_refinement"),
            "conflict_detected": step.get("conflict_detected"),
        },
        "labels": ["proposal", "critique", "refinement"],
        "output_schema": {
            "turn_type": "proposal|critique|refinement",
            "confidence": "float[0,1]",
        },
    }
    return [
        {
            "role": "system",
            "content": "Return only one JSON object with keys turn_type and confidence.",
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _parse_turn_type_response(raw_response: str) -> tuple[str, float] | None:
    payload = _try_parse_json_object(raw_response)
    if payload is None:
        return None
    turn_type = str(payload.get("turn_type", "")).strip().lower()
    if turn_type not in TURN_TYPES:
        return None
    confidence = _to_float(payload.get("confidence", 0.75), default=0.75)
    confidence = max(0.0, min(confidence, 1.0))
    return turn_type, confidence


def _heuristic_turn_type(step: dict[str, Any]) -> str:
    message_type = str(step.get("message_type", "")).strip().lower()
    if message_type == "forward":
        return "proposal"
    if bool(step.get("requires_refinement", False)):
        return "refinement"
    return "critique"


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text).strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {}


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value is not None else ""


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_elements(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            output.append(dict(item))
    return output


def _infer_quality_attribute(step: dict[str, Any]) -> str:
    elements = _coerce_elements(step.get("kaos_elements"))
    for element in elements:
        quality = str(element.get("quality_attribute", "")).strip()
        if quality:
            return quality
    return "Integrated"
