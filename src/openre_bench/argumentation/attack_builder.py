"""Build attack relations from extracted arguments."""

from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import dataclass
from typing import Any

from openre_bench.argumentation.argument_parser import Argument
from openre_bench.llm import LLMContract as Phase2LLMClient
from openre_bench.llm import chat_with_optional_seed_with_backoff_async

_NEGATION_TOKENS = (
    "not",
    "cannot",
    "can't",
    "incompatible",
    "violate",
    "conflict",
    "contradict",
)
_TRADEOFF_TOKENS = (
    "tradeoff",
    "trade-off",
    "balance",
    "compromise",
    "priority over",
)


@dataclass(frozen=True)
class AttackRelation:
    """Directed attack edge in an argumentation framework."""

    attacker_id: str
    target_id: str
    confidence: float
    source: str  # "rule:<pattern>" | "llm"
    reason: str


@dataclass(frozen=True)
class AttackBuildMeta:
    """Execution statistics for attack relation construction."""

    rule_based_attacks: int
    rule_pattern_counts: dict[str, int]
    llm_candidate_pairs: int
    llm_detected_attacks: int
    llm_failures: int
    llm_retry_count: int


@dataclass(frozen=True)
class _PairJob:
    left: Argument
    right: Argument


def build_attack_relations(
    *,
    arguments: list[Argument],
    llm_client: Phase2LLMClient | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_max_concurrency: int = 10,
    llm_backoff_cap_seconds: float = 60.0,
    llm_backoff_base_seconds: float = 1.0,
    llm_retry_limit: int = 1,
    confidence_threshold: float = 0.7,
    attack_detection_mode: str = "full",
    llm_confidence_floor: float = 0.85,
) -> tuple[list[AttackRelation], AttackBuildMeta]:
    """Build attack relations with rule-first pruning then parallel LLM checks."""

    mode = str(attack_detection_mode).strip().lower() or "full"
    if mode not in {"full", "rule_only", "llm_only"}:
        raise ValueError(f"Unsupported attack_detection_mode: {attack_detection_mode}")

    sorted_args = sorted(arguments, key=lambda arg: (arg.pair_key, arg.step_id, arg.argument_id))
    pattern_counts = {
        "critique_direct": 0,
        "refinement_prev_version": 0,
        "refinement_counter_critique": 0,
    }
    attacks_by_key: dict[tuple[str, str], AttackRelation] = {}
    by_pair: dict[str, list[Argument]] = {}
    for argument in sorted_args:
        by_pair.setdefault(argument.pair_key, []).append(argument)

    if mode in {"full", "rule_only"}:
        # Rule-based attacks are resolved in-pair with one-to-one targeting.
        for pair_key in sorted(by_pair):
            pair_arguments = sorted(by_pair[pair_key], key=lambda item: (item.step_id, item.argument_id))
            for relation, pattern_name in _rule_based_attacks_for_pair(pair_arguments):
                pattern_counts[pattern_name] += 1
                _upsert_attack(attacks_by_key, relation)

    llm_jobs: list[_PairJob] = []
    if mode in {"full", "llm_only"}:
        survivors = _surviving_arguments_per_pair(by_pair=by_pair, attacks_by_key=attacks_by_key)
        llm_jobs = [
            _PairJob(left=a, right=b)
            for a, b in itertools.combinations(sorted(survivors, key=lambda x: x.argument_id), 2)
            if a.pair_key != b.pair_key
        ]

    effective_threshold = max(float(llm_confidence_floor), float(confidence_threshold))
    llm_relations: list[AttackRelation] = []
    llm_detected = 0
    llm_failures = 0
    llm_retry_count = 0
    if llm_jobs:
        llm_relations, llm_detected, llm_failures, llm_retry_count = asyncio.run(
            _build_llm_attacks_async(
                jobs=llm_jobs,
                llm_client=llm_client,
                llm_model=llm_model,
                llm_temperature=llm_temperature,
                llm_max_tokens=llm_max_tokens,
                llm_seed=llm_seed,
                llm_retry_limit=llm_retry_limit,
                llm_max_concurrency=llm_max_concurrency,
                llm_backoff_cap_seconds=llm_backoff_cap_seconds,
                llm_backoff_base_seconds=llm_backoff_base_seconds,
                confidence_threshold=effective_threshold,
            )
        )
    for relation in llm_relations:
        _upsert_attack(attacks_by_key, relation)

    attacks = sorted(
        attacks_by_key.values(),
        key=lambda item: (item.attacker_id, item.target_id),
    )
    rule_based_attacks = sum(pattern_counts.values())
    meta = AttackBuildMeta(
        rule_based_attacks=rule_based_attacks,
        rule_pattern_counts=pattern_counts,
        llm_candidate_pairs=len(llm_jobs),
        llm_detected_attacks=llm_detected,
        llm_failures=llm_failures,
        llm_retry_count=llm_retry_count,
    )
    return attacks, meta


def _rule_based_attacks_for_pair(arguments: list[Argument]) -> list[tuple[AttackRelation, str]]:
    attacks: list[tuple[AttackRelation, str]] = []
    for index, current in enumerate(arguments):
        if current.turn_type == "critique":
            target = _resolve_direct_target(
                current=current,
                arguments=arguments,
                index=index,
                candidate_turn_types={"proposal", "refinement"},
            )
            if target is not None:
                attacks.append(
                    (
                        AttackRelation(
                            attacker_id=current.argument_id,
                            target_id=target.argument_id,
                            confidence=0.95,
                            source="rule:critique_direct",
                            reason="Critique directly challenges referenced or adjacent prior proposal/refinement.",
                        ),
                        "critique_direct",
                    )
                )
            continue

        if current.turn_type == "refinement":
            prev_version = _closest_previous(
                arguments=arguments,
                index=index,
                predicate=lambda item: item.turn_type in {"proposal", "refinement"}
                and item.focus_agent == current.focus_agent,
            )
            if prev_version is not None:
                attacks.append(
                    (
                        AttackRelation(
                            attacker_id=current.argument_id,
                            target_id=prev_version.argument_id,
                            confidence=0.92,
                            source="rule:refinement_prev_version",
                            reason="Refinement supersedes previous version from same agent.",
                        ),
                        "refinement_prev_version",
                    )
                )

            responded_critique = _resolve_direct_target(
                current=current,
                arguments=arguments,
                index=index,
                candidate_turn_types={"critique"},
            )
            if responded_critique is not None:
                attacks.append(
                    (
                        AttackRelation(
                            attacker_id=current.argument_id,
                            target_id=responded_critique.argument_id,
                            confidence=0.9,
                            source="rule:refinement_counter_critique",
                            reason="Refinement counter-attacks the directly addressed critique.",
                        ),
                        "refinement_counter_critique",
                    )
                )
    return attacks


def _surviving_arguments_per_pair(
    *,
    by_pair: dict[str, list[Argument]],
    attacks_by_key: dict[tuple[str, str], AttackRelation],
) -> list[Argument]:
    attacked_ids = {target for _, target in attacks_by_key}
    survivors: list[Argument] = []
    for pair_key in sorted(by_pair):
        candidates = [
            item
            for item in sorted(by_pair[pair_key], key=lambda x: (x.step_id, x.argument_id))
            if item.turn_type in {"proposal", "refinement"}
        ]
        if not candidates:
            continue
        alive = [item for item in candidates if item.argument_id not in attacked_ids]
        if alive:
            survivors.extend(alive)
            continue
        survivors.append(candidates[-1])
    return survivors


async def _build_llm_attacks_async(
    *,
    jobs: list[_PairJob],
    llm_client: Phase2LLMClient | None,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_retry_limit: int,
    llm_max_concurrency: int,
    llm_backoff_cap_seconds: float,
    llm_backoff_base_seconds: float,
    confidence_threshold: float,
) -> tuple[list[AttackRelation], int, int, int]:
    if not jobs or llm_client is None:
        return [], 0, 0, 0

    semaphore = asyncio.Semaphore(max(1, int(llm_max_concurrency)))
    tasks = [
        _classify_pair_attack(
            pair=pair,
            llm_client=llm_client,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_retry_limit=llm_retry_limit,
            semaphore=semaphore,
            llm_backoff_cap_seconds=llm_backoff_cap_seconds,
            llm_backoff_base_seconds=llm_backoff_base_seconds,
            confidence_threshold=confidence_threshold,
        )
        for pair in jobs
    ]
    results = await asyncio.gather(*tasks)

    relations = [item[0] for item in results if item[0] is not None]
    llm_detected = sum(1 for item in results if item[0] is not None)
    llm_failures = sum(1 for _, failed, _ in results if failed)
    llm_retry_count = sum(item[2] for item in results)
    return relations, llm_detected, llm_failures, llm_retry_count


async def _classify_pair_attack(
    *,
    pair: _PairJob,
    llm_client: Phase2LLMClient,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_seed: int,
    llm_retry_limit: int,
    semaphore: asyncio.Semaphore,
    llm_backoff_cap_seconds: float,
    llm_backoff_base_seconds: float,
    confidence_threshold: float,
) -> tuple[AttackRelation | None, bool, int]:
    response_tuple, retries, _ = await chat_with_optional_seed_with_backoff_async(
        llm_client=llm_client,
        messages=_build_llm_pair_messages(
            llm_model=llm_model,
            left=pair.left,
            right=pair.right,
        ),
        temperature=max(0.0, min(float(llm_temperature), 1.0)),
        max_tokens=max(128, min(int(llm_max_tokens), 500)),
        seed=llm_seed,
        retry_limit=max(0, int(llm_retry_limit)),
        semaphore=semaphore,
        backoff_cap_seconds=llm_backoff_cap_seconds,
        backoff_base_seconds=llm_backoff_base_seconds,
    )
    if response_tuple is None:
        return None, True, retries

    parsed = _parse_llm_conflict_response(response_tuple[0])
    if parsed is None:
        # Conservative fallback: malformed response => no-attack.
        return None, True, retries

    conflict, confidence, reason = parsed
    if not conflict or confidence < float(confidence_threshold):
        return None, False, retries
    attacker, target = _orient_pair(pair.left, pair.right)
    relation = AttackRelation(
        attacker_id=attacker.argument_id,
        target_id=target.argument_id,
        confidence=confidence,
        source="llm",
        reason=reason,
    )
    return relation, False, retries


def _build_llm_pair_messages(
    *,
    llm_model: str,
    left: Argument,
    right: Argument,
) -> list[dict[str, str]]:
    payload = {
        "task": "Determine whether two arguments are in semantic conflict",
        "model": llm_model,
        "argument_a": {
            "id": left.argument_id,
            "turn_type": left.turn_type,
            "text": _argument_text(left),
        },
        "argument_b": {
            "id": right.argument_id,
            "turn_type": right.turn_type,
            "text": _argument_text(right),
        },
        "output_schema": {
            "conflict": "boolean",
            "confidence": "float in [0,1]",
            "reason": "brief string",
        },
        "note": (
            "Two requirements with different quality concerns are NOT conflicts by default. "
            "Conflict exists ONLY when implementing one makes the other impossible or significantly harder."
        ),
        "required_output_json": {"conflict": True, "confidence": 0.0, "reason": "..."},
    }
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object with keys conflict, confidence, reason. "
                "No markdown fences."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _parse_llm_conflict_response(raw_response: str) -> tuple[bool, float, str] | None:
    payload = _try_parse_json_object(raw_response)
    if payload is None:
        return None
    conflict_raw = payload.get("conflict")
    if isinstance(conflict_raw, bool):
        conflict = conflict_raw
    elif isinstance(conflict_raw, (int, float)):
        conflict = bool(conflict_raw)
    elif isinstance(conflict_raw, str):
        lowered = conflict_raw.strip().lower()
        if lowered in {"true", "1", "yes"}:
            conflict = True
        elif lowered in {"false", "0", "no"}:
            conflict = False
        else:
            return None
    else:
        return None
    confidence = _to_float(payload.get("confidence", 0.0), default=0.0)
    confidence = max(0.0, min(confidence, 1.0))
    reason = str(payload.get("reason", "")).strip() or "semantic conflict signal"
    return conflict, confidence, reason


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
    return parsed if isinstance(parsed, dict) else None


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _argument_text(argument: Argument) -> str:
    return " ".join(
        part for part in (argument.analysis_text.strip(), argument.feedback.strip()) if part
    )


def _resolve_direct_target(
    *,
    current: Argument,
    arguments: list[Argument],
    index: int,
    candidate_turn_types: set[str],
) -> Argument | None:
    referenced_id = _extract_referenced_argument_id(current)
    if referenced_id:
        for item in arguments:
            if item.argument_id == referenced_id and item.turn_type in candidate_turn_types:
                return item
    immediate_prev = _closest_previous(
        arguments=arguments,
        index=index,
        predicate=lambda item: item.turn_type in candidate_turn_types and item.step_id == current.step_id - 1,
    )
    if immediate_prev is not None:
        return immediate_prev
    return _closest_previous(
        arguments=arguments,
        index=index,
        predicate=lambda item: item.turn_type in candidate_turn_types,
    )


def _extract_referenced_argument_id(argument: Argument) -> str:
    text = f"{argument.analysis_text} {argument.feedback}".strip()
    if not text:
        return ""
    # Match local argument id style like SafetyAgent_EfficiencyAgent-s2.
    parts = text.replace(",", " ").replace(";", " ").split()
    for token in parts:
        normalized = token.strip().strip("()[]{}.:")
        if "-s" in normalized and "_" in normalized:
            return normalized
    return ""


def _closest_previous(
    *,
    arguments: list[Argument],
    index: int,
    predicate: Any,
) -> Argument | None:
    for idx in range(index - 1, -1, -1):
        candidate = arguments[idx]
        if predicate(candidate):
            return candidate
    return None


def _orient_pair(left: Argument, right: Argument) -> tuple[Argument, Argument]:
    # Prefer explicit critique/refinement turn as attacker.
    if left.turn_type in {"critique", "refinement"} and right.turn_type == "proposal":
        return left, right
    if right.turn_type in {"critique", "refinement"} and left.turn_type == "proposal":
        return right, left
    # Otherwise, later step attacks earlier step.
    if left.step_id >= right.step_id:
        return left, right
    return right, left


def _upsert_attack(
    attacks_by_key: dict[tuple[str, str], AttackRelation],
    relation: AttackRelation,
) -> None:
    key = (relation.attacker_id, relation.target_id)
    existing = attacks_by_key.get(key)
    if existing is None or relation.confidence > existing.confidence:
        attacks_by_key[key] = relation


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
