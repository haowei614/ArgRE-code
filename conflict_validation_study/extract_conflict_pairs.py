#!/usr/bin/env python3
"""Extract attack-pair samples for the human conflict-validation study."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = Path(__file__).resolve().parent
RUN_RECORD_ROOT = REPO_ROOT / "report" / "blind_requirement_quality_formal" / "runs"
RUN_RECORD_GLOB = "argre_af_preferred/*/run_record.json"
SAMPLE_SEED = 20260614
TOTAL_ATTACK_SAMPLES = 25
TOTAL_NEGATIVE_CONTROLS = 5
PATTERN_MAP = {
    "rule:refinement_prev_version": "P1",
    "refinement_prev_version": "P1",
    "rule:critique_direct": "P2",
    "critique_direct": "P2",
    "rule:refinement_counter_critique": "P3",
    "refinement_counter_critique": "P3",
}
CASE_ORDER = ("AD", "ATM", "Bookkeeping")


@dataclass
class ArgumentRecord:
    """Flattened argument turn used by the annotation materials."""

    id: str
    text: str
    agent: str
    round: int | None
    quality_dimension: str
    kaos_level: str
    case_study: str
    seed: int
    source_file: str
    negotiation_id: str
    pair_key: str
    step_id: int | None
    turn_type: str
    context_snippet: str = ""


@dataclass
class AttackCandidate:
    """Attack edge with resolved argument metadata."""

    source: ArgumentRecord
    target: ArgumentRecord
    pattern: str
    raw_pattern: str
    case_study: str
    source_seed: int
    source_file: str
    confidence: float | None
    reason: str
    grounded_source_accepted: bool
    grounded_target_accepted: bool
    seeds: set[int] = field(default_factory=set)
    source_files: set[str] = field(default_factory=set)


def main() -> None:
    rng = random.Random(SAMPLE_SEED)
    run_records = _discover_run_records()
    if len(run_records) != 9:
        print(f"Warning: expected 9 run records, found {len(run_records)}.")

    attacks: list[AttackCandidate] = []
    negative_pool: list[tuple[ArgumentRecord, ArgumentRecord, str, int, str]] = []
    stats_by_case_seed: Counter[tuple[str, int]] = Counter()
    stats_by_pattern: Counter[str] = Counter()

    for run_record_path in run_records:
        run_record = _read_json(run_record_path)
        case_study = _case_name(run_record)
        seed = int(run_record.get("seed", _seed_from_text(str(run_record_path))))
        graph_path = _artifact_path(run_record, "argumentation_graph.json")
        graph = _read_json(graph_path)
        argument_index = {
            item.id: item
            for item in (
                _argument_record(raw, case_study=case_study, seed=seed, source_file=graph_path)
                for raw in graph.get("arguments", [])
            )
        }
        grounded_extension = set(graph.get("grounded_extension", []))
        directed_attacks: set[tuple[str, str]] = set()

        for attack in graph.get("attacks", graph.get("attack_relations", [])):
            source_id = _first_present(attack, "attacker_id", "source", "source_id", "from")
            target_id = _first_present(attack, "target_id", "target", "target_id", "to")
            if not source_id or not target_id:
                continue
            directed_attacks.add((str(source_id), str(target_id)))
            source = argument_index.get(str(source_id))
            target = argument_index.get(str(target_id))
            if source is None or target is None:
                continue
            raw_pattern = _raw_pattern(attack)
            pattern = PATTERN_MAP.get(raw_pattern, PATTERN_MAP.get(raw_pattern.replace("rule:", ""), "P?"))
            candidate = AttackCandidate(
                source=source,
                target=target,
                pattern=pattern,
                raw_pattern=raw_pattern,
                case_study=case_study,
                source_seed=seed,
                source_file=_relpath(graph_path),
                confidence=_to_float(attack.get("confidence")),
                reason=str(attack.get("reason", "")).strip(),
                grounded_source_accepted=source.id in grounded_extension,
                grounded_target_accepted=target.id in grounded_extension,
                seeds={seed},
                source_files={_relpath(graph_path)},
            )
            attacks.append(candidate)
            stats_by_case_seed[(case_study, seed)] += 1
            stats_by_pattern[pattern] += 1

        negative_pool.extend(
            _negative_candidates(
                argument_index=argument_index,
                directed_attacks=directed_attacks,
                case_study=case_study,
                seed=seed,
                source_file=_relpath(graph_path),
            )
        )

    unique_attacks = _dedupe_attacks(attacks)
    attack_sample = _sample_attacks(unique_attacks, rng)
    negative_sample = _sample_negative_controls(negative_pool, attack_sample, rng)
    samples = _build_samples(attack_sample, negative_sample, rng)
    reveal_mapping = _build_reveal_mapping(samples)

    samples_path = STUDY_DIR / "conflict_validation_samples.json"
    reveal_path = STUDY_DIR / "private_reveal_mapping_conflict.json"
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reveal_path.write_text(
        json.dumps(reveal_mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _print_summary(
        run_records=run_records,
        raw_attacks=attacks,
        unique_attacks=unique_attacks,
        samples=samples,
        stats_by_case_seed=stats_by_case_seed,
        stats_by_pattern=stats_by_pattern,
        negative_pool=negative_pool,
        samples_path=samples_path,
        reveal_path=reveal_path,
    )


def _discover_run_records() -> list[Path]:
    return sorted(RUN_RECORD_ROOT.glob(RUN_RECORD_GLOB), key=lambda p: str(p))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _artifact_path(run_record: dict[str, Any], filename: str) -> Path:
    candidate = (
        run_record.get("artifact_paths", {}).get(filename)
        or run_record.get("notes", {}).get("argumentation_artifacts", {}).get(filename)
    )
    if not candidate:
        raise KeyError(f"Missing artifact path for {filename} in {run_record.get('run_id')}")
    path = Path(str(candidate))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _case_name(run_record: dict[str, Any]) -> str:
    raw = str(run_record.get("case_id", "")).strip()
    if raw.lower() == "bookkeeping":
        return "Bookkeeping"
    return raw.upper()


def _argument_record(
    raw: dict[str, Any], *, case_study: str, seed: int, source_file: Path
) -> ArgumentRecord:
    argument_id = str(_first_present(raw, "argument_id", "id"))
    elements = raw.get("kaos_elements") if isinstance(raw.get("kaos_elements"), list) else []
    agent = _speaker_agent(raw)
    return ArgumentRecord(
        id=argument_id,
        text=_argument_text(raw, elements),
        agent=agent,
        round=_to_int(_first_present(raw, "round_number", "round")),
        quality_dimension=str(_first_present(raw, "quality_attribute", "quality_dimension") or ""),
        kaos_level=_kaos_level(elements),
        case_study=case_study,
        seed=seed,
        source_file=_relpath(source_file),
        negotiation_id=str(raw.get("negotiation_id", "")),
        pair_key=str(raw.get("pair_key", "")),
        step_id=_to_int(raw.get("step_id")),
        turn_type=str(raw.get("turn_type", "")),
        context_snippet=_context_snippet(raw, elements),
    )


def _speaker_agent(raw: dict[str, Any]) -> str:
    message_type = str(raw.get("message_type", "")).lower()
    if message_type == "backward" and raw.get("reviewer_agent"):
        return str(raw["reviewer_agent"])
    return str(raw.get("focus_agent") or raw.get("agent") or raw.get("reviewer_agent") or "")


def _argument_text(raw: dict[str, Any], elements: list[Any]) -> str:
    parts: list[str] = []
    analysis = str(raw.get("analysis_text", "")).strip()
    feedback = str(raw.get("feedback", "")).strip()
    if analysis:
        parts.append(f"Negotiation turn: {analysis}")
    if feedback:
        parts.append(f"Feedback: {feedback}")
    requirement_lines = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        description = str(element.get("description", "")).strip()
        if not description:
            continue
        element_type = str(element.get("element_type", "")).strip()
        name = str(element.get("name", "")).strip()
        label = " - ".join(part for part in (element_type, name) if part)
        requirement_lines.append(f"{label}: {description}" if label else description)
    if requirement_lines:
        parts.append("KAOS content:\n" + "\n".join(f"- {line}" for line in requirement_lines))
    return "\n\n".join(parts).strip()


def _context_snippet(raw: dict[str, Any], elements: list[Any]) -> str:
    analysis = str(raw.get("analysis_text", "")).strip()
    feedback = str(raw.get("feedback", "")).strip()
    task_descriptions = [
        str(item.get("description", "")).strip()
        for item in elements
        if isinstance(item, dict) and str(item.get("element_type", "")).lower() == "task"
    ]
    snippets = [part for part in (analysis, feedback, " ".join(task_descriptions[:2])) if part]
    return _truncate(" ".join(snippets), 500)


def _kaos_level(elements: list[Any]) -> str:
    levels = sorted(
        {
            str(item.get("hierarchy_level"))
            for item in elements
            if isinstance(item, dict) and item.get("hierarchy_level") is not None
        }
    )
    if levels:
        return ",".join(f"L{level}" for level in levels)
    types = sorted(
        {
            str(item.get("element_type"))
            for item in elements
            if isinstance(item, dict) and item.get("element_type")
        }
    )
    return ",".join(types)


def _raw_pattern(attack: dict[str, Any]) -> str:
    source = str(_first_present(attack, "pattern", "type", "source") or "").strip()
    return source


def _negative_candidates(
    *,
    argument_index: dict[str, ArgumentRecord],
    directed_attacks: set[tuple[str, str]],
    case_study: str,
    seed: int,
    source_file: str,
) -> list[tuple[ArgumentRecord, ArgumentRecord, str, int, str]]:
    candidates: list[tuple[int, ArgumentRecord, ArgumentRecord, str, int, str]] = []
    by_negotiation: dict[str, list[ArgumentRecord]] = defaultdict(list)
    for argument in argument_index.values():
        by_negotiation[argument.negotiation_id].append(argument)

    for arguments in by_negotiation.values():
        ordered = sorted(arguments, key=lambda item: (item.step_id or 0, item.id))
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if (left.id, right.id) in directed_attacks or (right.id, left.id) in directed_attacks:
                    continue
                if _norm_text(left.text) == _norm_text(right.text):
                    continue
                score = 0
                if left.agent != right.agent:
                    score += 4
                if left.round == right.round:
                    score += 2
                if left.quality_dimension != right.quality_dimension:
                    score += 1
                if left.turn_type in {"proposal", "refinement"} and right.turn_type in {
                    "proposal",
                    "refinement",
                    "critique",
                }:
                    score += 1
                candidates.append((score, left, right, case_study, seed, source_file))

    candidates.sort(key=lambda item: (-item[0], item[1].id, item[2].id))
    return [(left, right, case_study, seed, source_file) for _, left, right, case_study, seed, source_file in candidates]


def _dedupe_attacks(attacks: list[AttackCandidate]) -> list[AttackCandidate]:
    by_text_pair: dict[tuple[str, str], AttackCandidate] = {}
    for attack in attacks:
        key = tuple(sorted((_norm_text(attack.source.text), _norm_text(attack.target.text))))
        existing = by_text_pair.get(key)
        if existing is None:
            by_text_pair[key] = attack
            continue
        existing.seeds.add(attack.source_seed)
        existing.source_files.add(attack.source_file)
    return sorted(
        by_text_pair.values(),
        key=lambda item: (CASE_ORDER.index(item.case_study), item.pattern, item.source.id, item.target.id),
    )


def _sample_attacks(attacks: list[AttackCandidate], rng: random.Random) -> list[AttackCandidate]:
    if len(attacks) < TOTAL_ATTACK_SAMPLES:
        raise ValueError(f"Need {TOTAL_ATTACK_SAMPLES} unique attacks, found {len(attacks)}")

    by_case: dict[str, list[AttackCandidate]] = defaultdict(list)
    by_pattern: dict[str, list[AttackCandidate]] = defaultdict(list)
    for attack in attacks:
        by_case[attack.case_study].append(attack)
        by_pattern[attack.pattern].append(attack)
    for values in by_case.values():
        rng.shuffle(values)
    for values in by_pattern.values():
        rng.shuffle(values)

    quotas = _case_quotas({case: len(by_case.get(case, [])) for case in CASE_ORDER})
    selected: list[AttackCandidate] = []
    selected_keys: set[tuple[str, str]] = set()
    case_counts: Counter[str] = Counter()

    for pattern in ("P1", "P2", "P3"):
        options = [item for item in by_pattern.get(pattern, []) if case_counts[item.case_study] < quotas[item.case_study]]
        if not options:
            raise ValueError(f"No candidates available for pattern {pattern}")
        choice = min(options, key=lambda item: (case_counts[item.case_study], item.case_study, item.source.id))
        _select_attack(choice, selected, selected_keys, case_counts)

    while len(selected) < TOTAL_ATTACK_SAMPLES:
        underfull_cases = [case for case in CASE_ORDER if case_counts[case] < quotas[case]]
        if not underfull_cases:
            break
        case = min(underfull_cases, key=lambda item: (case_counts[item], CASE_ORDER.index(item)))
        options = [item for item in by_case.get(case, []) if _attack_key(item) not in selected_keys]
        if not options:
            quotas[case] = case_counts[case]
            continue
        _select_attack(options.pop(), selected, selected_keys, case_counts)

    if len(selected) < TOTAL_ATTACK_SAMPLES:
        remaining = [item for item in attacks if _attack_key(item) not in selected_keys]
        rng.shuffle(remaining)
        for attack in remaining:
            _select_attack(attack, selected, selected_keys, case_counts)
            if len(selected) == TOTAL_ATTACK_SAMPLES:
                break

    return selected


def _case_quotas(case_sizes: dict[str, int]) -> dict[str, int]:
    # Keep the human-study sample close to balanced by case (25 -> 9/8/8), while
    # still redistributing any quota that a smaller case cannot satisfy.
    if sum(case_sizes.values()) <= 0:
        raise ValueError("No attack candidates available")
    base = TOTAL_ATTACK_SAMPLES // len(CASE_ORDER)
    quotas = {case: min(base, case_sizes.get(case, 0)) for case in CASE_ORDER}
    while sum(quotas.values()) < TOTAL_ATTACK_SAMPLES:
        case = max(CASE_ORDER, key=lambda item: case_sizes.get(item, 0) - quotas[item])
        if case_sizes.get(case, 0) <= quotas[case]:
            break
        quotas[case] += 1
    return quotas


def _select_attack(
    attack: AttackCandidate,
    selected: list[AttackCandidate],
    selected_keys: set[tuple[str, str]],
    case_counts: Counter[str],
) -> None:
    selected.append(attack)
    selected_keys.add(_attack_key(attack))
    case_counts[attack.case_study] += 1


def _sample_negative_controls(
    negative_pool: list[tuple[ArgumentRecord, ArgumentRecord, str, int, str]],
    attack_sample: list[AttackCandidate],
    rng: random.Random,
) -> list[tuple[ArgumentRecord, ArgumentRecord, str, int, str]]:
    attack_text_keys = {
        tuple(sorted((_norm_text(item.source.text), _norm_text(item.target.text))))
        for item in attack_sample
    }
    by_case: dict[str, list[tuple[ArgumentRecord, ArgumentRecord, str, int, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for left, right, case_study, seed, source_file in negative_pool:
        key = tuple(sorted((_norm_text(left.text), _norm_text(right.text))))
        if key in seen or key in attack_text_keys:
            continue
        seen.add(key)
        by_case[case_study].append((left, right, case_study, seed, source_file))

    for values in by_case.values():
        rng.shuffle(values)

    selected: list[tuple[ArgumentRecord, ArgumentRecord, str, int, str]] = []
    case_cycle = ["AD", "ATM", "Bookkeeping", "AD", "ATM"]
    for case in case_cycle:
        if by_case.get(case):
            selected.append(by_case[case].pop())
        if len(selected) == TOTAL_NEGATIVE_CONTROLS:
            return selected

    remaining = [item for values in by_case.values() for item in values]
    rng.shuffle(remaining)
    selected.extend(remaining[: TOTAL_NEGATIVE_CONTROLS - len(selected)])
    if len(selected) < TOTAL_NEGATIVE_CONTROLS:
        raise ValueError(f"Need {TOTAL_NEGATIVE_CONTROLS} negative controls, found {len(selected)}")
    return selected


def _build_samples(
    attack_sample: list[AttackCandidate],
    negative_sample: list[tuple[ArgumentRecord, ArgumentRecord, str, int, str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in attack_sample:
        rows.append(
            {
                "argument_a": _argument_payload(attack.source),
                "argument_b": _argument_payload(attack.target),
                "case_study": attack.case_study,
                "source_seed": attack.source_seed,
                "_hidden_label": "detected_attack",
                "_hidden_pattern": attack.pattern,
                "_hidden_raw_pattern": attack.raw_pattern,
                "_hidden_source_file": attack.source_file,
                "_hidden_source_seed_count": len(attack.seeds),
                "_hidden_all_source_seeds": sorted(attack.seeds),
                "_hidden_all_source_files": sorted(attack.source_files),
                "_hidden_confidence": attack.confidence,
                "_hidden_reason": attack.reason,
                "_hidden_grounded_source_accepted": attack.grounded_source_accepted,
                "_hidden_grounded_target_accepted": attack.grounded_target_accepted,
            }
        )

    for left, right, case_study, seed, source_file in negative_sample:
        rows.append(
            {
                "argument_a": _argument_payload(left),
                "argument_b": _argument_payload(right),
                "case_study": case_study,
                "source_seed": seed,
                "_hidden_label": "no_attack",
                "_hidden_pattern": None,
                "_hidden_raw_pattern": None,
                "_hidden_source_file": source_file,
                "_hidden_source_seed_count": 1,
                "_hidden_all_source_seeds": [seed],
                "_hidden_all_source_files": [source_file],
                "_hidden_confidence": None,
                "_hidden_reason": "Negative control: no attack relation in either direction.",
                "_hidden_grounded_source_accepted": None,
                "_hidden_grounded_target_accepted": None,
            }
        )

    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["item_id"] = f"CV-{index:02d}"
    return [{"item_id": row.pop("item_id"), **row} for row in rows]


def _argument_payload(argument: ArgumentRecord) -> dict[str, Any]:
    return {
        "id": argument.id,
        "text": argument.text,
        "agent": argument.agent,
        "round": argument.round,
        "quality_dimension": argument.quality_dimension,
        "kaos_level": argument.kaos_level,
        "context_snippet": argument.context_snippet,
    }


def _build_reveal_mapping(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata": {
            "sample_seed": SAMPLE_SEED,
            "detected_attack_items": sum(
                1 for item in samples if item["_hidden_label"] == "detected_attack"
            ),
            "negative_control_items": sum(1 for item in samples if item["_hidden_label"] == "no_attack"),
        },
        "items": {
            item["item_id"]: {
                "label": item["_hidden_label"],
                "pattern": item["_hidden_pattern"],
                "raw_pattern": item["_hidden_raw_pattern"],
                "case_study": item["case_study"],
                "source_seed": item["source_seed"],
                "source_seed_count": item["_hidden_source_seed_count"],
                "all_source_seeds": item["_hidden_all_source_seeds"],
                "source_file": item["_hidden_source_file"],
                "all_source_files": item["_hidden_all_source_files"],
                "argument_a_id": item["argument_a"]["id"],
                "argument_b_id": item["argument_b"]["id"],
                "confidence": item["_hidden_confidence"],
                "reason": item["_hidden_reason"],
                "grounded_source_accepted": item["_hidden_grounded_source_accepted"],
                "grounded_target_accepted": item["_hidden_grounded_target_accepted"],
            }
            for item in samples
        },
    }


def _print_summary(
    *,
    run_records: list[Path],
    raw_attacks: list[AttackCandidate],
    unique_attacks: list[AttackCandidate],
    samples: list[dict[str, Any]],
    stats_by_case_seed: Counter[tuple[str, int]],
    stats_by_pattern: Counter[str],
    negative_pool: list[tuple[ArgumentRecord, ArgumentRecord, str, int, str]],
    samples_path: Path,
    reveal_path: Path,
) -> None:
    print("Conflict-validation extraction summary")
    print(f"Run records found: {len(run_records)}")
    print(f"Raw attacks found: {len(raw_attacks)}")
    print(f"Unique attack text pairs after seed de-duplication: {len(unique_attacks)}")
    print(f"Negative-control candidate pairs: {len(negative_pool)}")
    print("\nRaw attacks per case/seed:")
    for case in CASE_ORDER:
        for seed in (101, 202, 303):
            print(f"  {case} seed {seed}: {stats_by_case_seed[(case, seed)]}")
    print("\nRaw attacks per pattern:")
    for pattern, count in sorted(stats_by_pattern.items()):
        print(f"  {pattern}: {count}")
    print("\nFinal sample composition:")
    label_counts = Counter(item["_hidden_label"] for item in samples)
    case_counts = Counter(item["case_study"] for item in samples)
    pattern_counts = Counter(item["_hidden_pattern"] for item in samples if item["_hidden_pattern"])
    print(f"  Labels: {dict(label_counts)}")
    print(f"  Cases: {dict(case_counts)}")
    print(f"  Attack patterns: {dict(pattern_counts)}")
    print(f"\nWrote {samples_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {reveal_path.relative_to(REPO_ROOT)}")


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _attack_key(attack: AttackCandidate) -> tuple[str, str]:
    return (_norm_text(attack.source.text), _norm_text(attack.target.text))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seed_from_text(text: str) -> int:
    match = re.search(r"s(\d+)", text)
    return int(match.group(1)) if match else 0


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
