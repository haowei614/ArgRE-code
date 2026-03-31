"""AF-specific evaluation metrics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from openre_bench.schemas import load_json_file


def compute_af_metrics(artifacts_dir: Path) -> dict[str, Any]:
    """Compute TC/SD and AF graph statistics from argumentation artifacts."""

    graph_path = artifacts_dir / "argumentation_graph.json"
    if not graph_path.exists():
        return {
            "tc": 0.0,
            "sd": 0.0,
            "af_num_arguments": 0,
            "af_num_attacks": 0,
            "af_num_rule_attacks": 0,
            "af_num_llm_attacks": 0,
            "af_type_distribution": {},
            "af_agent_distribution": {},
        }

    graph = load_json_file(graph_path)
    arguments = graph.get("arguments", []) if isinstance(graph, dict) else []
    attacks = graph.get("attacks", []) if isinstance(graph, dict) else []
    grounded = set(_as_str_list(graph.get("grounded_extension", [])) if isinstance(graph, dict) else [])
    preferred_raw = graph.get("preferred_extensions", []) if isinstance(graph, dict) else []
    preferred_sets = [set(_as_str_list(item)) for item in preferred_raw if isinstance(item, list)]
    selected = set(_as_str_list(graph.get("selected_extension", [])) if isinstance(graph, dict) else [])

    argument_map = {
        str(item.get("argument_id", "")).strip(): item
        for item in arguments
        if isinstance(item, dict) and str(item.get("argument_id", "")).strip()
    }
    attack_pairs = _attack_pairs(attacks)

    tc = _trace_completeness(selected_ids=selected, argument_map=argument_map, attack_pairs=attack_pairs)
    sd = _semantics_divergence(grounded=grounded, preferred_sets=preferred_sets)
    type_distribution, agent_distribution = _distributions(arguments)

    rule_attacks = 0
    llm_attacks = 0
    for item in attacks:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip().lower()
        if source.startswith("rule:"):
            rule_attacks += 1
        elif source == "llm":
            llm_attacks += 1

    return {
        "tc": round(tc, 6),
        "sd": round(sd, 6),
        "af_num_arguments": len(argument_map),
        "af_num_attacks": len(attack_pairs),
        "af_num_rule_attacks": rule_attacks,
        "af_num_llm_attacks": llm_attacks,
        "af_type_distribution": type_distribution,
        "af_agent_distribution": agent_distribution,
    }


def _trace_completeness(
    *,
    selected_ids: set[str],
    argument_map: dict[str, dict[str, Any]],
    attack_pairs: set[tuple[str, str]],
) -> float:
    if not selected_ids:
        return 0.0
    proposals = {
        arg_id
        for arg_id in selected_ids
        if str(argument_map.get(arg_id, {}).get("turn_type", "")).strip().lower() == "proposal"
    }
    if not proposals:
        return 0.0
    incoming: dict[str, set[str]] = {}
    outgoing: dict[str, set[str]] = {}
    for attacker, target in attack_pairs:
        outgoing.setdefault(attacker, set()).add(target)
        incoming.setdefault(target, set()).add(attacker)

    def _reachable_proposal(start: str) -> bool:
        if start in proposals:
            return True
        visited = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            neighbors = incoming.get(current, set()) | outgoing.get(current, set())
            for nxt in neighbors:
                if nxt in visited:
                    continue
                if nxt in proposals:
                    return True
                visited.add(nxt)
                stack.append(nxt)
        return False

    satisfied = sum(1 for arg_id in selected_ids if _reachable_proposal(arg_id))
    return satisfied / max(1, len(selected_ids))


def _semantics_divergence(*, grounded: set[str], preferred_sets: list[set[str]]) -> float:
    if not preferred_sets:
        return 0.0
    distances: list[float] = []
    for preferred in preferred_sets:
        union = grounded | preferred
        if not union:
            distances.append(0.0)
            continue
        inter = grounded & preferred
        jaccard = len(inter) / len(union)
        distances.append(1.0 - jaccard)
    return sum(distances) / len(distances)


def _distributions(arguments: list[Any]) -> tuple[dict[str, int], dict[str, int]]:
    type_counter: Counter[str] = Counter()
    agent_counter: Counter[str] = Counter()
    for item in arguments:
        if not isinstance(item, dict):
            continue
        turn_type = str(item.get("turn_type", "")).strip()
        focus_agent = str(item.get("focus_agent", "")).strip()
        if turn_type:
            type_counter[turn_type] += 1
        if focus_agent:
            agent_counter[focus_agent] += 1
    return dict(sorted(type_counter.items())), dict(sorted(agent_counter.items()))


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _attack_pairs(attacks: list[Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in attacks:
        if not isinstance(item, dict):
            continue
        attacker = str(item.get("attacker_id", "")).strip()
        target = str(item.get("target_id", "")).strip()
        if attacker and target:
            pairs.add((attacker, target))
    return pairs
