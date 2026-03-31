"""Argumentation framework solvers (grounded / preferred / priority-guided)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from openre_bench.argumentation.argument_parser import Argument
from openre_bench.argumentation.attack_builder import AttackRelation


@dataclass(frozen=True)
class AFSolution:
    """Container for computed argumentation semantics."""

    grounded_extension: tuple[str, ...]
    preferred_extensions: tuple[tuple[str, ...], ...]
    selected_extension: tuple[str, ...]
    selection_strategy: str


def solve_argumentation_framework(
    *,
    arguments: list[Argument],
    attacks: list[AttackRelation],
    priority_weights: dict[str, float] | None = None,
    strategy: str = "preferred_priority",
) -> AFSolution:
    """Compute grounded/preferred extensions and one selected extension."""

    argument_ids = sorted({item.argument_id for item in arguments})
    grounded = tuple(sorted(compute_grounded_extension(argument_ids=argument_ids, attacks=attacks)))
    preferred_sets = compute_preferred_extensions(argument_ids=argument_ids, attacks=attacks)
    preferred = tuple(tuple(sorted(ext)) for ext in preferred_sets)
    if strategy == "grounded":
        selected = grounded
    else:
        selected = tuple(
            sorted(
                select_priority_guided_extension(
                    preferred_extensions=preferred_sets,
                    arguments=arguments,
                    priority_weights=priority_weights or {},
                )
            )
        )
    return AFSolution(
        grounded_extension=grounded,
        preferred_extensions=preferred,
        selected_extension=selected,
        selection_strategy=strategy,
    )


def compute_grounded_extension(
    *,
    argument_ids: list[str],
    attacks: list[AttackRelation],
) -> set[str]:
    """Least fixed-point grounded extension under Dung semantics."""

    attack_pairs = {(item.attacker_id, item.target_id) for item in attacks}
    extension: set[str] = set()
    while True:
        defended = _characteristic_function(extension, argument_ids, attack_pairs)
        if defended == extension:
            return extension
        extension = defended


def compute_preferred_extensions(
    *,
    argument_ids: list[str],
    attacks: list[AttackRelation],
) -> list[set[str]]:
    """Compute all maximal admissible (preferred) extensions."""

    attack_pairs = {(item.attacker_id, item.target_id) for item in attacks}
    admissible_sets: list[set[str]] = []
    for subset in _powerset(argument_ids):
        candidate = set(subset)
        if _is_admissible(candidate, attack_pairs):
            admissible_sets.append(candidate)

    preferred: list[set[str]] = []
    for candidate in admissible_sets:
        if any(candidate < other for other in admissible_sets):
            continue
        preferred.append(candidate)
    preferred.sort(key=lambda ext: (len(ext), sorted(ext)), reverse=True)
    return preferred


def select_priority_guided_extension(
    *,
    preferred_extensions: list[set[str]],
    arguments: list[Argument],
    priority_weights: dict[str, float],
) -> set[str]:
    """Select one preferred extension by priority sum then lexicographic tie-break."""

    if not preferred_extensions:
        return set()

    argument_map = {item.argument_id: item for item in arguments}

    def _score(ext: set[str]) -> tuple[float, int, tuple[str, ...]]:
        total = 0.0
        for arg_id in ext:
            argument = argument_map.get(arg_id)
            total += _argument_priority_weight(argument, priority_weights)
        return total, len(ext), tuple(sorted(ext))

    return max(preferred_extensions, key=_score)


def _characteristic_function(
    extension: set[str],
    argument_ids: list[str],
    attack_pairs: set[tuple[str, str]],
) -> set[str]:
    accepted: set[str] = set()
    for candidate in argument_ids:
        attackers = _attackers_of(candidate, attack_pairs)
        if all(_is_attacked_by_some(attacker, extension, attack_pairs) for attacker in attackers):
            accepted.add(candidate)
    return accepted


def _is_admissible(extension: set[str], attack_pairs: set[tuple[str, str]]) -> bool:
    if not _is_conflict_free(extension, attack_pairs):
        return False
    for argument_id in extension:
        attackers = _attackers_of(argument_id, attack_pairs)
        if not all(_is_attacked_by_some(attacker, extension, attack_pairs) for attacker in attackers):
            return False
    return True


def _is_conflict_free(extension: set[str], attack_pairs: set[tuple[str, str]]) -> bool:
    for attacker, target in attack_pairs:
        if attacker in extension and target in extension:
            return False
    return True


def _attackers_of(target: str, attack_pairs: set[tuple[str, str]]) -> set[str]:
    return {attacker for attacker, attacked in attack_pairs if attacked == target}


def _is_attacked_by_some(
    target: str,
    attackers: set[str],
    attack_pairs: set[tuple[str, str]],
) -> bool:
    return any((attacker, target) in attack_pairs for attacker in attackers)


def _powerset(items: list[str]) -> list[tuple[str, ...]]:
    ordered = sorted(items)
    subsets: list[tuple[str, ...]] = []
    for size in range(len(ordered) + 1):
        subsets.extend(itertools.combinations(ordered, size))
    return subsets


def _argument_priority_weight(argument: Argument | None, priority_weights: dict[str, float]) -> float:
    if argument is None:
        return 0.0
    quality = str(argument.quality_attribute).strip()
    if quality in priority_weights:
        return float(priority_weights[quality])
    if quality == "Sustainability" and "Green" in priority_weights:
        return float(priority_weights["Green"])
    return float(priority_weights.get("Integrated", 0.0))
