"""Argumentation-layer primitives for QUARE phase-2 extensions."""

from openre_bench.argumentation.argument_parser import Argument
from openre_bench.argumentation.argument_parser import ArgumentParseMeta
from openre_bench.argumentation.argument_parser import parse_phase2_arguments
from openre_bench.argumentation.attack_builder import AttackBuildMeta
from openre_bench.argumentation.attack_builder import AttackRelation
from openre_bench.argumentation.attack_builder import build_attack_relations
from openre_bench.argumentation.af_solver import AFSolution
from openre_bench.argumentation.af_solver import compute_grounded_extension
from openre_bench.argumentation.af_solver import compute_preferred_extensions
from openre_bench.argumentation.af_solver import select_priority_guided_extension
from openre_bench.argumentation.af_solver import solve_argumentation_framework

__all__ = [
    "Argument",
    "ArgumentParseMeta",
    "AttackRelation",
    "AttackBuildMeta",
    "AFSolution",
    "parse_phase2_arguments",
    "build_attack_relations",
    "compute_grounded_extension",
    "compute_preferred_extensions",
    "select_priority_guided_extension",
    "solve_argumentation_framework",
]
