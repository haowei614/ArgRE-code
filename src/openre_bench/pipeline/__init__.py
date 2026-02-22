"""Pipeline package — protocol-parity artifact generation for OpenRE-Bench.

This package re-exports the public API from the internal ``_core`` module.
All existing ``from openre_bench.pipeline import X`` continues to work.
"""

from openre_bench.pipeline._core import MARE_ROLE_QUALITY_ATTRIBUTES  # noqa: F401
from openre_bench.pipeline._core import MARE_RUNTIME_SEMANTICS_MODE  # noqa: F401
from openre_bench.pipeline._core import MareRuntimeExecutionMeta  # noqa: F401
from openre_bench.pipeline._core import Phase2ExecutionMeta  # noqa: F401
from openre_bench.pipeline._core import Phase2LLMClient  # noqa: F401
from openre_bench.pipeline._core import PipelineConfig  # noqa: F401
from openre_bench.pipeline._core import _latest_backward_elements  # noqa: F401
from openre_bench.pipeline._core import default_run_id  # noqa: F401
from openre_bench.pipeline._core import run_case_pipeline  # noqa: F401

__all__ = [
    "MARE_ROLE_QUALITY_ATTRIBUTES",
    "MARE_RUNTIME_SEMANTICS_MODE",
    "MareRuntimeExecutionMeta",
    "Phase2ExecutionMeta",
    "Phase2LLMClient",
    "PipelineConfig",
    "_latest_backward_elements",
    "default_run_id",
    "run_case_pipeline",
]
