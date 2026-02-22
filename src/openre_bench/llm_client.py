"""Backward-compatible shim — use ``openre_bench.llm`` instead.

.. deprecated:: 0.2.0
   All LLM client functionality has moved to :mod:`openre_bench.llm`.
"""

from openre_bench.llm import LLMClient  # noqa: F401
from openre_bench.llm import LLMClientError  # noqa: F401
