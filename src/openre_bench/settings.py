"""Backward-compatible shim — use ``openre_bench.llm`` instead.

.. deprecated:: 0.2.0
   All settings functionality has moved to :mod:`openre_bench.llm`.
"""

from openre_bench.llm import MissingAPIKeyError  # noqa: F401
from openre_bench.llm import OpenAISettings  # noqa: F401
from openre_bench.llm import load_openai_settings  # noqa: F401
