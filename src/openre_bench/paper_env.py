"""Environment-driven defaults for paper-faithful OpenRE-Bench / ArgRE tooling.

Tests set ``OPENRE_PAPER_TOOLS=0`` (see ``tests/conftest.py``) for fast deterministic
runs. In normal use the variable is unset and paper-aligned paths default **on**.
"""

from __future__ import annotations

import os


def paper_tools_enabled() -> bool:
    """Return whether paper-faithful defaults should be active."""

    raw = os.environ.get("OPENRE_PAPER_TOOLS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return True


def default_rag_backend() -> str:
    """Phase~1 RAG: Chroma + ada-002 when paper tools on, else lexical overlap."""

    return "chroma_ada002" if paper_tools_enabled() else "local_tfidf"
