"""Test configuration for local package imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Fast deterministic pipeline: disable paper-default BERT/Chroma/LLM tooling unless a test opts in.
os.environ.setdefault("OPENRE_PAPER_TOOLS", "0")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

