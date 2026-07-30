"""Shared test configuration.

The repository root is placed on ``sys.path`` here as well as through
``pyproject.toml``'s ``pythonpath`` setting, so the suite collects correctly
whether it is invoked as ``pytest``, ``python -m pytest``, or from another
working directory.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
