#!/usr/bin/env python3
"""
DEPRECATED: This is a compatibility shim.
Use ``scripts/run_eval.py`` as the canonical evaluation entrypoint.

This file forwards to the canonical implementation so that any existing
scripts that import from ``scripts.evaluation.run_eval`` keep working.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-export everything from the canonical module
from scripts.run_eval import *  # noqa: F401,F403
from scripts.run_eval import main  # noqa: F401

if __name__ == "__main__":
    print(
        "WARNING: scripts/evaluation/run_eval.py is deprecated. "
        "Use scripts/run_eval.py instead.",
        file=sys.stderr,
    )
    main()
