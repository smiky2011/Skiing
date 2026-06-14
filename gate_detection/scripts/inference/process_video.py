"""
Thin delegation shim — all logic lives in scripts/process_video.py.

This entry point exists so that callers that discovered the script at
``scripts/inference/process_video.py`` keep working without change.
It forwards *all* CLI arguments transparently to the canonical script.

Usage (identical to the canonical script):
    python scripts/inference/process_video.py VIDEO_PATH \
        --gate-model models/gate_detector_best.pt --summary
"""
import runpy
import sys
from pathlib import Path

# Ensure project root is on sys.path so both the canonical script and the
# ski_racing package can be imported correctly from any working directory.
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Run the canonical script as __main__, preserving sys.argv so all flags
# (including deprecated shims defined there) are forwarded unchanged.
runpy.run_path(
    str(_project_root / "scripts" / "process_video.py"),
    run_name="__main__",
)
