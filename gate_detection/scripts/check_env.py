#!/usr/bin/env python3
"""
Environment check script — run this BEFORE processing videos.

Verifies all required dependencies are correctly installed,
with special focus on the ByteTrack linear assignment solver (lap/lapjv)
which is required for skier tracking and has no acceptable fallback.

Usage:
    python scripts/check_env.py

Expected output when everything is OK:
    [OK] ultralytics
    [OK] lap (ByteTrack solver)
    [OK] ByteTrack end-to-end
    [OK] opencv-python
    [OK] torch
    [OK] numpy
    [OK] matplotlib
    All checks passed. Environment is ready.
"""
import sys

PASS = "[OK] "
FAIL = "[FAIL] "
WARN = "[WARN] "

errors = []
warnings = []


def check(label, fn):
    try:
        result = fn()
        msg = f" ({result})" if result else ""
        print(f"{PASS}{label}{msg}")
        return True
    except Exception as e:
        print(f"{FAIL}{label}: {e}")
        errors.append((label, str(e)))
        return False


def check_warn(label, fn):
    try:
        result = fn()
        msg = f" ({result})" if result else ""
        print(f"{PASS}{label}{msg}")
        return True
    except Exception as e:
        print(f"{WARN}{label}: {e}")
        warnings.append((label, str(e)))
        return False


print("=" * 55)
print("Environment check for Alpine Ski Racing AI pipeline")
print("=" * 55)
print()

# --- Core ML dependencies ---
print("--- Core dependencies ---")

check("ultralytics", lambda: __import__("ultralytics").__version__)
check("torch", lambda: __import__("torch").__version__)
check("opencv-python", lambda: __import__("cv2").__version__)
check("numpy", lambda: __import__("numpy").__version__)
check("matplotlib", lambda: __import__("matplotlib").__version__)

print()

# --- ByteTrack solver (CRITICAL) ---
print("--- ByteTrack solver (CRITICAL for skier tracking) ---")

lap_ok = check("lap (ByteTrack solver)", lambda: (
    __import__("lap"),
    "ok"
)[1])

if not lap_ok:
    lapjv_ok = check_warn("lapjv (alternative solver)", lambda: (
        __import__("lapjv"),
        "ok"
    )[1])
    if not lapjv_ok:
        print()
        print("  *** CRITICAL: Neither lap nor lapjv is installed! ***")
        print("  ByteTrack will raise a RuntimeError when processing videos.")
        print()
        print("  Install with ONE of:")
        print("    pip install lap>=0.5.12")
        print("    pip install lapjv           (if lap fails to build)")
        print()
        errors.append(("lap/lapjv", "Neither package is installed"))

# --- ByteTrack end-to-end smoke test ---
print()
print("--- ByteTrack end-to-end smoke test ---")

def bytetrack_smoke_test():
    from ultralytics import YOLO
    import numpy as np
    # The 'track' call is what triggers the lap import internally.
    # We instantiate the model but don't actually run tracking here
    # (that requires a video). Instead we verify the import chain works.
    # Import the ByteTracker class directly to catch the lap dependency.
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except ImportError:
        # Older ultralytics versions use a different path
        from ultralytics.trackers import BYTETracker
    # Try to instantiate with minimal args to trigger the lap import
    args = type("args", (), {
        "track_high_thresh": 0.3,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.4,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
    })()
    tracker = BYTETracker(args, frame_rate=30)
    return "import chain OK"

check("ByteTrack end-to-end", bytetrack_smoke_test)

# --- Model files ---
print()
print("--- Model files ---")

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

def check_model(name, path):
    full = project_root / path
    if full.exists():
        size_mb = full.stat().st_size / 1024 / 1024
        return f"{size_mb:.1f} MB"
    raise FileNotFoundError(f"Not found: {full}")

check("gate_detector_best.pt", lambda: check_model("gate detector", "models/gate_detector_best.pt"))
check_warn("yolov8n.pt (person tracker)", lambda: check_model("yolov8n", "models/yolov8n.pt"))
check_warn("yolov8s.pt (person tracker)", lambda: check_model("yolov8s", "models/yolov8s.pt"))

# --- Summary ---
print()
print("=" * 55)
if errors:
    print(f"RESULT: {len(errors)} error(s) found — fix before running tests")
    for label, msg in errors:
        print(f"  ✗ {label}: {msg}")
    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for label, msg in warnings:
            print(f"  ! {label}: {msg}")
    sys.exit(1)
elif warnings:
    print(f"RESULT: All critical checks passed, {len(warnings)} warning(s)")
    for label, msg in warnings:
        print(f"  ! {label}: {msg}")
    print("\nEnvironment is ready (warnings are non-blocking).")
    sys.exit(0)
else:
    print("RESULT: All checks passed. Environment is ready.")
    sys.exit(0)
