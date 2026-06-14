"""
Smoke tests for both process_video.py entry points.

Verifies that:
  - scripts/process_video.py (canonical) exits cleanly on a synthetic video.
  - scripts/inference/process_video.py (shim) delegates correctly and exits cleanly.
  - Deprecated flags (--gate-spacing, --no-physics, etc.) are accepted without
    "unrecognized argument" errors (DeprecationWarning is OK).

Strategy: create a tiny synthetic MP4 with cv2, then subprocess-invoke each
entry point with --gate-model pointing to a non-existent model path so YOLO
loading will fail.  We accept *either* a clean run (status=ok) or a graceful
error that writes status=error to run_summary.json (not an unhandled crash /
non-zero exit from bad CLI parsing).

Note: because the gate-model is a dummy path, the pipeline will raise inside
the per-video try/except, writing {"status": "error"} to run_summary.json.
That is the expected outcome — the smoke test verifies that:
  1. The script *starts* without TypeError / unrecognized-argument errors.
  2. run_summary.json is written and contains a list with the expected entry.
"""
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = str(PROJECT_ROOT / "scripts" / "process_video.py")
SHIM = str(PROJECT_ROOT / "scripts" / "inference" / "process_video.py")
DUMMY_MODEL = str(PROJECT_ROOT / "models" / "gate_detector_best.pt")


def _make_synthetic_video(path: str, frames: int = 30, fps: int = 30) -> None:
    """Write a tiny solid-grey MP4 to *path*."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    for _ in range(frames):
        out.write(frame)
    out.release()


def _run_entry_point(script: str, video: str, output_dir: str,
                     extra_args=()) -> subprocess.CompletedProcess:
    """Run *script* in a subprocess; return the CompletedProcess."""
    cmd = [
        sys.executable, script,
        video,
        "--gate-model", DUMMY_MODEL,
        "--output-dir", output_dir,
        "--max-frames", "10",
    ] + list(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestCanonicalEntryPoint(unittest.TestCase):
    """scripts/process_video.py smoke tests."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.video = str(self.tmp / "smoke_test.mp4")
        _make_synthetic_video(self.video)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _assert_summary_written(self, output_dir: str, video_name: str) -> dict:
        """Return the summary entry; fail if summary missing or malformed."""
        summary_path = Path(output_dir) / "run_summary.json"
        self.assertTrue(summary_path.exists(),
                        f"run_summary.json not written to {output_dir}")
        data = json.loads(summary_path.read_text())
        self.assertIsInstance(data, list, "run_summary.json must be a list")
        self.assertGreater(len(data), 0, "run_summary.json must not be empty")
        entry = data[0]
        self.assertEqual(entry.get("video"), video_name,
                         f"Expected video={video_name!r}, got {entry!r}")
        self.assertIn(entry.get("status"), ("ok", "error"),
                      f"status must be 'ok' or 'error', got {entry!r}")
        return entry

    # ── Test 1: canonical script starts and writes run_summary.json ──────────

    def test_canonical_runs_without_crash(self):
        """Canonical entry point should not crash (exit 0 or 1 — not 2)."""
        out_dir = str(self.tmp / "out1")
        proc = _run_entry_point(CANONICAL, self.video, out_dir)
        # Exit code 2 means argparse error (unrecognized argument etc.)
        self.assertNotEqual(proc.returncode, 2,
                            f"argparse error:\n{proc.stderr}")
        self._assert_summary_written(out_dir, "smoke_test.mp4")

    # ── Test 2: deprecated flags pass silently (no argparse exit-2) ─────────

    def test_canonical_accepts_deprecated_flags(self):
        """Deprecated flags must NOT trigger 'unrecognized argument' (exit 2)."""
        out_dir = str(self.tmp / "out2")
        proc = _run_entry_point(
            CANONICAL, self.video, out_dir,
            extra_args=[
                "--gate-spacing", "9.5",
                "--no-physics",
                "--projection", "scale",
                "--camera-mode", "affine",
                "--camera-pitch-deg", "6.0",
            ],
        )
        self.assertNotEqual(proc.returncode, 2,
                            f"argparse error on deprecated flags:\n{proc.stderr}")
        self._assert_summary_written(out_dir, "smoke_test.mp4")


class TestShimEntryPoint(unittest.TestCase):
    """scripts/inference/process_video.py shim smoke tests."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.video = str(self.tmp / "smoke_shim.mp4")
        _make_synthetic_video(self.video)

    def tearDown(self):
        self._tmpdir.cleanup()

    # ── Test 3: shim delegates to canonical, writes run_summary.json ────────

    def test_shim_delegates_and_writes_summary(self):
        """Shim must start without crash and produce run_summary.json."""
        out_dir = str(self.tmp / "out3")
        proc = _run_entry_point(SHIM, self.video, out_dir)
        self.assertNotEqual(proc.returncode, 2,
                            f"argparse/delegation error:\n{proc.stderr}")
        summary_path = Path(out_dir) / "run_summary.json"
        self.assertTrue(summary_path.exists(),
                        "Shim did not produce run_summary.json")
        data = json.loads(summary_path.read_text())
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn(data[0].get("status"), ("ok", "error"))

    # ── Test 4: shim also accepts deprecated flags ───────────────────────────

    def test_shim_accepts_deprecated_flags(self):
        """Deprecated flags forwarded through shim must not trigger exit 2."""
        out_dir = str(self.tmp / "out4")
        proc = _run_entry_point(
            SHIM, self.video, out_dir,
            extra_args=["--gate-spacing", "27.0", "--no-physics"],
        )
        self.assertNotEqual(proc.returncode, 2,
                            f"argparse error through shim:\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
