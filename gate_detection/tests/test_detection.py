"""
Tests for ski_racing/detection.py

Focuses on the detect_from_best_frame tie-breaking logic:
  - Primary: frame with more gates wins outright.
  - Tie-break: equal gate count → higher mean confidence wins,
    but only when the margin exceeds +0.02 (avoids churn).
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ski_racing.detection import GateDetector  # noqa: E402


def _make_gates(n, mean_conf):
    """Return n fake gate dicts all with the given confidence."""
    return [
        {
            "class": 0,
            "class_name": "gate",
            "center_x": float(i * 100),
            "center_y": 300.0,
            "base_y": 350.0,
            "bbox": [i * 100, 270, i * 100 + 40, 350],
            "confidence": float(mean_conf),
        }
        for i in range(n)
    ]


class TestDetectFromBestFrameTieBreak(unittest.TestCase):
    """
    Unit-test detect_from_best_frame without touching the file system.

    Strategy: patch cv2.VideoCapture and GateDetector.detect_in_frame so the
    method "sees" a fixed sequence of (frame_idx, gates) pairs and we can
    assert which result it picks.
    """

    def _run(self, frame_gate_sequence):
        """
        Run detect_from_best_frame against a fake video.

        frame_gate_sequence: list of gate-lists, one per frame (stride=1).
        Returns the best_gates list chosen by the method.
        """
        # Build a minimal fake VideoCapture that yields one frame per read()
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        # Each read() call returns (True, fake_frame) until exhausted, then (False, None)
        fake_frames = [(True, MagicMock()) for _ in frame_gate_sequence]
        fake_frames.append((False, None))
        fake_cap.read.side_effect = fake_frames

        detector = GateDetector.__new__(GateDetector)  # skip __init__ (no model needed)
        detector.model = MagicMock()

        # detect_in_frame returns the pre-defined gates for each frame index
        call_count = {"n": 0}

        def fake_detect(frame, conf, iou):
            idx = call_count["n"]
            call_count["n"] += 1
            return frame_gate_sequence[idx]

        with patch("cv2.VideoCapture", return_value=fake_cap):
            with patch.object(detector, "detect_in_frame", side_effect=fake_detect):
                result = detector.detect_from_best_frame(
                    "fake_video.mp4",
                    conf=0.35,
                    iou=0.55,
                    max_frames=len(frame_gate_sequence),
                    stride=1,
                )
        return result

    # ── Scenario A: equal count, second frame has meaningfully higher conf ──

    def test_equal_count_higher_conf_wins(self):
        """Second frame: same count, mean_conf 0.10 higher → second wins."""
        frame0_gates = _make_gates(3, mean_conf=0.50)
        frame1_gates = _make_gates(3, mean_conf=0.60)  # +0.10 > +0.02 margin
        result = self._run([frame0_gates, frame1_gates])
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]["confidence"], 0.60)

    # ── Scenario B: equal count, conf improvement below +0.02 margin ────────

    def test_equal_count_conf_below_margin_keeps_first(self):
        """Second frame: same count, mean_conf only 0.01 higher → first wins."""
        frame0_gates = _make_gates(3, mean_conf=0.50)
        frame1_gates = _make_gates(3, mean_conf=0.51)  # +0.01 < +0.02 margin
        result = self._run([frame0_gates, frame1_gates])
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]["confidence"], 0.50)

    # ── Scenario C: second frame has strictly more gates ────────────────────

    def test_more_gates_wins_regardless_of_conf(self):
        """Second frame: one more gate, even at lower confidence → second wins."""
        frame0_gates = _make_gates(3, mean_conf=0.90)
        frame1_gates = _make_gates(4, mean_conf=0.40)
        result = self._run([frame0_gates, frame1_gates])
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0]["confidence"], 0.40)

    # ── Scenario D: empty frames before a good one ───────────────────────────

    def test_empty_then_good_frame(self):
        """First two frames have no gates; third has 2 → third wins."""
        result = self._run([[], [], _make_gates(2, mean_conf=0.70)])
        self.assertEqual(len(result), 2)

    # ── Scenario E: exact +0.02 margin is not enough ────────────────────────

    def test_exact_margin_boundary_keeps_first(self):
        """Second frame at exactly +0.02 over first → first still wins (strict >)."""
        frame0_gates = _make_gates(2, mean_conf=0.50)
        frame1_gates = _make_gates(2, mean_conf=0.52)  # exactly +0.02, not > +0.02
        result = self._run([frame0_gates, frame1_gates])
        self.assertAlmostEqual(result[0]["confidence"], 0.50)


if __name__ == "__main__":
    unittest.main()
