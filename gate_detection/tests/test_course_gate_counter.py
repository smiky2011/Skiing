import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ski_racing.detection import CourseGateCounter


def _gate(cx, by, conf=0.7, cls=0, name="gate"):
    """Create a gate detection dict."""
    return {
        "class": int(cls),
        "class_name": str(name),
        "center_x": float(cx),
        "center_y": float(by - 40.0),
        "base_y": float(by),
        "bbox": [float(cx - 15), float(by - 80), float(cx + 15), float(by)],
        "confidence": float(conf),
    }


def _make_fake_cap(n_frames):
    """Create a fake cv2.VideoCapture that returns n_frames frames then stops."""
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    frames = [(True, MagicMock(shape=(720, 1280, 3))) for _ in range(n_frames)]
    frames.append((False, None))
    fake_cap.read.side_effect = frames
    return fake_cap


def _make_detector(per_frame_gates):
    """
    Create a fake GateDetector that returns per_frame_gates[call_index] on each
    detect_in_frame call.
    """
    detector = MagicMock()
    detector.model = MagicMock()
    detector.model.names = {0: "gate", 1: "blue_gate"}
    call_state = {"n": 0}

    def fake_detect(frame, conf=0.25, iou=0.45):
        idx = call_state["n"]
        call_state["n"] += 1
        if idx < len(per_frame_gates):
            return per_frame_gates[idx]
        return []

    detector.detect_in_frame.side_effect = fake_detect
    return detector


class TestCourseGateCounter(unittest.TestCase):

    def test_spawns_new_track_for_unmatched_detection(self):
        """Gate at (100,200) in frame 0; gate at (500,600) in frames 1+2 -> count == 2."""
        per_frame = [
            [_gate(100, 200, 0.8)],                         # frame 0: gate A
            [_gate(100, 200, 0.8), _gate(500, 600, 0.75)],  # frame 1: gate A + gate B
            [_gate(100, 200, 0.8), _gate(500, 600, 0.75)],  # frame 2: gate A + gate B
            [_gate(100, 200, 0.8), _gate(500, 600, 0.75)],  # frame 3: gate A + gate B
        ]
        detector = _make_detector(per_frame)
        fake_cap = _make_fake_cap(len(per_frame))

        counter = CourseGateCounter(
            detector=detector,
            conf=0.20,
            stride=1,
            min_hits=2,
            track_missing_max=5,
        )

        with patch("cv2.VideoCapture", return_value=fake_cap):
            result = counter.count("fake.mp4", 1280, 720)

        self.assertEqual(result["course_gates_count"], 2)

    def test_filters_short_lived_noise_tracks(self):
        """5 frames: 1 stable gate + 1-frame noise, min_hits=3 -> count == 1."""
        per_frame = [
            [_gate(100, 200, 0.8)],
            [_gate(100, 200, 0.8), _gate(900, 100, 0.3)],  # noise in frame 1 only
            [_gate(100, 200, 0.8)],
            [_gate(100, 200, 0.8)],
            [_gate(100, 200, 0.8)],
        ]
        detector = _make_detector(per_frame)
        fake_cap = _make_fake_cap(len(per_frame))

        counter = CourseGateCounter(
            detector=detector,
            conf=0.20,
            stride=1,
            min_hits=3,
            track_missing_max=5,
        )

        with patch("cv2.VideoCapture", return_value=fake_cap):
            result = counter.count("fake.mp4", 1280, 720)

        self.assertEqual(result["course_gates_count"], 1)

    def test_merges_track_fragments_with_time_gap(self):
        """Gate in frames 0-4, gap 5-19, reappears 20-24; merge_gap_max=20 -> count==1."""
        per_frame = []
        for i in range(25):
            if i < 5 or i >= 20:
                per_frame.append([_gate(100, 200, 0.8)])
            else:
                per_frame.append([])  # gap
        detector = _make_detector(per_frame)
        fake_cap = _make_fake_cap(len(per_frame))

        counter = CourseGateCounter(
            detector=detector,
            conf=0.20,
            stride=1,
            min_hits=3,
            track_missing_max=3,  # forces track break during gap
            fragment_merge_gap_max=20,
            fragment_merge_dist_ratio=0.10,
        )

        with patch("cv2.VideoCapture", return_value=fake_cap):
            result = counter.count("fake.mp4", 1280, 720)

        self.assertEqual(result["course_gates_count"], 1)
        self.assertGreaterEqual(result["diagnostics"]["merged_pairs"], 1)

    def test_dedups_parallel_duplicate_tracks(self):
        """Two near-identical detections (dx=3px, dy=2px) in all frames -> count==1."""
        per_frame = []
        for _ in range(10):
            per_frame.append([
                _gate(100, 200, 0.8),
                _gate(103, 202, 0.75),
            ])
        detector = _make_detector(per_frame)
        fake_cap = _make_fake_cap(len(per_frame))

        counter = CourseGateCounter(
            detector=detector,
            conf=0.20,
            stride=1,
            min_hits=3,
            track_missing_max=5,
            match_thresh_min=2.0,  # force two separate tracks
            match_thresh_ratio=0.001,
            dedup_dx_ratio=0.03,
            dedup_dy_ratio=0.05,
            dedup_overlap_thresh=0.50,
        )

        with patch("cv2.VideoCapture", return_value=fake_cap):
            result = counter.count("fake.mp4", 1280, 720)

        self.assertEqual(result["course_gates_count"], 1)
        self.assertGreaterEqual(result["diagnostics"]["dedup_drops"], 1)

    def test_count_stable_under_stride_change(self):
        """20 frames, 3 stable gates; stride=1 vs stride=2 both return count==3."""
        base_gates = [_gate(100, 200, 0.8), _gate(400, 400, 0.75), _gate(700, 600, 0.85)]
        per_frame = [list(base_gates) for _ in range(20)]

        for test_stride in (1, 2):
            detector = _make_detector(list(per_frame))
            fake_cap = _make_fake_cap(len(per_frame))

            counter = CourseGateCounter(
                detector=detector,
                conf=0.20,
                stride=test_stride,
                min_hits=2,
                track_missing_max=5,
            )

            with patch("cv2.VideoCapture", return_value=fake_cap):
                result = counter.count("fake.mp4", 1280, 720)

            self.assertEqual(
                result["course_gates_count"], 3,
                f"stride={test_stride}: expected 3, got {result['course_gates_count']}"
            )


if __name__ == "__main__":
    unittest.main()
