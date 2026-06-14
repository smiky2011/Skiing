import unittest
from unittest.mock import MagicMock, patch

from ski_racing.detection import GateDetector


def _gate(center_x, base_y, confidence=0.7, cls=0, name="gate"):
    return {
        "class": int(cls),
        "class_name": str(name),
        "center_x": float(center_x),
        "center_y": float(base_y - 40.0),
        "base_y": float(base_y),
        "bbox": [float(center_x - 15), float(base_y - 80), float(center_x + 15), float(base_y)],
        "confidence": float(confidence),
    }


class TestDetectFromConsensus(unittest.TestCase):
    def _run_consensus(self, per_frame_gates, **kwargs):
        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        fake_frames = [(True, MagicMock(shape=(720, 1280, 3))) for _ in per_frame_gates]
        fake_frames.append((False, None))
        fake_cap.read.side_effect = fake_frames

        detector = GateDetector.__new__(GateDetector)
        detector.model = MagicMock()

        call_count = {"n": 0}

        def fake_detect(_frame, **_kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return per_frame_gates[idx]

        with patch("cv2.VideoCapture", return_value=fake_cap):
            with patch.object(detector, "detect_in_frame", side_effect=fake_detect):
                return detector.detect_from_consensus(
                    "fake_video.mp4",
                    conf=kwargs.get("conf", 0.35),
                    iou=kwargs.get("iou", 0.55),
                    max_frames=len(per_frame_gates),
                    stride=1,
                    min_support=kwargs.get("min_support", 2),
                    frame_height=720,
                )

    def test_consensus_stable_under_frame_permutation(self):
        frame0 = [_gate(100, 200, 0.8), _gate(500, 600, 0.78)]
        frame1 = [_gate(102, 201, 0.82), _gate(498, 599, 0.80)]
        frame2 = [_gate(101, 199, 0.79), _gate(502, 601, 0.77)]

        result_a = self._run_consensus([frame0, frame1, frame2], min_support=2)
        result_b = self._run_consensus([frame2, frame0, frame1], min_support=2)

        def as_tuple_list(gates):
            return [
                (
                    round(float(g["center_x"]), 3),
                    round(float(g["base_y"]), 3),
                    round(float(g["confidence"]), 3),
                )
                for g in gates
            ]

        self.assertEqual(as_tuple_list(result_a), as_tuple_list(result_b))
        self.assertEqual(len(result_a), 2)

    def test_consensus_falls_back_when_support_too_low(self):
        frame0 = [_gate(100, 200, 0.9)]
        frame1 = [_gate(500, 600, 0.9)]
        frame2 = [_gate(900, 300, 0.9)]

        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        fake_frames = [(True, MagicMock(shape=(720, 1280, 3))) for _ in (frame0, frame1, frame2)]
        fake_frames.append((False, None))
        fake_cap.read.side_effect = fake_frames

        fallback = [_gate(222, 333, 0.66)]
        detector = GateDetector.__new__(GateDetector)
        detector.model = MagicMock()
        detector.detect_from_best_frame = MagicMock(return_value=fallback)

        call_count = {"n": 0}

        def fake_detect(_frame, **_kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return [frame0, frame1, frame2][idx]

        with patch("cv2.VideoCapture", return_value=fake_cap):
            with patch.object(detector, "detect_in_frame", side_effect=fake_detect):
                result = detector.detect_from_consensus(
                    "fake_video.mp4",
                    conf=0.35,
                    iou=0.55,
                    max_frames=3,
                    stride=1,
                    min_support=2,
                    frame_height=720,
                )

        detector.detect_from_best_frame.assert_called_once()
        self.assertEqual(result, fallback)

    def test_consensus_filters_low_conf_clusters(self):
        frame0 = [_gate(100, 200, 0.8), _gate(500, 600, 0.2)]
        frame1 = [_gate(102, 201, 0.81), _gate(498, 601, 0.22)]
        frame2 = [_gate(99, 199, 0.79), _gate(502, 599, 0.24)]

        fake_cap = MagicMock()
        fake_cap.isOpened.return_value = True
        fake_frames = [(True, MagicMock(shape=(720, 1280, 3))) for _ in (frame0, frame1, frame2)]
        fake_frames.append((False, None))
        fake_cap.read.side_effect = fake_frames

        detector = GateDetector.__new__(GateDetector)
        detector.model = MagicMock()
        fallback = [_gate(333, 444, 0.77)]
        detector.detect_from_best_frame = MagicMock(return_value=fallback)

        call_count = {"n": 0}

        def fake_detect(_frame, **_kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return [frame0, frame1, frame2][idx]

        with patch("cv2.VideoCapture", return_value=fake_cap):
            with patch.object(detector, "detect_in_frame", side_effect=fake_detect):
                result = detector.detect_from_consensus(
                    "fake_video.mp4",
                    conf=0.35,
                    iou=0.55,
                    max_frames=3,
                    stride=1,
                    min_support=2,
                    frame_height=720,
                )

        detector.detect_from_best_frame.assert_called_once()
        self.assertEqual(result, fallback)


if __name__ == "__main__":
    unittest.main()
