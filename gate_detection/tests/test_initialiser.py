from pathlib import Path
from unittest.mock import Mock
import json
import sys

import pytest


TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from ski_racing.initialiser import SequenceInitialiser


def _det(
    det_id: str,
    *,
    gate_id: str = "gate_1",
    bev_x: float = 0.0,
    bev_y: float = 0.0,
    log_red: float = -0.01,
    log_blue: float = -2.0,
    log_dnf: float = -3.0,
) -> dict:
    return {
        "detection_id": det_id,
        "gate_id": gate_id,
        "class_label": "red",
        "bbox_xyxy": [0.0, 0.0, 2.0, 4.0],
        "bev_x": float(bev_x),
        "bev_y": float(bev_y),
        "emission_log_prob": {
            "log_prob_red": float(log_red),
            "log_prob_blue": float(log_blue),
            "log_prob_dnf": float(log_dnf),
        },
    }


def test_early_gates_occluded_initialises_and_retroactively_assigns(tmp_path: Path):
    init = SequenceInitialiser(
        clip_id="occluded_clip",
        outputs_dir=tmp_path,
        t_min=5,
        n_persist=3,
        tau_seq=-1.5,
    )

    result = None
    for frame_idx in range(5):
        detections = [] if frame_idx < 4 else [_det("occluded_00004_000", gate_id="gate_5", bev_y=10.0)]
        decoder_output = {
            "gates_confirmed": frame_idx + 1,
            "score_valid": frame_idx == 4,
            "s_star": -1.12 if frame_idx == 4 else None,
            "topology_ok": True,
            "persistence_ok": True,
        }
        result = init.update(
            frame_idx=frame_idx,
            detections=detections,
            bev_positions=[],
            delta_t_s=1.0 / 30.0,
            decoder_output=decoder_output,
        )

    assert result is not None
    assert result["triggered"] is True
    assert result["event"]["event"] == "INITIALIZED"
    assert result["event"]["buffer_depth_at_trigger"] == 5

    retro_frames = [f["frame_idx"] for f in result["retroactive_assignments"]]
    assert retro_frames == [0, 1, 2, 3, 4]

    payload = json.loads((tmp_path / "occluded_clip_init.json").read_text(encoding="utf-8"))
    assert payload["events"][-1]["event"] == "INITIALIZED"


def test_no_chain_in_90_frames_resets_and_emits_system_uninitialized(tmp_path: Path):
    init = SequenceInitialiser(
        clip_id="no_chain_clip",
        outputs_dir=tmp_path,
        max_buffer_depth=90,
        t_min=5,
    )

    result = None
    for frame_idx in range(90):
        detections = [
            _det(
                f"no_chain_{frame_idx:05d}_000",
                gate_id=f"rand_{frame_idx}",
                bev_x=float((-1) ** frame_idx * 100.0),
                bev_y=float(frame_idx % 2),
            )
        ]
        result = init.update(
            frame_idx=frame_idx,
            detections=detections,
            bev_positions=[],
            delta_t_s=1.0 / 30.0,
            decoder_output={
                "gates_confirmed": 1,
                "score_valid": False,
                "s_star": None,
                "topology_ok": False,
                "persistence_ok": False,
            },
        )

    assert result is not None
    assert result["reset"] is True
    assert result["event"]["event"] == "RESET"
    assert init.buffer_depth == 0
    assert init.system_uninitialized is True
    assert init.flag_events[-1]["SYSTEM_UNINITIALIZED"] is True


def test_retro_decode_does_not_recall_detector(tmp_path: Path):
    detector_mock = Mock()
    init = SequenceInitialiser(
        clip_id="no_detector_recall_clip",
        outputs_dir=tmp_path,
        t_min=5,
        detector_callback=detector_mock,
    )

    before_trigger_calls = 0
    result = None
    for frame_idx in range(5):
        result = init.update(
            frame_idx=frame_idx,
            detections=[_det(f"nd_{frame_idx:05d}_000", bev_x=float(frame_idx), bev_y=float(frame_idx))],
            bev_positions=[],
            delta_t_s=1.0,
            decoder_output={
                "gates_confirmed": frame_idx + 1,
                "score_valid": frame_idx == 4,
                "s_star": -1.0 if frame_idx == 4 else None,
                "topology_ok": True,
                "persistence_ok": True,
            },
        )
        if frame_idx == 3:
            before_trigger_calls = detector_mock.call_count

    assert result is not None and result["triggered"] is True
    assert detector_mock.call_count == before_trigger_calls == 0


class _FakeTracker:
    def __init__(self) -> None:
        self.registered = []

    def register_track(self, **kwargs):
        self.registered.append(kwargs)
        return True


def test_velocity_seeding_within_20_percent_of_expected(tmp_path: Path):
    tracker = _FakeTracker()
    init = SequenceInitialiser(
        clip_id="velocity_seed_clip",
        outputs_dir=tmp_path,
        tracker=tracker,
        t_min=5,
        n_persist=3,
    )

    result = None
    for frame_idx in range(5):
        result = init.update(
            frame_idx=frame_idx,
            detections=[
                _det(
                    f"vel_{frame_idx:05d}_000",
                    gate_id="gate_velocity",
                    bev_x=5.0 * frame_idx,
                    bev_y=0.0,
                )
            ],
            bev_positions=[],
            delta_t_s=1.0,
            decoder_output={
                "gates_confirmed": frame_idx + 1,
                "score_valid": frame_idx == 4,
                "s_star": -1.0 if frame_idx == 4 else None,
                "topology_ok": True,
            },
        )

    assert result is not None and result["triggered"] is True
    assert len(tracker.registered) == 1
    state_vector = tracker.registered[0]["state_vector"]
    vx = state_vector[2]
    assert vx == pytest.approx(5.0, rel=0.2)

