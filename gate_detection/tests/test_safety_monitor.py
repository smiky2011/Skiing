from pathlib import Path
import sys


TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from ski_racing.safety import SafetyMonitor


def _run_sequence(delta2_values, alpha_values):
    monitor = SafetyMonitor(eis_threshold=50.0, stability_window=0)
    outputs = []
    for frame_idx, (delta2_eis, alpha_t) in enumerate(zip(delta2_values, alpha_values)):
        bev_frame = {
            "frame_idx": frame_idx,
            "delta2_eis": float(delta2_eis),
            "alpha_t": float(alpha_t),
        }
        detection_frame = {"frame_idx": frame_idx, "detections": []}
        outputs.append(monitor.update(bev_frame, detection_frame))
    monitor.flush()
    return outputs


def test_eis_two_frame_spike_triggers_degraded_only_on_spike_frames():
    delta2 = [0, 0, 100, 100, 0]
    alpha = [0.5] * len(delta2)
    outputs = _run_sequence(delta2, alpha)
    degraded = [frame["DEGRADED"] for frame in outputs]
    assert degraded == [False, False, True, True, False]


def test_eis_five_frame_elevation_is_classified_as_pan_and_suppressed():
    delta2 = [0, 100, 100, 100, 100, 100, 0]
    alpha = [0.5] * len(delta2)
    outputs = _run_sequence(delta2, alpha)
    assert all(not frame["DEGRADED"] for frame in outputs)


def test_vp_collapse_alpha_zero_triggers_degraded_on_all_frames():
    delta2 = [0, 0, 0]
    alpha = [0.0, 0.0, 0.0]
    outputs = _run_sequence(delta2, alpha)
    assert all(frame["DEGRADED"] for frame in outputs)
    assert all(frame["degraded_reason"] == "vp_collapse" for frame in outputs)


def test_score_collapse_triggers_system_uninitialized():
    monitor = SafetyMonitor(eis_threshold=50.0, stability_window=0)
    bev_frame = {
        "frame_idx": 0,
        "delta2_eis": 0.0,
        "alpha_t": 1.0,
        "decoder_score": 0.20,  # synthetic S* below tau_seq
        "tau_seq": 0.30,
        "system_initialized": False,
    }
    detection_frame = {"frame_idx": 0, "detections": []}

    output = monitor.update(bev_frame, detection_frame)

    assert bev_frame["decoder_score"] < bev_frame["tau_seq"]
    assert output["SYSTEM_UNINITIALIZED"] is True


def test_s_star_collapse_triggers_low_confidence():
    """Wave 4: S* below confidence_floor triggers LOW_CONFIDENCE=True."""
    monitor = SafetyMonitor(eis_threshold=50.0, stability_window=0, confidence_floor=-2.0)
    decoder_frame = {
        "frame_idx": 0,
        "state": "R",
        "score_valid": True,
        "s_star": -3.5,
        "s_star_margin": 0.2,
    }

    result = monitor.update_with_decoder(decoder_frame)

    assert result["LOW_CONFIDENCE"] is True
    assert result["degraded_reason"] == "s_star_collapse"


def test_s_star_above_floor_no_flag():
    """Wave 4: S* above confidence_floor does not trigger flag."""
    monitor = SafetyMonitor(eis_threshold=50.0, stability_window=0, confidence_floor=-2.0)
    decoder_frame = {
        "frame_idx": 1,
        "state": "R",
        "score_valid": True,
        "s_star": -1.0,
        "s_star_margin": 0.8,
    }

    result = monitor.update_with_decoder(decoder_frame)

    assert result["LOW_CONFIDENCE"] is False
    assert result["degraded_reason"] is None


def test_score_not_valid_skips_collapse_check():
    """Wave 4: When score_valid=False, collapse check is skipped."""
    monitor = SafetyMonitor(eis_threshold=50.0, stability_window=0, confidence_floor=-2.0)
    decoder_frame = {
        "frame_idx": 2,
        "state": "R",
        "score_valid": False,
        "s_star": -99.0,
        "s_star_margin": 0.0,
    }

    result = monitor.update_with_decoder(decoder_frame)

    assert result["LOW_CONFIDENCE"] is False
    assert result["degraded_reason"] is None
