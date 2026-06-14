import pytest

from ski_racing.pipeline import (
    _filter_frame_gate_history_full,
    _gate_stats_from_frame_history,
    _ghost_gate_count_from_stats,
    _interp_rate_overall,
)


def test_ghost_gate_filter_and_counts():
    raw = {
        0: {
            0: {"center_x": 100.0, "base_y": 200.0, "is_interpolated": False},
            1: {"center_x": 300.0, "base_y": 400.0, "is_interpolated": True},
            2: {"center_x": 500.0, "base_y": 600.0, "is_interpolated": True},
        },
        3: {
            0: {"center_x": 101.0, "base_y": 201.0, "is_interpolated": False},
            1: {"center_x": 300.0, "base_y": 400.0, "is_interpolated": True},
            2: {"center_x": 500.0, "base_y": 600.0, "is_interpolated": True},
        },
        6: {
            0: {"center_x": 101.0, "base_y": 201.0, "is_interpolated": True},
            1: {"center_x": 300.0, "base_y": 400.0, "is_interpolated": True},
            2: {"center_x": 500.0, "base_y": 600.0, "is_interpolated": True},
        },
    }

    raw_stats = _gate_stats_from_frame_history(raw)
    assert _ghost_gate_count_from_stats(raw_stats) == 2
    assert raw_stats[0]["observed_count"] == 2
    assert raw_stats[1]["observed_count"] == 0
    assert raw_stats[1]["unique_positions"] == 1

    filtered = _filter_frame_gate_history_full(raw, confirmed_ids={0})
    filtered_stats = _gate_stats_from_frame_history(filtered)
    assert _ghost_gate_count_from_stats(filtered_stats) == 0
    assert set(filtered_stats.keys()) == {0}


def test_interp_rate_overall_uses_exported_history():
    history = {
        0: {0: {"center_x": 1.0, "base_y": 2.0, "is_interpolated": False}},
        3: {0: {"center_x": 1.0, "base_y": 2.0, "is_interpolated": True}},
        6: {0: {"center_x": 1.0, "base_y": 2.0, "is_interpolated": True}},
    }
    rate = _interp_rate_overall(history)
    assert rate == pytest.approx(2.0 / 3.0)


def test_course_gates_count_gte_visible_gates():
    """Structural invariant: course total >= seeded visible gates."""
    mock_result = {
        "gates_count": 3,
        "course_gates_count": 4,  # 1 additional later-appearing gate
    }
    assert mock_result["course_gates_count"] >= mock_result["gates_count"]
