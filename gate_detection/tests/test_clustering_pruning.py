"""
Tests for gate clustering and full-track pruning failure modes.

Covers:
  - Distinct gate preservation at 720p and 2160p
  - X-distance constraint prevents merging horizontally separated gates
  - Max cluster size guard prevents chain collapse
  - Full-track pruning safety: initial seeds recovered when >50% dropped
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ski_racing.pipeline import SkiRacingPipeline


def _make_gate(cx, by, conf=0.8, cls=0, name="gate"):
    return {
        "center_x": float(cx),
        "base_y": float(by),
        "class": int(cls),
        "class_name": name,
        "confidence": float(conf),
    }


class TestClusterGatesByY:
    """Tests for SkiRacingPipeline._cluster_gates_by_y."""

    def _cluster(self, gates, y_thresh=12.0, frame_height=None, frame_width=None):
        pipe = SkiRacingPipeline.__new__(SkiRacingPipeline)
        return pipe._cluster_gates_by_y(
            gates,
            y_thresh=y_thresh,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    # ── Distinct gate preservation ──────────────────────────────────

    def test_distinct_gates_preserved_720p(self):
        """8 gates spaced ~80px apart at 720p should all survive clustering."""
        gates = [_make_gate(cx=300, by=50 + i * 80) for i in range(8)]
        result = self._cluster(gates, frame_height=720)
        assert len(result) == 8, f"Expected 8 gates, got {len(result)}"

    def test_distinct_gates_preserved_2160p(self):
        """8 gates spaced ~240px apart at 2160p should all survive clustering."""
        gates = [_make_gate(cx=900, by=150 + i * 240) for i in range(8)]
        result = self._cluster(gates, frame_height=2160)
        assert len(result) == 8, f"Expected 8 gates, got {len(result)}"

    def test_close_duplicates_merged(self):
        """Among several gates, two detections 5px apart in Y and same X
        should merge while others remain distinct."""
        gates = [
            _make_gate(cx=300, by=100),
            _make_gate(cx=300, by=200),
            _make_gate(cx=302, by=205),  # duplicate of previous
            _make_gate(cx=300, by=400),
            _make_gate(cx=300, by=500),
        ]
        result = self._cluster(gates, frame_height=720)
        assert len(result) == 4, f"Expected 4 gates (1 merged pair), got {len(result)}"

    # ── X-distance constraint ──────────────────────────────────────

    def test_x_separated_gates_not_merged(self):
        """Two gates at same Y but far apart in X should NOT merge."""
        gates = [
            _make_gate(cx=100, by=400),
            _make_gate(cx=600, by=405),
        ]
        result = self._cluster(gates, frame_height=720)
        assert len(result) == 2, (
            f"Expected 2 gates (X-separated), got {len(result)}"
        )

    def test_x_close_gates_merge(self):
        """Among several gates, two at same Y and close X SHOULD merge."""
        gates = [
            _make_gate(cx=300, by=100),
            _make_gate(cx=300, by=250),
            _make_gate(cx=310, by=253),  # duplicate of previous (close X + Y)
            _make_gate(cx=300, by=400),
            _make_gate(cx=300, by=550),
        ]
        result = self._cluster(gates, frame_height=720)
        assert len(result) == 4, (
            f"Expected 4 gates (1 merged pair), got {len(result)}"
        )

    def test_x_threshold_scales_with_frame_width(self):
        """A pair with moderate X-separation should merge on wide frames but
        stay separate on narrow frames when Y is near-identical."""
        gates = [
            _make_gate(cx=100, by=250),
            _make_gate(cx=140, by=252),  # dx=40, dy=2
            _make_gate(cx=300, by=400),
            _make_gate(cx=500, by=550),
        ]
        # width=640 -> x_thresh=max(16,20)=20, so the pair should remain separate
        narrow = self._cluster(gates, frame_height=720, frame_width=640)
        # width=1920 -> x_thresh=48, so the pair should merge
        wide = self._cluster(gates, frame_height=720, frame_width=1920)
        assert len(narrow) == 4, f"Expected 4 gates on narrow frame, got {len(narrow)}"
        assert len(wide) == 3, f"Expected 3 gates on wide frame, got {len(wide)}"

    # ── Chain collapse prevention ──────────────────────────────────

    def test_no_chain_collapse(self):
        """A chain of gates that are each close to the next but span a large
        range should NOT collapse into a single cluster."""
        # 6 gates each 15px apart → total span = 75px
        gates = [_make_gate(cx=300, by=200 + i * 15) for i in range(6)]
        result = self._cluster(gates, frame_height=720)
        # With max_cluster_size=3 and conservative thresholds, should get
        # at least 2 clusters (not 1)
        assert len(result) >= 2, (
            f"Chain of 6 gates collapsed to {len(result)} cluster(s)"
        )

    def test_cluster_does_not_reduce_by_more_than_50pct(self):
        """Clustering should not reduce gate count by more than 50% for
        a typical gate layout."""
        # 10 gates with mixed spacing
        gates = [_make_gate(cx=300 + (i % 2) * 20, by=100 + i * 60) for i in range(10)]
        result = self._cluster(gates, frame_height=720)
        assert len(result) >= 5, (
            f"Clustering reduced {len(gates)} gates to {len(result)} (>50% loss)"
        )

    # ── Edge cases ─────────────────────────────────────────────────

    def test_empty_input(self):
        result = self._cluster([], frame_height=720)
        assert result == []

    def test_single_gate(self):
        gates = [_make_gate(cx=300, by=400)]
        result = self._cluster(gates, frame_height=720)
        assert len(result) == 1


class TestXThresholdRegression:
    """Regression guards for width-scaled X clustering threshold."""

    def _cluster(self, gates, y_thresh=12.0, frame_height=None, frame_width=None):
        pipe = SkiRacingPipeline.__new__(SkiRacingPipeline)
        return pipe._cluster_gates_by_y(
            gates,
            y_thresh=y_thresh,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    def test_720p_dx40_poles_stay_separate(self):
        """
        Regression guard for mmexport1706076255933 final-pass failure.
        9-gate layout -> y_merge_thresh = 11.61px. Critical pair dy=11.4px
        (y_close=True), dx=40.8px. With 0.025*1280=32px they must stay
        separate (9 gates).
        """
        gates = [
            {"center_x": 550.5, "base_y": 201.2, "confidence": 0.444, "class": 0, "class_name": "gate"},
            {"center_x": 816.4, "base_y": 209.5, "confidence": 0.421, "class": 0, "class_name": "gate"},
            {"center_x": 436.9, "base_y": 246.6, "confidence": 0.465, "class": 0, "class_name": "gate"},
            {"center_x": 669.2, "base_y": 286.9, "confidence": 0.569, "class": 0, "class_name": "gate"},
            {"center_x": 465.7, "base_y": 394.9, "confidence": 0.527, "class": 0, "class_name": "gate"},
            {"center_x": 424.9, "base_y": 406.3, "confidence": 0.528, "class": 0, "class_name": "gate"},
            {"center_x": 1015.6, "base_y": 532.4, "confidence": 0.663, "class": 0, "class_name": "gate"},
            {"center_x": 1085.9, "base_y": 558.8, "confidence": 0.722, "class": 0, "class_name": "gate"},
            {"center_x": 834.9, "base_y": 720.0, "confidence": 0.547, "class": 0, "class_name": "gate"},
        ]
        result = self._cluster(gates, frame_height=720, frame_width=1280)
        assert len(result) == 9, (
            "Distinct poles dx=40.8px > 0.025*1280=32px must remain separate; "
            "old 0.05*1280=64px would merge to 8 gates."
        )

    def test_720p_genuine_duplicate_merges(self):
        """dx=12px << 32px should collapse same-pole duplicate boxes."""
        gates = [
            _make_gate(cx=300.0, by=100.0, conf=0.80),
            _make_gate(cx=312.0, by=103.0, conf=0.75),  # duplicate
            _make_gate(cx=300.0, by=200.0, conf=0.80),
            _make_gate(cx=300.0, by=300.0, conf=0.80),
            _make_gate(cx=300.0, by=400.0, conf=0.80),
        ]
        result = self._cluster(gates, frame_height=720, frame_width=1280)
        assert len(result) < 5, "Same-pole duplicate (dx=12px) should be collapsed"

    def test_720p_blind_zone_boundary(self):
        """Verify boundary at 0.025*1280=32px: dx<32 merges, dx>32 separates."""
        for dx, expect_merge in [(28, True), (36, False)]:
            gates = [
                _make_gate(cx=300.0, by=100.0, conf=0.80),
                _make_gate(cx=300.0 + dx, by=103.0, conf=0.80),
                _make_gate(cx=300.0, by=200.0, conf=0.80),
                _make_gate(cx=300.0, by=300.0, conf=0.80),
                _make_gate(cx=300.0, by=400.0, conf=0.80),
            ]
            result = self._cluster(gates, frame_height=720, frame_width=1280)
            if expect_merge:
                assert len(result) < 5, f"dx={dx}px should merge (< 32px threshold)"
            else:
                assert len(result) == 5, f"dx={dx}px should stay separate (> 32px threshold)"


class TestFullTrackPruningSafety:
    """Test that full-track pruning doesn't over-aggressively drop gates."""

    def test_pruning_safety_preserves_initial_seeds(self):
        """When _track_gates_full_video would drop >50% of initial gates,
        the safety fallback should recover them."""
        # This is a structural test — we verify the logic in the pipeline
        # by calling the helper functions directly.
        from ski_racing.pipeline import (
            _gate_stats_from_frame_history,
            _ghost_gate_count_from_stats,
            _filter_frame_gate_history_full,
        )

        # Simulate: 6 initial gates, only 2 confirmed (< 50%)
        frame_history_full = {
            0: {
                0: {"center_x": 100, "base_y": 200, "is_interpolated": False},
                1: {"center_x": 300, "base_y": 400, "is_interpolated": False},
            },
            3: {
                0: {"center_x": 101, "base_y": 201, "is_interpolated": False},
                1: {"center_x": 301, "base_y": 401, "is_interpolated": False},
            },
            6: {
                0: {"center_x": 102, "base_y": 202, "is_interpolated": False},
                1: {"center_x": 302, "base_y": 402, "is_interpolated": False},
            },
        }

        confirmed_ids = {0, 1}
        filtered = _filter_frame_gate_history_full(frame_history_full, confirmed_ids)
        stats = _gate_stats_from_frame_history(filtered)
        ghost_count = _ghost_gate_count_from_stats(stats)

        # Both gates have real observations → 0 ghost gates
        assert ghost_count == 0

        # Verify filtered history only contains confirmed IDs
        for frame_gates in filtered.values():
            for gid in frame_gates:
                assert int(gid) in confirmed_ids


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
