"""
Tests for ski_racing/visualize.py

Focuses on create_summary_figure robustness against the 2D-first sprint
JSON schema where trajectory_3d and physics_validation are the sentinel
string "disabled" rather than data structures.

All tests write real temp JSON files because create_summary_figure takes
a file path, not a dict.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ski_racing.visualize import create_summary_figure  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal valid analysis payload (2D-only, as produced by current pipeline)
# ---------------------------------------------------------------------------

def _base_analysis(**overrides):
    """Return a minimal analysis dict suitable for writing to JSON."""
    payload = {
        "video": "/fake/race.mp4",
        "video_info": {"fps": 25.0, "width": 1920, "height": 1080, "total_frames": 500},
        "discipline": "slalom",
        "discipline_source": "auto_detected",
        "discipline_detection": None,
        "gate_conf": 0.25,
        "gate_iou": 0.45,
        "stabilized": False,
        "gates": [
            {"center_x": 400.0, "base_y": 300.0, "class": 0, "class_name": "gate", "confidence": 0.8},
            {"center_x": 600.0, "base_y": 500.0, "class": 1, "class_name": "gate", "confidence": 0.75},
        ],
        "gates_count": 2,
        "trajectory_2d": [
            {"frame": 0, "x": 500.0, "y": 200.0},
            {"frame": 1, "x": 510.0, "y": 220.0},
            {"frame": 2, "x": 520.0, "y": 240.0},
        ],
        "trajectory_3d": "disabled",
        "physics_validation": "disabled",
        "timestamp": "2026-02-25T00:00:00",
        "outlier_count": 0,
        "outlier_frames": [],
        "bytetrack_coverage": 0.9,
        "track_id_switches": 0,
        "tracking_diagnostics": {},
        "runtime_profile_sec": {},
        "pipeline_params": {
            "trajectory_3d": "disabled",
            "physics_validation": "disabled",
        },
    }
    payload.update(overrides)
    return payload


def _write_and_summarise(analysis_dict, tmp_path):
    """Write analysis dict to a temp JSON, call create_summary_figure, return PNG path."""
    json_path = tmp_path / "analysis.json"
    json_path.write_text(json.dumps(analysis_dict), encoding="utf-8")
    png_path = tmp_path / "summary.png"
    create_summary_figure(str(json_path), str(png_path))
    return png_path


# ---------------------------------------------------------------------------
# Tests: sentinel / missing trajectory_3d variants
# ---------------------------------------------------------------------------

class TestCreateSummaryFigureDisabled3D:
    """create_summary_figure must not raise when trajectory_3d is disabled."""

    def test_sentinel_string_disabled(self, tmp_path):
        """Standard 2D-first pipeline output: trajectory_3d = 'disabled'."""
        png = _write_and_summarise(_base_analysis(trajectory_3d="disabled"), tmp_path)
        assert png.exists(), "PNG must be written"

    def test_key_absent(self, tmp_path):
        """Legacy JSON produced before sentinel was introduced."""
        analysis = _base_analysis()
        del analysis["trajectory_3d"]
        png = _write_and_summarise(analysis, tmp_path)
        assert png.exists()

    def test_empty_list(self, tmp_path):
        """Explicitly empty 3D list — treated as disabled."""
        png = _write_and_summarise(_base_analysis(trajectory_3d=[]), tmp_path)
        assert png.exists()

    def test_none_value(self, tmp_path):
        """Null in JSON — treated as disabled."""
        png = _write_and_summarise(_base_analysis(trajectory_3d=None), tmp_path)
        assert png.exists()

    def test_physics_sentinel_string(self, tmp_path):
        """physics_validation = 'disabled' must not crash stats panel."""
        png = _write_and_summarise(
            _base_analysis(trajectory_3d="disabled", physics_validation="disabled"),
            tmp_path,
        )
        assert png.exists()

    def test_physics_absent(self, tmp_path):
        """physics_validation key absent — treated as no metrics."""
        analysis = _base_analysis()
        del analysis["physics_validation"]
        png = _write_and_summarise(analysis, tmp_path)
        assert png.exists()


class TestCreateSummaryFigureEnabled3D:
    """When trajectory_3d is a real list, Panel 2 renders data (not placeholder)."""

    def test_valid_3d_trajectory(self, tmp_path, monkeypatch):
        """PNG is written and Panel 2 is not the '3D disabled' placeholder."""
        from matplotlib.axes._axes import Axes

        seen_text = []
        original_text = Axes.text

        def _spy_text(self, x, y, s, *args, **kwargs):
            if isinstance(s, str):
                seen_text.append(s)
            return original_text(self, x, y, s, *args, **kwargs)

        monkeypatch.setattr(Axes, "text", _spy_text, raising=True)

        traj_3d = [
            {"frame": 0, "x": 0.0, "y": 0.0},
            {"frame": 1, "x": 1.0, "y": 0.5},
            {"frame": 2, "x": 2.0, "y": 1.0},
        ]
        png = _write_and_summarise(_base_analysis(trajectory_3d=traj_3d), tmp_path)
        assert png.exists()
        assert "3D disabled\n(2D-first mode)" not in seen_text

    def test_valid_3d_with_physics(self, tmp_path):
        """PNG is written when both 3D trajectory and physics metrics are present."""
        traj_3d = [
            {"frame": 0, "x": 0.0, "y": 0.0},
            {"frame": 1, "x": 1.0, "y": 0.5},
            {"frame": 2, "x": 2.0, "y": 1.0},
        ]
        physics = {
            "valid": True,
            "issues": [],
            "metrics": {
                "total_distance_m": 10.0,
                "duration_s": 5.0,
                "speeds_kmh": {"mean": 40.0, "max": 65.0, "p90": 60.0},
                "g_forces": {"max": 2.5},
                "turn_radii_m": {"min": 8.0},
                "smoothness": {"max_jump_m": 0.1},
            },
        }
        png = _write_and_summarise(
            _base_analysis(trajectory_3d=traj_3d, physics_validation=physics),
            tmp_path,
        )
        assert png.exists()
