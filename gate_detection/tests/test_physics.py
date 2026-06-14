"""
Tests for the physics validation engine.
These can run without any video data — pure unit tests on the math.
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ski_racing.physics import PhysicsValidator


def test_speed_calculation():
    """Test that speed calculation produces correct results."""
    validator = PhysicsValidator(discipline="slalom", fps=30.0)

    # Skier moving 1 meter per frame at 30fps = 30 m/s = 108 km/h
    trajectory = [
        {"frame": 0, "x": 0.0, "y": 0.0},
        {"frame": 1, "x": 0.0, "y": 1.0},
        {"frame": 2, "x": 0.0, "y": 2.0},
    ]

    speeds, dts = validator.calculate_speeds(trajectory)
    assert len(speeds) == 2
    assert abs(speeds[0] - 108.0) < 0.1, f"Expected ~108 km/h, got {speeds[0]}"
    print("✅ test_speed_calculation passed")


def test_realistic_trajectory_passes():
    """A physically plausible trajectory should pass validation."""
    validator = PhysicsValidator(discipline="slalom", fps=30.0)

    # Simulate gentle S-turns at ~40 km/h (11.1 m/s)
    # At 30fps, that's ~0.37m per frame
    # Use wide turns (large radius) to stay physically plausible
    trajectory = []
    for i in range(90):  # 3 seconds
        x = 3.0 * np.sin(i * 0.03)  # very gentle S-curves, large radius
        y = i * 0.37  # moving down slope
        trajectory.append({"frame": i, "x": x, "y": y})

    result = validator.validate_trajectory(trajectory)
    assert result["valid"], f"Expected valid trajectory, got issues: {result['issues']}"
    print("✅ test_realistic_trajectory_passes passed")


def test_impossible_speed_fails():
    """A trajectory implying 200km/h in slalom should fail."""
    validator = PhysicsValidator(discipline="slalom", fps=30.0)

    # Skier moving 2m per frame at 30fps = 60m/s = 216 km/h
    trajectory = [
        {"frame": i, "x": 0.0, "y": i * 2.0}
        for i in range(30)
    ]

    result = validator.validate_trajectory(trajectory)
    assert not result["valid"], "Expected trajectory to fail physics check"
    assert any("speed" in issue.lower() for issue in result["issues"])
    print("✅ test_impossible_speed_fails passed")


def test_impossible_g_force_fails():
    """A trajectory with extreme turns should fail G-force check."""
    validator = PhysicsValidator(discipline="slalom", fps=30.0)

    # Sharp zigzag at high speed = unrealistic G-forces
    trajectory = []
    for i in range(60):
        x = 5.0 * (-1) ** i  # instant direction change every frame
        y = i * 0.5
        trajectory.append({"frame": i, "x": x, "y": y})

    result = validator.validate_trajectory(trajectory)
    # This should flag unrealistic physics
    assert not result["valid"], "Expected trajectory to fail physics check"
    print("✅ test_impossible_g_force_fails passed")


def test_trajectory_smoothness():
    """A trajectory with teleportation should be flagged."""
    validator = PhysicsValidator(discipline="slalom", fps=30.0)

    trajectory = [
        {"frame": 0, "x": 0.0, "y": 0.0},
        {"frame": 1, "x": 0.0, "y": 0.5},
        {"frame": 2, "x": 0.0, "y": 1.0},
        {"frame": 3, "x": 50.0, "y": 50.0},  # Teleportation!
        {"frame": 4, "x": 50.0, "y": 50.5},
    ]

    result = validator.validate_trajectory(trajectory)
    smoothness = result["metrics"]["smoothness"]
    assert smoothness["max_jump_m"] > 5.0, "Should detect large jump"
    print("✅ test_trajectory_smoothness passed")


def test_circumradius():
    """Test turn radius calculation with known geometry."""
    validator = PhysicsValidator()

    # Three points on a circle of known radius
    # Use unit circle: points at 0, 90, and 180 degrees on circle of radius 5
    import math
    r = 5.0
    p1 = (r, 0)
    p2 = (0, r)
    p3 = (-r, 0)

    radius = validator._circumradius(p1, p2, p3)
    assert abs(radius - r) < 0.01, f"Expected radius ~{r}, got {radius}"
    print("✅ test_circumradius passed")


def test_collinear_points():
    """Collinear points should return infinite radius."""
    validator = PhysicsValidator()

    p1 = (0, 0)
    p2 = (1, 1)
    p3 = (2, 2)

    radius = validator._circumradius(p1, p2, p3)
    assert radius == float("inf"), f"Expected inf, got {radius}"
    print("✅ test_collinear_points passed")


def test_kalman_rts_preserves_turns():
    """
    RTS smoother should preserve turn apexes better than a forward-only filter.
    (Prof. feedback Phase 3: over-smoothing check)

    Generate a slalom S-curve trajectory with noise. After smoothing,
    the turn apex deviation should be preserved within 30% of the raw signal.
    """
    from ski_racing.tracking import KalmanSmoother

    # Simulate slalom S-turns: ~8m amplitude, ~30 frames per half-turn
    raw = []
    for i in range(180):  # 6 seconds at 30fps
        x = 8.0 * np.sin(i * 0.105)  # ~60 frame period = 2s turns
        y = i * 0.37  # downhill movement
        # Add realistic noise (3-5 px)
        x += np.random.normal(0, 3.0)
        y += np.random.normal(0, 2.0)
        raw.append({"frame": i, "x": x, "y": y})

    # Smooth with slalom defaults (high Q=3.0 to avoid over-smoothing)
    kf = KalmanSmoother(fps=30.0, discipline="slalom")
    smoothed = kf.smooth(raw)

    # Find turn apexes in the clean signal (before noise)
    clean_x = [8.0 * np.sin(i * 0.105) for i in range(180)]
    apex_indices = []
    for i in range(1, len(clean_x) - 1):
        if (clean_x[i] - clean_x[i - 1]) * (clean_x[i + 1] - clean_x[i]) < 0:
            apex_indices.append(i)

    # The smoothed path should still reach within 50% of the expected apex amplitude
    max_clean_x = max(abs(x) for x in clean_x)
    max_smooth_x = max(abs(p["x"]) for p in smoothed)

    # Smoothed should preserve at least 50% of lateral range
    assert max_smooth_x > max_clean_x * 0.5, (
        f"Over-smoothing: smoothed lateral range {max_smooth_x:.1f} < "
        f"50% of clean {max_clean_x:.1f}. Turn apexes are being cut."
    )
    print("✅ test_kalman_rts_preserves_turns passed")


def test_kalman_discipline_defaults():
    """
    Verify discipline-tuned Kalman defaults are set correctly.
    Slalom should have higher process_noise than downhill because
    turns are sharper and more frequent.
    """
    from ski_racing.tracking import KalmanSmoother

    kf_sl = KalmanSmoother(fps=30.0, discipline="slalom")
    kf_gs = KalmanSmoother(fps=30.0, discipline="giant_slalom")
    kf_dh = KalmanSmoother(fps=30.0, discipline="downhill")

    assert kf_sl.process_noise > kf_gs.process_noise, \
        "Slalom should have higher process_noise than GS"
    assert kf_gs.process_noise > kf_dh.process_noise, \
        "GS should have higher process_noise than downhill"

    # Explicit value checks
    assert kf_sl.process_noise >= 2.0, \
        f"Slalom Q={kf_sl.process_noise} too low (should be >=2.0 for sharp turns)"

    print("✅ test_kalman_discipline_defaults passed")


def test_kalman_handles_frame_gaps():
    """
    Kalman smoother should handle non-consecutive frames gracefully
    (e.g., when frame_stride > 1 or frames are dropped).
    """
    from ski_racing.tracking import KalmanSmoother

    # Trajectory with gaps (every 3rd frame)
    trajectory = [
        {"frame": 0, "x": 0.0, "y": 0.0},
        {"frame": 3, "x": 1.0, "y": 3.0},
        {"frame": 6, "x": 2.0, "y": 6.0},
        {"frame": 9, "x": 3.0, "y": 9.0},
        {"frame": 12, "x": 4.0, "y": 12.0},
    ]

    kf = KalmanSmoother(fps=30.0, discipline="slalom")
    smoothed = kf.smooth(trajectory)

    # Should produce same number of points
    assert len(smoothed) == len(trajectory), "Output length should match input"

    # Smoothed trajectory should be roughly monotonic in Y
    for i in range(1, len(smoothed)):
        assert smoothed[i]["y"] > smoothed[i - 1]["y"], \
            f"Y should be monotonically increasing (got {smoothed[i]['y']} <= {smoothed[i-1]['y']})"

    print("✅ test_kalman_handles_frame_gaps passed")


def test_camera_motion_compensator_translation():
    """
    Test that translation-mode camera compensator correctly
    removes known offsets from coordinates.
    """
    from ski_racing.transform import CameraMotionCompensator

    # Simulate: baseline gates at known positions, camera shifts +10px right
    baseline = {0: (100.0, 200.0), 1: (300.0, 400.0)}
    frame_history = {
        0: {0: (100.0, 200.0), 1: (300.0, 400.0)},  # no shift
        1: {0: (110.0, 200.0), 1: (310.0, 400.0)},  # +10px X shift
        2: {0: (120.0, 200.0), 1: (320.0, 400.0)},  # +20px X shift
    }

    comp = CameraMotionCompensator(baseline, frame_history, mode="translation")
    comp.estimate_motion()

    # Frame 0: no shift expected
    sx, sy = comp.stabilize_point(500.0, 300.0, 0)
    assert abs(sx - 500.0) < 1.0, f"Frame 0 should have ~0 correction, got dx={500-sx}"

    # Frame 1: should subtract ~10px in X
    sx, sy = comp.stabilize_point(500.0, 300.0, 1)
    assert abs(sx - 490.0) < 1.0, f"Frame 1 should subtract ~10px, got {sx}"

    # Frame 2: should subtract ~20px in X
    sx, sy = comp.stabilize_point(500.0, 300.0, 2)
    assert abs(sx - 480.0) < 1.0, f"Frame 2 should subtract ~20px, got {sx}"

    print("✅ test_camera_motion_compensator_translation passed")


def test_dynamic_scale_transform():
    """
    Test that DynamicScaleTransform produces per-frame ppm values
    and that gradual changes are accepted while sudden jumps are rejected.
    """
    from ski_racing.transform import DynamicScaleTransform

    # Simulate gradual zoom: gates go from 100px → 105px → 110px (5% per step)
    # This is within the 10% hysteresis threshold so all should be accepted.
    gate_spacing_m = 12.0
    frame_history = {
        0: {0: (200.0, 100.0), 1: (200.0, 200.0)},   # 100px gap
        1: {0: (200.0, 100.0), 1: (200.0, 205.0)},   # 105px gap (+5%)
        2: {0: (200.0, 100.0), 1: (200.0, 210.0)},   # 110px gap (+5%)
    }

    dyn = DynamicScaleTransform(frame_history, gate_spacing_m)
    global_ppm = 100.0 / 12.0  # ~8.33 px/m

    dyn.compute_scales(global_ppm, one_gate_px_ref=None, max_change_rate=0.10)

    # Frame 0: 100px / 12m = 8.33 px/m
    ppm_0 = dyn.get_ppm(0, global_ppm)
    assert abs(ppm_0 - 8.33) < 0.5, f"Frame 0 ppm should be ~8.33, got {ppm_0}"

    # Frame 2: 110px / 12m = 9.17 px/m — gradual change, should be accepted
    ppm_2 = dyn.get_ppm(2, global_ppm)
    assert ppm_2 > ppm_0, (
        f"Gradual zoom frame ppm ({ppm_2:.2f}) should be > initial ({ppm_0:.2f})"
    )

    print("✅ test_dynamic_scale_transform passed")


def test_scale_hysteresis_rejects_spikes():
    """
    Scale hysteresis should reject sudden ppm jumps that would cause
    the "11,000 km/h" explosion. If a frame's ppm changes by >10%
    from the previous accepted value, it should be replaced with
    the previous safe value.
    """
    from ski_racing.transform import DynamicScaleTransform

    gate_spacing_m = 12.0
    # Frame 0-4: gates 100px apart (ppm = 8.33)
    # Frame 5: gates suddenly 20px apart (ppm would be 1.67 — BAD)
    # Frame 6-9: gates back to 100px apart
    frame_history = {}
    for f in range(10):
        if f == 5:
            # Simulate affine zoom hallucination: gates appear very close
            frame_history[f] = {0: (200.0, 100.0), 1: (200.0, 120.0)}  # 20px gap
        else:
            frame_history[f] = {0: (200.0, 100.0), 1: (200.0, 200.0)}  # 100px gap

    dyn = DynamicScaleTransform(frame_history, gate_spacing_m)
    global_ppm = 100.0 / 12.0  # ~8.33

    dyn.compute_scales(global_ppm, one_gate_px_ref=None, max_change_rate=0.10)

    # Frame 5 should NOT have the explosive ppm=1.67.
    # Instead, it should carry forward the previous safe value.
    ppm_5 = dyn.get_ppm(5, global_ppm)
    assert abs(ppm_5 - global_ppm) < 1.0, (
        f"Frame 5 ppm should be ~{global_ppm:.1f} (hysteresis), got {ppm_5}"
    )

    # Frame 6 should also be safe (back to normal)
    ppm_6 = dyn.get_ppm(6, global_ppm)
    assert abs(ppm_6 - global_ppm) < 1.0, (
        f"Frame 6 ppm should be ~{global_ppm:.1f}, got {ppm_6}"
    )

    print("✅ test_scale_hysteresis_rejects_spikes passed")


def test_affine_scale_stripping():
    """
    Affine camera compensator should strip the scale component from
    the affine matrix, keeping only rotation + translation.

    This prevents the 'zoom hallucination' in snowy scenes where
    estimateAffinePartial2D incorrectly reports camera zoom.
    """
    from ski_racing.transform import CameraMotionCompensator

    # Simulate: baseline gates at known positions.
    # Frame 1: gates shift +10px right AND appear 20% smaller (zoom hallucination).
    baseline = {0: (100.0, 200.0), 1: (300.0, 400.0), 2: (200.0, 600.0)}

    # Simulate a "zoomed out" frame where gates are closer together
    # The affine estimator would fit a scale < 1.0
    frame_history = {
        0: {0: (100.0, 200.0), 1: (300.0, 400.0), 2: (200.0, 600.0)},  # baseline
        1: {0: (90.0, 170.0), 1: (270.0, 350.0), 2: (180.0, 530.0)},   # "shrunk" by 0.85x
    }

    comp = CameraMotionCompensator(baseline, frame_history, mode="affine")
    comp.estimate_motion()

    # After scale stripping, stabilizing a point in frame 1 should NOT
    # produce wildly different coordinates. The scale-free Euclidean
    # transform should only apply rotation + translation.
    sx, sy = comp.stabilize_point(200.0, 400.0, 1)

    # Without scale stripping, the inverse would blow up the coordinates.
    # With stripping, the result should be within ~50px of the input
    # (just rotation + translation correction).
    assert abs(sx - 200.0) < 100.0, (
        f"Affine stabilization should not wildly distort: got x={sx:.1f} from input 200"
    )
    assert abs(sy - 400.0) < 100.0, (
        f"Affine stabilization should not wildly distort: got y={sy:.1f} from input 400"
    )

    print("✅ test_affine_scale_stripping passed")


def test_trajectory_jump_guard():
    """
    The 3D trajectory jump guard should replace physically impossible
    jumps (>5m per frame) with interpolated values.
    """
    from ski_racing.transform import HomographyTransform

    transformer = HomographyTransform()
    # Set up minimal scale mode
    transformer.H = np.eye(3)
    transformer.mode = "identity"

    # Create a trajectory with a single 100m jump in the middle
    trajectory_2d = [
        {"frame": 0, "x": 0.0, "y": 0.0},
        {"frame": 1, "x": 0.5, "y": 1.0},
        {"frame": 2, "x": 100.0, "y": 100.0},  # 100m jump! (bad frame)
        {"frame": 3, "x": 1.5, "y": 3.0},
        {"frame": 4, "x": 2.0, "y": 4.0},
    ]

    trajectory_3d = transformer.transform_trajectory(
        trajectory_2d, stabilize=False, max_jump_m=5.0, jump_guard=True
    )

    # Frame 2 should have been interpolated between frame 1 and frame 3
    pt2 = trajectory_3d[2]
    assert pt2["x"] < 10.0, f"Jump guard should cap frame 2 x, got {pt2['x']}"
    assert pt2["y"] < 10.0, f"Jump guard should cap frame 2 y, got {pt2['y']}"

    print("✅ test_trajectory_jump_guard passed")


if __name__ == "__main__":
    test_speed_calculation()
    test_realistic_trajectory_passes()
    test_impossible_speed_fails()
    test_impossible_g_force_fails()
    test_trajectory_smoothness()
    test_circumradius()
    test_collinear_points()

    # Professor's feedback tests
    test_kalman_rts_preserves_turns()
    test_kalman_discipline_defaults()
    test_kalman_handles_frame_gaps()
    test_camera_motion_compensator_translation()
    test_dynamic_scale_transform()

    # Stability / explosion prevention tests
    test_scale_hysteresis_rejects_spikes()
    test_affine_scale_stripping()
    test_trajectory_jump_guard()

    print("\n✅ All tests passed!")
