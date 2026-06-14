import math


def test_translation_camera_motion_spike_rejection():
    from ski_racing.transform import CameraMotionCompensator

    baseline = {
        0: (100.0, 200.0),
        1: (200.0, 200.0),
    }

    history = {}
    # Smooth drift: dx increases by 1px per frame.
    for f in range(10):
        dx = float(f)
        history[f] = {
            0: (baseline[0][0] + dx, baseline[0][1]),
            1: (baseline[1][0] + dx, baseline[1][1]),
        }

    # Inject a one-frame glitch.
    history[5] = {
        0: (baseline[0][0] + 1000.0, baseline[0][1]),
        1: (baseline[1][0] + 1000.0, baseline[1][1]),
    }

    comp = CameraMotionCompensator(baseline, history, mode="translation")
    comp.estimate_motion()

    # The interpolated + smoothed offset at the spike frame should be near the
    # expected drift (≈5), not the injected 1000px.
    assert 5 in comp.offsets
    dx5, dy5 = comp.offsets[5]
    assert abs(dy5) < 1e-6
    assert abs(dx5 - 5.0) < 5.0  # allow some median smoothing slack


def test_jump_guard_records_and_interpolates():
    from ski_racing.transform import HomographyTransform

    t = HomographyTransform()
    # Simple scale mapping: 100px vertical gap over 10m => 10 px/m.
    t.calculate_scale_from_gates(gates_2d=[(0.0, 0.0), (0.0, 100.0)], gate_spacing_m=10.0)

    traj2d = [
        {"frame": 0, "x": 0.0, "y": 50.0},
        {"frame": 1, "x": 10.0, "y": 50.0},
        {"frame": 2, "x": 10000.0, "y": 50.0},  # huge teleport
        {"frame": 3, "x": 20.0, "y": 50.0},
        {"frame": 4, "x": 30.0, "y": 50.0},
    ]

    out = t.transform_trajectory(
        traj2d,
        stabilize=False,
        jump_guard=True,
        max_jump_m=5.0,
        fps=30.0,
        stabilize_after_scale=True,
    )

    assert isinstance(t.jump_guard_info, dict)
    assert t.jump_guard_info.get("enabled") is True
    assert t.jump_guard_info.get("interpolated_points", 0) >= 1

    # Ensure the teleport is removed (no ~1000m jumps remain).
    max_jump = 0.0
    for i in range(1, len(out)):
        d = math.hypot(out[i]["x"] - out[i - 1]["x"], out[i]["y"] - out[i - 1]["y"])
        max_jump = max(max_jump, d)
    assert max_jump < 50.0

