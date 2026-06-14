import pytest

from ski_racing.pipeline import _apply_kalman_oob_safety


def _build_in_frame_trajectory(n, width, height):
    return [
        {
            "frame": i,
            "x": float((i * 7) % width),
            "y": float((i * 5) % height),
        }
        for i in range(n)
    ]


def test_kalman_oob_triggers_revert():
    """If Kalman output has >2% OOB points, pipeline reverts to pre-Kalman trajectory."""
    width, height = 1280, 720
    n = 100
    pre_kalman = _build_in_frame_trajectory(n=n, width=width, height=height)
    post_kalman = [dict(pt) for pt in pre_kalman]

    # Inject 10% OOB points
    for i in range(10):
        post_kalman[i * 10]["x"] = -500.0

    result, reverted, ratio = _apply_kalman_oob_safety(
        trajectory=post_kalman,
        trajectory_raw=pre_kalman,
        frame_width=width,
        frame_height=height,
        threshold=0.02,
    )

    assert reverted is True
    assert ratio == pytest.approx(0.10)
    assert result == pre_kalman
    for pt in result:
        assert 0.0 <= pt["x"] < width
        assert 0.0 <= pt["y"] < height


def test_kalman_oob_clamps_without_revert():
    """If OOB ratio <=2%, keep Kalman output and hard-clamp OOB points."""
    width, height = 1280, 720
    n = 100
    pre_kalman = _build_in_frame_trajectory(n=n, width=width, height=height)
    post_kalman = [dict(pt) for pt in pre_kalman]

    # 1/100 points OOB: should not revert.
    post_kalman[0]["x"] = float(width + 500)

    result, reverted, ratio = _apply_kalman_oob_safety(
        trajectory=post_kalman,
        trajectory_raw=pre_kalman,
        frame_width=width,
        frame_height=height,
        threshold=0.02,
    )

    assert reverted is False
    assert ratio == pytest.approx(0.01)
    assert result[0]["x"] == float(width - 1)
    for pt in result:
        assert 0.0 <= pt["x"] < width
        assert 0.0 <= pt["y"] < height
