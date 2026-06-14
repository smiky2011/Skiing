from __future__ import annotations

import math

from ski_racing.decoder import Observation, ViterbiDecoder


def make_obs(frame_idx: int, state: str, force_dnf: bool = False) -> Observation:
    high = 0.0
    low = -8.0
    if state == "R":
        emission = {"R": high, "B": low, "DNF": low}
    elif state == "B":
        emission = {"R": low, "B": high, "DNF": low}
    else:
        emission = {"R": low, "B": low, "DNF": high}
    return Observation(
        frame_idx=frame_idx,
        emission_log_prob=emission,
        geometric_residual=0.0,
        force_dnf=force_dnf,
    )


def test_dnf_absorbing_after_crash() -> None:
    decoder = ViterbiDecoder(lag=2, t_min=5)
    observations = [
        make_obs(0, "R"),
        make_obs(1, "B"),
        make_obs(2, "R"),
        make_obs(3, "B"),
        make_obs(4, "R"),
        make_obs(5, "DNF", force_dnf=True),
        make_obs(6, "DNF"),
        make_obs(7, "DNF"),
        make_obs(8, "DNF"),
    ]
    decoded = decoder.decode_fixed_lag(observations)
    states = [row["state"] for row in decoded]
    assert states[:5] == ["R", "B", "R", "B", "R"]
    assert states[5:] == ["DNF", "DNF", "DNF", "DNF"]


def test_score_valid_guard_short_and_minimum_length() -> None:
    decoder = ViterbiDecoder(lag=1, t_min=5)

    seq3 = [make_obs(i, "R" if i % 2 == 0 else "B") for i in range(3)]
    out3 = decoder.decode_fixed_lag(seq3)
    assert all(not row["score_valid"] for row in out3)
    assert all(row["s_star"] is None for row in out3)

    seq5 = [make_obs(i, "R" if i % 2 == 0 else "B") for i in range(5)]
    out5 = decoder.decode_fixed_lag(seq5)
    valid_flags = [row["score_valid"] for row in out5]
    assert valid_flags[:3] == [False, False, False]
    assert valid_flags[3:] == [True, True]


def test_perfect_alternation_hits_max_score() -> None:
    decoder = ViterbiDecoder(lag=9, t_min=5, residual_bonus_scale=0.0)
    observations = [make_obs(i, "R" if i % 2 == 0 else "B") for i in range(10)]
    out = decoder.decode_fixed_lag(observations)

    # First emitted frame uses full 10-frame window.
    s_star0 = out[0]["s_star"]
    assert s_star0 is not None
    expected = (9 * math.log(0.90)) / 10.0
    assert abs(s_star0 - expected) < 1e-6


def test_log_space_finite_for_90_frames() -> None:
    decoder = ViterbiDecoder(lag=12, t_min=5, debug=True)
    observations = []
    for i in range(90):
        phase = i % 3
        if phase == 0:
            emission = {"R": -0.1, "B": -2.5, "DNF": -3.0}
        elif phase == 1:
            emission = {"R": -2.5, "B": -0.1, "DNF": -3.0}
        else:
            emission = {"R": -1.7, "B": -1.7, "DNF": -1.6}
        observations.append(
            Observation(
                frame_idx=i,
                emission_log_prob=emission,
                geometric_residual=0.1,
                force_dnf=False,
            )
        )

    out = decoder.decode_fixed_lag(observations)
    assert len(out) == 90
    for row in out:
        if row["score_valid"]:
            assert row["s_star"] is not None
            assert math.isfinite(row["s_star"])
            if row["s_star_margin"] is not None:
                assert math.isfinite(row["s_star_margin"])

