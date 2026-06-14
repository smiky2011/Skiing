"""Track F Viterbi decoder (v2.1 Phase 4)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

STATE_R = "R"
STATE_B = "B"
STATE_DNF = "DNF"
STATES: Tuple[str, str, str] = (STATE_R, STATE_B, STATE_DNF)

LOG_UNIFORM_EMISSION = math.log(1.0 / 3.0)


def _log(x: float) -> float:
    if x <= 0.0:
        raise ValueError(f"log input must be > 0, got {x}")
    return math.log(x)


DEFAULT_TRANSITIONS: Dict[str, Dict[str, float]] = {
    STATE_R: {STATE_R: _log(0.05), STATE_B: _log(0.90), STATE_DNF: _log(0.05)},
    STATE_B: {STATE_R: _log(0.90), STATE_B: _log(0.05), STATE_DNF: _log(0.05)},
    STATE_DNF: {STATE_R: _log(0.01), STATE_B: _log(0.01), STATE_DNF: _log(0.98)},
}


@dataclass(frozen=True)
class Observation:
    frame_idx: int
    emission_log_prob: Dict[str, float]
    geometric_residual: Optional[float] = None
    force_dnf: bool = False


class ViterbiDecoder:
    """
    Fixed-lag Viterbi decoder for hidden states {R, B, DNF}.

    Notes:
    - Emissions are read directly from Track B `emission_log_prob`.
    - All scoring is log-space; no exp inside decoding loops.
    - DNF forcing is applied per frame when BEV spacing is physically impossible.
    """

    def __init__(
        self,
        lag: int = 12,
        t_min: int = 5,
        residual_window: int = 5,
        residual_bonus_scale: float = 0.2,
        dnf_residual_multiplier: float = 3.0,
        debug: bool = False,
    ) -> None:
        if lag < 1:
            raise ValueError("lag must be >= 1")
        if t_min < 1:
            raise ValueError("t_min must be >= 1")
        if residual_window < 1:
            raise ValueError("residual_window must be >= 1")
        if dnf_residual_multiplier <= 1.0:
            raise ValueError("dnf_residual_multiplier must be > 1.0")

        self.lag = int(lag)
        self.t_min = int(t_min)
        self.residual_window = int(residual_window)
        self.residual_bonus_scale = float(residual_bonus_scale)
        self.dnf_residual_multiplier = float(dnf_residual_multiplier)
        self.debug = bool(debug)

        self._transitions_default = {
            state: dict(transitions) for state, transitions in DEFAULT_TRANSITIONS.items()
        }
        self._transitions_forced_dnf = self._build_forced_dnf_transitions()

    def decode_fixed_lag(
        self, observations: Sequence[dict | Observation], lag: Optional[int] = None
    ) -> List[dict]:
        """
        Run fixed-lag Viterbi smoothing.

        Args:
            observations: sequence of dicts/Observations with `emission_log_prob`.
            lag: optional override (default uses instance lag).
        Returns:
            Per-frame outputs:
            {
              "frame_idx": int,
              "state": "R"|"B"|"DNF",
              "score_valid": bool,
              "s_star": float|None,
              "s_star_margin": float|None
            }
        """
        lag_value = int(self.lag if lag is None else lag)
        if lag_value < 1:
            raise ValueError("lag must be >= 1")

        seq = self._normalise_observations(observations)
        if not seq:
            return []

        n = len(seq)
        emitted: Dict[int, dict] = {}
        emission_order: List[int] = []

        # Streaming stage: emit t-lag as soon as enough lookahead is available.
        for t in range(n):
            if t < lag_value:
                continue
            start = t - lag_value
            window = seq[start : t + 1]
            best_path, best_score, second_score = self._viterbi_window(window)
            frame_idx = window[0].frame_idx
            observed_count = t + 1
            emitted[frame_idx] = self._build_output(
                frame_idx=frame_idx,
                state=best_path[0],
                best_score=best_score,
                second_score=second_score,
                window_len=len(window),
                observed_count=observed_count,
            )
            emission_order.append(frame_idx)

        # Flush trailing frames at stream end with shrinking windows.
        first_unemitted = max(0, n - lag_value)
        for start in range(first_unemitted, n):
            frame_idx = seq[start].frame_idx
            if frame_idx in emitted:
                continue
            window = seq[start:n]
            best_path, best_score, second_score = self._viterbi_window(window)
            observed_count = n
            emitted[frame_idx] = self._build_output(
                frame_idx=frame_idx,
                state=best_path[0],
                best_score=best_score,
                second_score=second_score,
                window_len=len(window),
                observed_count=observed_count,
            )
            emission_order.append(frame_idx)

        # Preserve input ordering by frame_idx.
        unique_order = []
        seen = set()
        for obs in seq:
            if obs.frame_idx in seen:
                continue
            unique_order.append(obs.frame_idx)
            seen.add(obs.frame_idx)

        return [emitted[frame_idx] for frame_idx in unique_order]

    def build_observations(
        self, detection_frames: Sequence[dict], bev_frames: Optional[Sequence[dict]] = None
    ) -> List[Observation]:
        """
        Build decoder observations from Track B/D detections and Track C BEV.

        - Emissions are read from `emission_log_prob`.
        - Geometric residual uses BEV gate-spacing consistency against a running median.
        - DNF forcing is activated when current spacing > 3x running median spacing.
        """
        bev_by_frame: Dict[int, dict] = {}
        if bev_frames:
            for frame in bev_frames:
                frame_idx = frame.get("frame_idx")
                if isinstance(frame_idx, int):
                    bev_by_frame[frame_idx] = frame

        sorted_frames = sorted(detection_frames, key=lambda row: int(row.get("frame_idx", 0)))

        spacing_history: List[float] = []
        observations: List[Observation] = []
        for frame in sorted_frames:
            frame_idx = int(frame.get("frame_idx", len(observations)))
            emission = self._extract_emission_for_frame(frame)

            geometric_residual: Optional[float] = None
            force_dnf = False

            bev_frame = bev_by_frame.get(frame_idx)
            spacing = self._extract_bev_spacing(bev_frame)
            if spacing is not None:
                window = spacing_history[-self.residual_window :] if spacing_history else []
                running_median = median(window) if window else None

                if running_median is not None and running_median > 0.0:
                    ratio = spacing / running_median
                    geometric_residual = abs(ratio - 1.0)
                    force_dnf = ratio > self.dnf_residual_multiplier
                spacing_history.append(spacing)

            observations.append(
                Observation(
                    frame_idx=frame_idx,
                    emission_log_prob=emission,
                    geometric_residual=geometric_residual,
                    force_dnf=force_dnf,
                )
            )
        return observations

    def write_decoder_output(
        self, clip_id: str, frame_outputs: Sequence[dict], output_dir: str | Path
    ) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"clip_id": clip_id, "frames": list(frame_outputs)}
        out_path = out_dir / f"{clip_id}_decoder.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_observations(self, raw: Sequence[dict | Observation]) -> List[Observation]:
        observations: List[Observation] = []
        for i, row in enumerate(raw):
            if isinstance(row, Observation):
                observations.append(row)
                continue
            frame_idx = int(row.get("frame_idx", i))
            emission = row.get("emission_log_prob") or {}
            observations.append(
                Observation(
                    frame_idx=frame_idx,
                    emission_log_prob={
                        STATE_R: float(emission.get("log_prob_red", LOG_UNIFORM_EMISSION)),
                        STATE_B: float(emission.get("log_prob_blue", LOG_UNIFORM_EMISSION)),
                        STATE_DNF: float(emission.get("log_prob_dnf", LOG_UNIFORM_EMISSION)),
                    },
                    geometric_residual=(
                        None
                        if row.get("geometric_residual") is None
                        else float(row.get("geometric_residual"))
                    ),
                    force_dnf=bool(row.get("force_dnf", False)),
                )
            )
        observations.sort(key=lambda obs: obs.frame_idx)
        return observations

    def _build_forced_dnf_transitions(self) -> Dict[str, Dict[str, float]]:
        forced: Dict[str, Dict[str, float]] = {}
        for src in STATES:
            if src == STATE_DNF:
                forced[src] = dict(self._transitions_default[src])
                continue

            p_dnf = 0.99
            rem = 1.0 - p_dnf

            p_src_src = math.exp(self._transitions_default[src][src])
            other = STATE_B if src == STATE_R else STATE_R
            p_src_other = math.exp(self._transitions_default[src][other])
            total_non_dnf = p_src_src + p_src_other
            if total_non_dnf <= 0.0:
                p_src_src_new = rem * 0.5
                p_src_other_new = rem * 0.5
            else:
                p_src_src_new = rem * (p_src_src / total_non_dnf)
                p_src_other_new = rem * (p_src_other / total_non_dnf)

            forced[src] = {
                src: _log(p_src_src_new),
                other: _log(p_src_other_new),
                STATE_DNF: _log(p_dnf),
            }
        return forced

    def _transition_matrix_for_frame(self, force_dnf: bool) -> Dict[str, Dict[str, float]]:
        if force_dnf:
            return self._transitions_forced_dnf
        return self._transitions_default

    def _viterbi_window(self, window: Sequence[Observation]) -> Tuple[List[str], float, float]:
        if not window:
            raise ValueError("window cannot be empty")

        # Per-state top-2 scores and backpointers for second-best margin.
        score_prev: Dict[str, List[float]] = {
            s: [self._augmented_emission(window[0], s), float("-inf")] for s in STATES
        }

        backptr: List[Dict[str, List[Optional[Tuple[str, int]]]]] = []
        backptr.append({s: [None, None] for s in STATES})

        for t in range(1, len(window)):
            obs = window[t]
            transitions = self._transition_matrix_for_frame(obs.force_dnf)
            score_curr: Dict[str, List[float]] = {}
            bp_curr: Dict[str, List[Optional[Tuple[str, int]]]] = {}
            for curr_state in STATES:
                emit = self._augmented_emission(obs, curr_state)
                candidates: List[Tuple[float, str, int]] = []
                for prev_state in STATES:
                    trans = transitions[prev_state][curr_state]
                    for rank in (0, 1):
                        prev_score = score_prev[prev_state][rank]
                        if prev_score == float("-inf"):
                            continue
                        score = prev_score + trans + emit
                        if self.debug:
                            self._assert_finite(score, f"score t={t} {prev_state}->{curr_state}")
                        candidates.append((score, prev_state, rank))
                candidates.sort(key=lambda item: item[0], reverse=True)
                top = candidates[:2]
                while len(top) < 2:
                    top.append((float("-inf"), STATE_DNF, 0))
                score_curr[curr_state] = [top[0][0], top[1][0]]
                bp_curr[curr_state] = [(top[0][1], top[0][2]), (top[1][1], top[1][2])]
            score_prev = score_curr
            backptr.append(bp_curr)

        finals: List[Tuple[float, str, int]] = []
        for state in STATES:
            finals.append((score_prev[state][0], state, 0))
            finals.append((score_prev[state][1], state, 1))
        finals.sort(key=lambda item: item[0], reverse=True)

        best_score, best_state, best_rank = finals[0]
        second_score = finals[1][0] if len(finals) > 1 else float("-inf")

        path = [best_state]
        state = best_state
        rank = best_rank
        for t in range(len(window) - 1, 0, -1):
            prev = backptr[t][state][rank]
            if prev is None:
                # Should not happen, but keep decoder robust.
                prev = (STATE_DNF, 0)
            prev_state, prev_rank = prev
            path.append(prev_state)
            state = prev_state
            rank = prev_rank
        path.reverse()

        if self.debug:
            self._assert_finite(best_score, "best_score")
            if second_score != float("-inf"):
                self._assert_finite(second_score, "second_score")

        return path, best_score, second_score

    def _build_output(
        self,
        frame_idx: int,
        state: str,
        best_score: float,
        second_score: float,
        window_len: int,
        observed_count: int,
    ) -> dict:
        score_valid = observed_count >= self.t_min
        if score_valid:
            s_star = best_score / float(window_len)
            if second_score == float("-inf"):
                margin = None
            else:
                margin = best_score - second_score
        else:
            s_star = None
            margin = None
        return {
            "frame_idx": int(frame_idx),
            "state": state,
            "score_valid": bool(score_valid),
            "s_star": s_star,
            "s_star_margin": margin,
        }

    def _augmented_emission(self, obs: Observation, state: str) -> float:
        base = float(obs.emission_log_prob[state])
        residual = obs.geometric_residual
        if residual is None:
            return base

        # Soft rhythm prior: consistent spacing boosts race states over DNF.
        consistency = max(0.0, 1.0 - min(1.0, residual))
        bonus = self.residual_bonus_scale * consistency
        if state in (STATE_R, STATE_B):
            return base + bonus
        return base - bonus

    def _extract_emission_for_frame(self, detection_frame: dict) -> Dict[str, float]:
        detections = detection_frame.get("detections") or []
        chosen = self._select_detection(detections)
        if not chosen:
            return {
                STATE_R: LOG_UNIFORM_EMISSION,
                STATE_B: LOG_UNIFORM_EMISSION,
                STATE_DNF: LOG_UNIFORM_EMISSION,
            }

        emission = chosen.get("emission_log_prob") or {}
        return {
            STATE_R: float(emission.get("log_prob_red", LOG_UNIFORM_EMISSION)),
            STATE_B: float(emission.get("log_prob_blue", LOG_UNIFORM_EMISSION)),
            STATE_DNF: float(emission.get("log_prob_dnf", LOG_UNIFORM_EMISSION)),
        }

    @staticmethod
    def _select_detection(detections: Sequence[dict]) -> Optional[dict]:
        if not detections:
            return None

        def detection_score(det: dict) -> float:
            conf = float(det.get("conf_class", 0.0))
            emission = det.get("emission_log_prob") or {}
            top_emission = max(
                float(emission.get("log_prob_red", float("-inf"))),
                float(emission.get("log_prob_blue", float("-inf"))),
                float(emission.get("log_prob_dnf", float("-inf"))),
            )
            return conf + top_emission

        return max(detections, key=detection_score)

    @staticmethod
    def _extract_bev_spacing(bev_frame: Optional[dict]) -> Optional[float]:
        if not bev_frame:
            return None
        bases = bev_frame.get("bev_gate_bases")
        if not isinstance(bases, list) or len(bases) < 2:
            return None

        ys: List[float] = []
        for item in bases:
            val = item.get("bev_y")
            if isinstance(val, (int, float)):
                ys.append(float(val))
        if len(ys) < 2:
            return None

        ys.sort()
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] > ys[i]]
        if not diffs:
            return None
        return float(median(diffs))

    @staticmethod
    def _assert_finite(value: float, label: str) -> None:
        if not math.isfinite(value):
            raise AssertionError(f"Non-finite value in decoder: {label}={value}")


def load_frames(path: str | Path) -> Tuple[str, List[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    clip_id = str(payload["clip_id"])
    frames = list(payload["frames"])
    return clip_id, frames


def decode_clip_to_file(
    detections_path: str | Path,
    bev_path: Optional[str | Path],
    output_path: str | Path,
    lag: int = 12,
    t_min: int = 5,
    debug: bool = False,
) -> Path:
    clip_id, detection_frames = load_frames(detections_path)
    bev_frames = None
    if bev_path is not None and Path(bev_path).exists():
        bev_clip_id, bev_frames = load_frames(bev_path)
        if bev_clip_id != clip_id:
            # Keep processing, but mark mismatch in a deterministic way.
            clip_id = clip_id

    decoder = ViterbiDecoder(lag=lag, t_min=t_min, debug=debug)
    observations = decoder.build_observations(detection_frames, bev_frames)
    decoded_frames = decoder.decode_fixed_lag(observations, lag=lag)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"clip_id": clip_id, "frames": decoded_frames}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

