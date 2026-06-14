"""Sequence-based tracker initialisation using a bounded FIFO buffer."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Lazy import guard — ViterbiDecoder lives in the same package.
# Using a try/except so unit tests that mock ski_racing.initialiser
# in isolation still work even if decoder is unavailable.
try:
    from ski_racing.decoder import ViterbiDecoder as _ViterbiDecoder
    _VITERBI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ViterbiDecoder = None  # type: ignore[assignment,misc]
    _VITERBI_AVAILABLE = False


class SequenceInitialiser:
    """Bootstrap tracker state from a retroactive decode once a valid chain is observed."""

    _STATE_R = "R"
    _STATE_B = "B"
    _STATE_DNF = "DNF"
    _STATE_ORDER = (_STATE_R, _STATE_B, _STATE_DNF)

    def __init__(
        self,
        clip_id: str,
        *,
        tracker: Optional[Any] = None,
        safety_monitor: Optional[Any] = None,
        retro_decoder: Optional[Callable[[List[Dict[str, float]]], Any]] = None,
        detector_callback: Optional[Callable[..., Any]] = None,
        outputs_dir: Optional[Path] = None,
        max_buffer_depth: int = 90,
        t_min: int = 5,
        n_persist: int = 3,
        tau_seq: float = -1.5,
        max_position_reversal: float = 2.5,
    ) -> None:
        self.clip_id = str(clip_id)
        self.tracker = tracker
        self.safety_monitor = safety_monitor
        self.retro_decoder = retro_decoder
        self.detector_callback = detector_callback
        self.max_buffer_depth = int(max_buffer_depth)
        self.t_min = int(t_min)
        self.n_persist = int(n_persist)
        self.tau_seq = float(tau_seq)
        self.max_position_reversal = float(max_position_reversal)

        # Default output dir: <project_root>/outputs/  (two levels up from ski_racing/)
        project_root = Path(__file__).resolve().parents[1]
        base_outputs = Path(outputs_dir) if outputs_dir is not None else project_root / "outputs"
        base_outputs.mkdir(parents=True, exist_ok=True)
        self.output_path = base_outputs / f"{self.clip_id}_init.json"

        self.events: List[Dict[str, Any]] = []
        if self.output_path.exists():
            try:
                payload = json.loads(self.output_path.read_text(encoding="utf-8"))
                maybe_events = payload.get("events")
                if isinstance(maybe_events, list):
                    self.events = list(maybe_events)
            except (json.JSONDecodeError, OSError):
                self.events = []

        self._buffer: List[Dict[str, Any]] = []
        self.flag_events: List[Dict[str, Any]] = []
        self.initialized = False
        self.observation_mode = True
        self.system_uninitialized = True

    @property
    def buffer_depth(self) -> int:
        return len(self._buffer)

    def update(
        self,
        *,
        frame_idx: int,
        detections: Optional[Sequence[Dict[str, Any]]] = None,
        bev_positions: Optional[Sequence[Any]] = None,
        delta_t_s: float = 0.0,
        decoder_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a frame to the FIFO and conditionally trigger initialisation/reset.

        Returns a status payload for the current frame.
        """
        entry = self._build_buffer_entry(
            frame_idx=frame_idx,
            detections=detections,
            bev_positions=bev_positions,
            delta_t_s=delta_t_s,
        )
        self._buffer.append(entry)

        if self.initialized:
            return {
                "frame_idx": int(frame_idx),
                "triggered": False,
                "reset": False,
                "initialized": True,
                "system_uninitialized": self.system_uninitialized,
            }

        trigger, checks = self._should_trigger(decoder_output or {})
        if trigger:
            return self._trigger_initialisation(
                frame_idx=int(frame_idx),
                decoder_output=decoder_output or {},
                checks=checks,
            )

        if len(self._buffer) >= self.max_buffer_depth:
            return self._reset_buffer(
                frame_idx=int(frame_idx),
                reason=f"no_valid_chain_in_{self.max_buffer_depth}_frames",
            )

        return {
            "frame_idx": int(frame_idx),
            "triggered": False,
            "reset": False,
            "initialized": False,
            "system_uninitialized": self.system_uninitialized,
            "buffer_depth": len(self._buffer),
            "checks": checks,
        }

    def _build_buffer_entry(
        self,
        *,
        frame_idx: int,
        detections: Optional[Sequence[Dict[str, Any]]],
        bev_positions: Optional[Sequence[Any]],
        delta_t_s: float,
    ) -> Dict[str, Any]:
        dets = [copy.deepcopy(d) for d in (detections or [])]
        emissions: List[Dict[str, float]] = []
        for det in dets:
            emission = det.get("emission_log_prob", {})
            emissions.append(
                {
                    "log_prob_red": float(emission.get("log_prob_red", -0.5)),
                    "log_prob_blue": float(emission.get("log_prob_blue", -0.5)),
                    "log_prob_dnf": float(emission.get("log_prob_dnf", -0.5)),
                }
            )
        return {
            "frame_idx": int(frame_idx),
            "detections": dets,
            "emission_log_probs": emissions,
            "bev_positions": copy.deepcopy(list(bev_positions or [])),
            "delta_t_s": float(delta_t_s),
        }

    def _should_trigger(self, decoder_output: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        gates_confirmed = self._count_confirmed_gates(decoder_output)
        score_valid = bool(decoder_output.get("score_valid", False))
        s_star = decoder_output.get("s_star")
        s_star_ok = isinstance(s_star, (int, float)) and float(s_star) >= self.tau_seq
        topology_ok = self._check_topology(decoder_output)
        persistence_ok = self._check_persistence(decoder_output, gates_confirmed)

        checks = {
            "gates_confirmed": int(gates_confirmed),
            "score_valid": bool(score_valid),
            "s_star": None if s_star is None else float(s_star),
            "s_star_ok": bool(s_star_ok),
            "topology_ok": bool(topology_ok),
            "persistence_ok": bool(persistence_ok),
        }
        trigger = (
            gates_confirmed >= self.t_min
            and score_valid
            and s_star_ok
            and topology_ok
            and persistence_ok
        )
        return trigger, checks

    def _count_confirmed_gates(self, decoder_output: Dict[str, Any]) -> int:
        maybe_count = decoder_output.get("gates_confirmed")
        if isinstance(maybe_count, int):
            return max(0, maybe_count)
        confirmed_ids = decoder_output.get("confirmed_gate_ids")
        if isinstance(confirmed_ids, list):
            return len({str(v) for v in confirmed_ids})
        gate_keys = set()
        for entry in self._buffer:
            for det_idx, det in enumerate(entry["detections"]):
                key = self._gate_key(
                    detection=det,
                    bev_positions=entry["bev_positions"],
                    det_index=det_idx,
                )
                if key is not None:
                    gate_keys.add(key)
        return len(gate_keys)

    def _check_topology(self, decoder_output: Dict[str, Any]) -> bool:
        if isinstance(decoder_output.get("topology_ok"), bool):
            return bool(decoder_output["topology_ok"])

        y_positions: List[float] = []
        for entry in self._buffer:
            ys: List[float] = []
            for det_idx, det in enumerate(entry["detections"]):
                bev = self._resolve_bev_position(
                    detection=det,
                    bev_positions=entry["bev_positions"],
                    det_index=det_idx,
                )
                if bev is not None:
                    ys.append(float(bev[1]))
            if ys:
                y_positions.append(sum(ys) / len(ys))

        if len(y_positions) < 3:
            return True

        diffs = [y_positions[i] - y_positions[i - 1] for i in range(1, len(y_positions))]
        non_zero = [d for d in diffs if abs(d) > 1e-6]
        if not non_zero:
            return True

        nominal_dir = 1.0 if sum(non_zero) >= 0.0 else -1.0
        for d in non_zero:
            if d * nominal_dir < -self.max_position_reversal:
                return False
        return True

    def _check_persistence(self, decoder_output: Dict[str, Any], gates_confirmed: int) -> bool:
        if isinstance(decoder_output.get("persistence_ok"), bool):
            return bool(decoder_output["persistence_ok"])

        frames_by_key: Dict[str, List[int]] = {}
        for entry in self._buffer:
            present_keys = set()
            frame_idx = int(entry["frame_idx"])
            for det_idx, det in enumerate(entry["detections"]):
                key = self._gate_key(
                    detection=det,
                    bev_positions=entry["bev_positions"],
                    det_index=det_idx,
                )
                if key is not None:
                    present_keys.add(key)
            for key in present_keys:
                frames_by_key.setdefault(key, []).append(frame_idx)

        if not frames_by_key:
            return False

        persisted = 0
        for frame_indices in frames_by_key.values():
            longest = 1
            current = 1
            for idx in range(1, len(frame_indices)):
                if frame_indices[idx] == frame_indices[idx - 1] + 1:
                    current += 1
                else:
                    current = 1
                longest = max(longest, current)
            if longest >= self.n_persist:
                persisted += 1

        required = min(max(1, gates_confirmed), len(frames_by_key))
        return persisted >= required

    def _gate_key(
        self,
        *,
        detection: Dict[str, Any],
        bev_positions: Sequence[Any],
        det_index: int,
    ) -> Optional[str]:
        for key_name in ("gate_id", "track_id", "persistent_id"):
            if key_name in detection:
                return str(detection[key_name])

        bev = self._resolve_bev_position(
            detection=detection,
            bev_positions=bev_positions,
            det_index=det_index,
        )
        class_label = str(detection.get("class_label", "unknown"))
        if bev is not None:
            return f"{class_label}:{round(float(bev[0]), 1)}"
        if "class_label" in detection:
            return class_label
        return None

    @staticmethod
    def _resolve_bev_position(
        *,
        detection: Dict[str, Any],
        bev_positions: Sequence[Any],
        det_index: int,
    ) -> Optional[Tuple[float, float]]:
        if "bev_x" in detection and "bev_y" in detection:
            return float(detection["bev_x"]), float(detection["bev_y"])

        det_id = detection.get("detection_id")
        if det_id is not None:
            for item in bev_positions:
                if (
                    isinstance(item, dict)
                    and item.get("detection_id") == det_id
                    and "bev_x" in item
                    and "bev_y" in item
                ):
                    return float(item["bev_x"]), float(item["bev_y"])

        if det_index < len(bev_positions):
            item = bev_positions[det_index]
            if isinstance(item, dict):
                if "bev_x" in item and "bev_y" in item:
                    return float(item["bev_x"]), float(item["bev_y"])
                if "x" in item and "y" in item:
                    return float(item["x"]), float(item["y"])
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                return float(item[0]), float(item[1])

        return None

    def _trigger_initialisation(
        self,
        *,
        frame_idx: int,
        decoder_output: Dict[str, Any],
        checks: Dict[str, Any],
    ) -> Dict[str, Any]:
        states, retro_s_star = self._run_retroactive_viterbi()
        retro_assignments = self._build_retroactive_assignments(states)
        seeded_tracks = self._bootstrap_tracks(retro_assignments)

        s_star_at_trigger = decoder_output.get("s_star")
        if not isinstance(s_star_at_trigger, (int, float)):
            s_star_at_trigger = retro_s_star

        event = {
            "frame_idx": int(frame_idx),
            "event": "INITIALIZED",
            "buffer_depth_at_trigger": len(self._buffer),
            "gates_confirmed": int(checks["gates_confirmed"]),
            "s_star_at_trigger": None
            if s_star_at_trigger is None
            else float(s_star_at_trigger),
        }
        self._append_event(event)

        self.initialized = True
        self.observation_mode = False
        self._emit_system_uninitialized(False, frame_idx=frame_idx)
        self._buffer.clear()

        return {
            "frame_idx": int(frame_idx),
            "triggered": True,
            "reset": False,
            "initialized": True,
            "system_uninitialized": self.system_uninitialized,
            "forward_halted": True,
            "event": event,
            "retroactive_assignments": retro_assignments,
            "seeded_tracks": seeded_tracks,
            "retro_s_star": retro_s_star,
        }

    def _reset_buffer(self, *, frame_idx: int, reason: str) -> Dict[str, Any]:
        event = {
            "frame_idx": int(frame_idx),
            "event": "RESET",
            "reason": str(reason),
        }
        self._append_event(event)
        self._buffer.clear()
        self.initialized = False
        self.observation_mode = True
        self._emit_system_uninitialized(True, frame_idx=frame_idx)
        return {
            "frame_idx": int(frame_idx),
            "triggered": False,
            "reset": True,
            "initialized": False,
            "system_uninitialized": self.system_uninitialized,
            "event": event,
        }

    def _run_retroactive_viterbi(self) -> Tuple[List[str], Optional[float]]:
        frame_emissions = [
            self._aggregate_emissions(frame_entry["emission_log_probs"])
            for frame_entry in self._buffer
        ]
        if not frame_emissions:
            return [], None

        # --- Path 1: caller-supplied retro_decoder callback ---
        if self.retro_decoder is not None:
            decoded = self.retro_decoder(frame_emissions)
            parsed_states, parsed_score = self._parse_external_decoder_result(decoded)
            if parsed_states:
                return parsed_states, parsed_score

        # --- Path 2: canonical ViterbiDecoder from ski_racing.decoder (Track F) ---
        if _VITERBI_AVAILABLE and _ViterbiDecoder is not None:
            # Build Observation-compatible dicts from the aggregated emissions.
            obs_dicts = [
                {
                    "frame_idx": i,
                    "emission_log_prob": {
                        "log_prob_red":  em["log_prob_red"],
                        "log_prob_blue": em["log_prob_blue"],
                        "log_prob_dnf":  em["log_prob_dnf"],
                    },
                }
                for i, em in enumerate(frame_emissions)
            ]
            decoder = _ViterbiDecoder(lag=len(obs_dicts), t_min=self.t_min)
            frame_outputs = decoder.decode_fixed_lag(obs_dicts, lag=len(obs_dicts))
            states = [fo["state"] for fo in frame_outputs]
            # Compute normalised score from the outputs
            s_stars = [fo["s_star"] for fo in frame_outputs if fo.get("s_star") is not None]
            score = float(sum(s_stars) / len(s_stars)) if s_stars else None
            if states:
                return states, score

        # --- Path 3: internal fallback (Track G's original — only reached if
        #     ski_racing.decoder is unavailable, e.g. in isolated unit tests) ---
        log_transitions = {
            self._STATE_R: {
                self._STATE_R: math.log(0.10),
                self._STATE_B: math.log(0.88),
                self._STATE_DNF: math.log(0.02),
            },
            self._STATE_B: {
                self._STATE_R: math.log(0.88),
                self._STATE_B: math.log(0.10),
                self._STATE_DNF: math.log(0.02),
            },
            self._STATE_DNF: {
                self._STATE_R: math.log(1e-6),
                self._STATE_B: math.log(1e-6),
                self._STATE_DNF: math.log(1.0 - 2e-6),
            },
        }
        log_start = {
            self._STATE_R: math.log(0.49),
            self._STATE_B: math.log(0.49),
            self._STATE_DNF: math.log(0.02),
        }

        dp: List[Dict[str, float]] = []
        backptr: List[Dict[str, str]] = []

        first_emission = frame_emissions[0]
        first = {
            state: log_start[state] + first_emission[self._state_to_emission_key(state)]
            for state in self._STATE_ORDER
        }
        dp.append(first)
        backptr.append({})

        for t in range(1, len(frame_emissions)):
            emission = frame_emissions[t]
            curr: Dict[str, float] = {}
            prev_choice: Dict[str, str] = {}
            for state in self._STATE_ORDER:
                best_prev = None
                best_score = -float("inf")
                for prev_state in self._STATE_ORDER:
                    score = dp[t - 1][prev_state] + log_transitions[prev_state][state]
                    if score > best_score:
                        best_score = score
                        best_prev = prev_state
                curr[state] = best_score + emission[self._state_to_emission_key(state)]
                prev_choice[state] = str(best_prev)
            dp.append(curr)
            backptr.append(prev_choice)

        last_scores = dp[-1]
        best_last_state = max(last_scores, key=last_scores.get)
        best_total_score = float(last_scores[best_last_state])

        states_rev = [best_last_state]
        for t in range(len(frame_emissions) - 1, 0, -1):
            prev_state = backptr[t][states_rev[-1]]
            states_rev.append(prev_state)
        states = list(reversed(states_rev))
        normalized = best_total_score / float(len(frame_emissions))
        return states, normalized

    @staticmethod
    def _state_to_emission_key(state: str) -> str:
        if state == SequenceInitialiser._STATE_R:
            return "log_prob_red"
        if state == SequenceInitialiser._STATE_B:
            return "log_prob_blue"
        return "log_prob_dnf"

    @staticmethod
    def _aggregate_emissions(emissions: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not emissions:
            return {
                "log_prob_red": -0.5,
                "log_prob_blue": -0.5,
                "log_prob_dnf": -0.5,
            }
        return {
            "log_prob_red": max(float(e["log_prob_red"]) for e in emissions),
            "log_prob_blue": max(float(e["log_prob_blue"]) for e in emissions),
            "log_prob_dnf": max(float(e["log_prob_dnf"]) for e in emissions),
        }

    def _parse_external_decoder_result(self, decoded: Any) -> Tuple[List[str], Optional[float]]:
        if isinstance(decoded, dict):
            raw_states = decoded.get("states") or decoded.get("state_sequence") or []
            raw_score = decoded.get("s_star")
        elif isinstance(decoded, tuple) and len(decoded) >= 1:
            raw_states = decoded[0]
            raw_score = decoded[1] if len(decoded) > 1 else None
        else:
            raw_states = decoded if isinstance(decoded, list) else []
            raw_score = None

        states: List[str] = []
        for state in raw_states:
            s = str(state)
            if s in self._STATE_ORDER:
                states.append(s)
        score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        return states, score

    def _build_retroactive_assignments(self, states: Sequence[str]) -> List[Dict[str, Any]]:
        gate_to_track: Dict[str, int] = {}
        next_track_id = 1
        output: List[Dict[str, Any]] = []

        for i, entry in enumerate(self._buffer):
            state = str(states[i]) if i < len(states) else self._STATE_DNF
            detections_out: List[Dict[str, Any]] = []
            for det_idx, det in enumerate(entry["detections"]):
                gate_key = self._gate_key(
                    detection=det,
                    bev_positions=entry["bev_positions"],
                    det_index=det_idx,
                )
                track_id: Optional[int] = None
                if gate_key is not None and state != self._STATE_DNF:
                    track_id = gate_to_track.get(gate_key)
                    if track_id is None:
                        track_id = next_track_id
                        gate_to_track[gate_key] = track_id
                        next_track_id += 1

                detections_out.append(
                    {
                        "det_index": int(det_idx),
                        "detection_id": str(det.get("detection_id", f"{entry['frame_idx']}_{det_idx}")),
                        "state": state,
                        "track_id": track_id,
                    }
                )

            output.append(
                {
                    "frame_idx": int(entry["frame_idx"]),
                    "state": state,
                    "detections": detections_out,
                }
            )
        return output

    def _bootstrap_tracks(
        self, retroactive_assignments: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        histories: Dict[int, List[Dict[str, float]]] = {}
        for entry, assignment in zip(self._buffer, retroactive_assignments):
            if assignment["state"] == self._STATE_DNF:
                continue
            frame_idx = int(entry["frame_idx"])
            for det_assignment in assignment["detections"]:
                track_id = det_assignment.get("track_id")
                if track_id is None:
                    continue
                det_index = int(det_assignment["det_index"])
                if det_index >= len(entry["detections"]):
                    continue
                det = entry["detections"][det_index]
                bev = self._resolve_bev_position(
                    detection=det,
                    bev_positions=entry["bev_positions"],
                    det_index=det_index,
                )
                if bev is None:
                    continue
                area = self._bbox_area(det.get("bbox_xyxy"))
                histories.setdefault(int(track_id), []).append(
                    {
                        "frame_idx": float(frame_idx),
                        "x": float(bev[0]),
                        "y": float(bev[1]),
                        "delta_t_s": max(float(entry["delta_t_s"]), 1e-6),
                        "bbox_area": float(area),
                    }
                )

        seeded: List[Dict[str, Any]] = []
        for track_id in sorted(histories):
            history = histories[track_id]
            if not history:
                continue
            x = float(history[-1]["x"])
            y = float(history[-1]["y"])
            vx, vy = self._estimate_velocity(history)
            areas = [h["bbox_area"] for h in history if h["bbox_area"] > 0.0]
            scale = float(sum(areas) / len(areas)) if areas else 1.0

            state_vector = [x, y, vx, vy, scale, 0.0]
            payload = {
                "track_id": int(track_id),
                "frame_idx": int(history[-1]["frame_idx"]),
                "state_vector": state_vector,
                "history_length": len(history),
            }
            payload["registered"] = self._register_track(payload)
            seeded.append(payload)

        return seeded

    @staticmethod
    def _bbox_area(bbox_xyxy: Any) -> float:
        if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
            return 0.0
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _estimate_velocity(history: Sequence[Dict[str, float]]) -> Tuple[float, float]:
        if len(history) < 2:
            return 0.0, 0.0
        tail = list(history[-3:])
        weighted_vx = 0.0
        weighted_vy = 0.0
        weight_total = 0.0
        for prev, curr in zip(tail[:-1], tail[1:]):
            dt = max(float(curr["delta_t_s"]), 1e-6)
            seg_vx = (float(curr["x"]) - float(prev["x"])) / dt
            seg_vy = (float(curr["y"]) - float(prev["y"])) / dt
            weighted_vx += seg_vx * dt
            weighted_vy += seg_vy * dt
            weight_total += dt
        if weight_total <= 0.0:
            return 0.0, 0.0
        return weighted_vx / weight_total, weighted_vy / weight_total

    def _register_track(self, payload: Dict[str, Any]) -> bool:
        if self.tracker is None:
            return False

        methods = ("register_track", "initialise_track", "initialize_track", "seed_track")
        for method_name in methods:
            method = getattr(self.tracker, method_name, None)
            if not callable(method):
                continue
            try:
                result = method(
                    track_id=payload["track_id"],
                    state_vector=payload["state_vector"],
                    frame_idx=payload["frame_idx"],
                    metadata={"history_length": payload["history_length"]},
                )
                return bool(True if result is None else result)
            except TypeError:
                try:
                    result = method(payload)
                    return bool(True if result is None else result)
                except TypeError:
                    continue

        if callable(self.tracker):
            try:
                result = self.tracker(payload)
                return bool(True if result is None else result)
            except TypeError:
                return False

        return False

    def _append_event(self, event: Dict[str, Any]) -> None:
        self.events.append(dict(event))
        payload = {"clip_id": self.clip_id, "events": self.events}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _emit_system_uninitialized(self, value: bool, *, frame_idx: int) -> None:
        flag = bool(value)
        self.system_uninitialized = flag
        self.flag_events.append({"frame_idx": int(frame_idx), "SYSTEM_UNINITIALIZED": flag})

        if self.safety_monitor is None:
            return

        monitor = self.safety_monitor
        if callable(monitor):
            try:
                monitor(flag=flag, frame_idx=frame_idx)
                return
            except TypeError:
                monitor(flag, frame_idx)
                return

        for method_name in ("update_system_uninitialized", "set_system_uninitialized"):
            method = getattr(monitor, method_name, None)
            if not callable(method):
                continue
            try:
                method(flag=flag, frame_idx=frame_idx)
            except TypeError:
                method(flag)
            return

