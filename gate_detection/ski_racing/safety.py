"""Wave 3 degraded-safety scaffold for Track E."""

from __future__ import annotations

from typing import Dict, List, Optional


class SafetyMonitor:
    """
    Monitors per-frame BEV + detection signals and emits safety flags.

    Wave 3 scope:
    - EIS snap-vs-pan discriminator from `delta2_eis`
    - VP collapse trigger from `alpha_t`
    - Tier-3 fallback low-confidence trigger from detection `is_degraded`

    Wave 4 scope:
    - S* confidence-collapse trigger from decoder `s_star` when `score_valid=True`
    """

    def __init__(self, eis_threshold: float, stability_window: int, confidence_floor: float = -2.0):
        self.eis_threshold = float(eis_threshold)
        self.stability_window = int(stability_window)
        self.confidence_floor = float(confidence_floor)
        if self.stability_window < 0:
            raise ValueError("stability_window must be >= 0")

        self._history: List[Dict] = []
        self._by_frame_idx: Dict[int, Dict] = {}

        self._consecutive_spike_count = 0
        self._spike_frame_indices: List[int] = []
        self._degraded_latched = False
        self._clear_count = 0

    @property
    def history(self) -> List[Dict]:
        """Mutable per-frame output history (latest state per processed frame)."""
        return self._history

    def update(self, bev_frame: dict, detection_frame: dict) -> dict:
        frame_idx = self._resolve_frame_idx(bev_frame, detection_frame)
        alpha_t = float(bev_frame.get("alpha_t", 1.0))
        delta2_eis = float(bev_frame.get("delta2_eis", 0.0))

        result = {
            "frame_idx": frame_idx,
            "SYSTEM_UNINITIALIZED": self._is_system_uninitialized(bev_frame, detection_frame),
            "DEGRADED": False,
            "LOW_CONFIDENCE": False,
            "degraded_reason": None,
        }

        vp_collapse = alpha_t == 0.0
        if vp_collapse:
            result["DEGRADED"] = True
            result["degraded_reason"] = "vp_collapse"

        if self._has_tier3_active(detection_frame):
            result["LOW_CONFIDENCE"] = True
            if result["degraded_reason"] is None:
                result["degraded_reason"] = "tier3_active"

        self._store_output(result)

        # Track EIS spike runs and retroactively classify 1-2 frame bursts as snaps.
        if delta2_eis > self.eis_threshold:
            self._consecutive_spike_count += 1
            self._spike_frame_indices.append(frame_idx)
        elif self._consecutive_spike_count > 0:
            self._finalize_spike_run()

        # Stability latch: keep DEGRADED active for N clean frames before clearing.
        raw_degraded = bool(result["DEGRADED"])
        if raw_degraded:
            self._degraded_latched = True
            self._clear_count = 0
        elif self._degraded_latched:
            if self._clear_count < self.stability_window:
                result["DEGRADED"] = True
                if result["degraded_reason"] is None:
                    result["degraded_reason"] = "vp_collapse"
                self._clear_count += 1
            else:
                self._degraded_latched = False
                self._clear_count = 0

        return result

    def update_with_decoder(self, decoder_frame: dict) -> dict:
        """
        Process a decoder frame and check for S* confidence-collapse.

        Args:
            decoder_frame: dict with keys: frame_idx, state, score_valid, s_star, s_star_margin.
                Expected contract from Track F DECODER_API.md.

        Returns:
            dict with keys: frame_idx, LOW_CONFIDENCE, degraded_reason.
                Sets LOW_CONFIDENCE=True and degraded_reason='s_star_collapse'
                when score_valid=True AND s_star < confidence_floor.
        """
        frame_idx = int(decoder_frame.get("frame_idx", 0))
        score_valid = bool(decoder_frame.get("score_valid", False))
        s_star = float(decoder_frame.get("s_star", 0.0))

        result = {
            "frame_idx": frame_idx,
            "LOW_CONFIDENCE": False,
            "degraded_reason": None,
        }

        # Only check confidence collapse if score_valid=True
        if score_valid and s_star < self.confidence_floor:
            result["LOW_CONFIDENCE"] = True
            result["degraded_reason"] = "s_star_collapse"

        return result

    def flush(self) -> None:
        """Finalize a pending EIS run when the stream ends."""
        if self._consecutive_spike_count > 0:
            self._finalize_spike_run()

    def _finalize_spike_run(self) -> None:
        # 1-2 spikes: EIS snap. 3+ spikes: legitimate pan onset (suppressed).
        if 1 <= self._consecutive_spike_count <= 2:
            for idx in self._spike_frame_indices:
                frame_out = self._by_frame_idx.get(idx)
                if frame_out is None:
                    continue
                if frame_out["DEGRADED"] and frame_out["degraded_reason"] != "eis_snap":
                    continue
                frame_out["DEGRADED"] = True
                if frame_out["degraded_reason"] is None:
                    frame_out["degraded_reason"] = "eis_snap"
        else:
            for idx in self._spike_frame_indices:
                frame_out = self._by_frame_idx.get(idx)
                if frame_out is None:
                    continue
                if frame_out["degraded_reason"] == "eis_snap":
                    frame_out["DEGRADED"] = False
                    frame_out["degraded_reason"] = None

        self._consecutive_spike_count = 0
        self._spike_frame_indices = []

    def _resolve_frame_idx(self, bev_frame: dict, detection_frame: dict) -> int:
        for source in (bev_frame, detection_frame):
            value = source.get("frame_idx")
            if isinstance(value, int):
                return value
        return len(self._history)

    def _is_system_uninitialized(self, bev_frame: dict, detection_frame: dict) -> bool:
        initialized = None
        for source in (bev_frame, detection_frame):
            if "system_initialized" in source:
                initialized = bool(source["system_initialized"])
                break
        if initialized is None:
            return False
        return not initialized

    def _has_tier3_active(self, detection_frame: dict) -> bool:
        detections: Optional[List[dict]] = detection_frame.get("detections")
        if not detections:
            return False
        for det in detections:
            if det.get("is_degraded") is True:
                return True
            if det.get("base_fallback_tier") == 3:
                return True
        return False

    def _store_output(self, output: Dict) -> None:
        self._history.append(output)
        self._by_frame_idx[output["frame_idx"]] = output

