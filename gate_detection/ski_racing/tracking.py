"""
Skier tracking module.
Uses YOLOv8 with ByteTrack for robust multi-person tracking,
correctly identifying the racer among spectators and course workers.

Includes Kalman filter smoothing (Phase 3) for removing noise
from bounding-box-based position estimates.
"""
import cv2
import numpy as np
from pathlib import Path


class TrajectoryOutlierFilter:
    """
    Robust pre-smoothing outlier filter for 2D trajectories.

    Uses a sliding-window median and MAD test. Flagged points are replaced
    by interpolation between nearest inlier neighbors while preserving the
    original detections for debugging.
    """

    def __init__(self, window=5, mad_threshold=3.0, min_mad_px=1.0):
        self.window = max(3, int(window))
        if self.window % 2 == 0:
            self.window += 1
        self.mad_threshold = float(mad_threshold)
        self.min_mad_px = float(min_mad_px)

    def filter(self, trajectory_2d):
        """
        Args:
            trajectory_2d: List of {"frame", "x", "y", ...}

        Returns:
            (filtered_trajectory, diagnostics)
        """
        n = len(trajectory_2d)
        if n == 0:
            return trajectory_2d, {"outlier_count": 0, "outlier_frames": []}

        filtered = [dict(pt) for pt in trajectory_2d]
        half = self.window // 2
        is_outlier = [False] * n

        xs = np.array([float(pt["x"]) for pt in trajectory_2d], dtype=np.float64)
        ys = np.array([float(pt["y"]) for pt in trajectory_2d], dtype=np.float64)

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)

            win_x = xs[lo:hi]
            win_y = ys[lo:hi]
            if len(win_x) < 3:
                continue

            med_x = float(np.median(win_x))
            med_y = float(np.median(win_y))
            mad_x = float(np.median(np.abs(win_x - med_x)))
            mad_y = float(np.median(np.abs(win_y - med_y)))

            limit_x = self.mad_threshold * max(self.min_mad_px, mad_x)
            limit_y = self.mad_threshold * max(self.min_mad_px, mad_y)

            dx = abs(float(xs[i]) - med_x)
            dy = abs(float(ys[i]) - med_y)
            if dx > limit_x or dy > limit_y:
                is_outlier[i] = True

        outlier_frames = []
        for i, pt in enumerate(filtered):
            raw_x = float(pt["x"])
            raw_y = float(pt["y"])
            pt["raw_x"] = raw_x
            pt["raw_y"] = raw_y
            pt["is_outlier"] = bool(is_outlier[i])
            if is_outlier[i]:
                outlier_frames.append(int(pt["frame"]))

        inlier_indices = [i for i in range(n) if not is_outlier[i]]
        if not inlier_indices:
            return filtered, {
                "outlier_count": int(n),
                "outlier_frames": outlier_frames,
            }

        for i in range(n):
            if not is_outlier[i]:
                continue

            prev_idx = None
            next_idx = None

            j = i - 1
            while j >= 0:
                if not is_outlier[j]:
                    prev_idx = j
                    break
                j -= 1

            j = i + 1
            while j < n:
                if not is_outlier[j]:
                    next_idx = j
                    break
                j += 1

            if prev_idx is not None and next_idx is not None:
                f0 = float(filtered[prev_idx]["frame"])
                f1 = float(filtered[next_idx]["frame"])
                fk = float(filtered[i]["frame"])
                t = (fk - f0) / (f1 - f0) if f1 > f0 else 0.5
                x0, y0 = float(filtered[prev_idx]["x"]), float(filtered[prev_idx]["y"])
                x1, y1 = float(filtered[next_idx]["x"]), float(filtered[next_idx]["y"])
                filtered[i]["x"] = float(x0 + t * (x1 - x0))
                filtered[i]["y"] = float(y0 + t * (y1 - y0))
                c0 = float(filtered[prev_idx].get("confidence", 0.5))
                c1 = float(filtered[next_idx].get("confidence", 0.5))
                filtered[i]["confidence"] = float(0.5 * (c0 + c1))
            elif prev_idx is not None:
                filtered[i]["x"] = float(filtered[prev_idx]["x"])
                filtered[i]["y"] = float(filtered[prev_idx]["y"])
                filtered[i]["confidence"] = float(filtered[prev_idx].get("confidence", 0.5))
            elif next_idx is not None:
                filtered[i]["x"] = float(filtered[next_idx]["x"])
                filtered[i]["y"] = float(filtered[next_idx]["y"])
                filtered[i]["confidence"] = float(filtered[next_idx].get("confidence", 0.5))

        return filtered, {
            "outlier_count": int(len(outlier_frames)),
            "outlier_frames": outlier_frames,
        }


class KalmanSmoother:
    """
    Rauch-Tung-Striebel (RTS) smoother for 2D trajectory smoothing.

    State vector: [x, y, vx, vy]
    Measurement: [x, y] (noisy pixel coordinates from detector)

    Uses a forward Kalman filter pass followed by a backward RTS
    smoothing pass. The backward pass corrects the forward estimates
    using future information, which is critical at turn apexes where
    a forward-only filter would lag behind direction changes.

    Pure NumPy implementation — no external dependencies.

    Professor's feedback on over-smoothing:
        If process_noise (Q) is too low, the filter trusts its constant-
        velocity prediction too much and cuts slalom turn corners. For
        slalom (turns every 1-2s), use process_noise >= 2.0. For GS/DH
        (smoother arcs), 1.0-1.5 is fine.
    """

    # Discipline-tuned defaults: slalom needs higher Q to track rapid turns.
    # process_noise (sigma_a) is in pixels/s² and represents expected
    # acceleration standard deviation. This is fps-independent.
    # Slalom: rapid direction changes every 1-2s → high acceleration.
    # GS/DH: smoother arcs → lower acceleration expected.
    DISCIPLINE_DEFAULTS = {
        "slalom":       {"process_noise": 800.0, "measurement_noise": 5.0},
        "giant_slalom": {"process_noise": 400.0, "measurement_noise": 5.0},
        "downhill":     {"process_noise": 250.0, "measurement_noise": 5.0},
    }

    def __init__(self, process_noise=None, measurement_noise=None, fps=30.0,
                 discipline=None, innovation_gate=9.0,
                 low_conf_threshold=0.5, low_conf_r_multiplier=2.5):
        """
        Args:
            process_noise: Expected acceleration std dev (pixels/s²).
                          Higher = trust measurements more, lower = smoother.
                          If None, uses discipline default.
                          Typical values: 200-1000 for ski racing.
            measurement_noise: Expected detection noise std dev (pixels).
                              Typical bounding box center jitter is 3-10 px.
                              If None, uses discipline default.
            fps: Video frame rate for time delta.
            discipline: "slalom", "giant_slalom", or "downhill".
                       Used to auto-tune process_noise if not specified.
            innovation_gate: Mahalanobis gate threshold for innovation
                             (squared distance). 9.0 ≈ 3σ for 2D.
            low_conf_threshold: Detection confidence threshold where
                                measurement noise starts increasing.
            low_conf_r_multiplier: Maximum multiplier applied to
                                   measurement noise std-dev when confidence
                                   is near zero.
        """
        self.dt = 1.0 / fps
        self.discipline = discipline or "slalom"

        defaults = self.DISCIPLINE_DEFAULTS.get(self.discipline, self.DISCIPLINE_DEFAULTS["slalom"])
        self.process_noise = process_noise if process_noise is not None else defaults["process_noise"]
        self.measurement_noise = measurement_noise if measurement_noise is not None else defaults["measurement_noise"]
        self.innovation_gate = float(innovation_gate) if innovation_gate is not None else None
        self.low_conf_threshold = float(low_conf_threshold)
        self.low_conf_r_multiplier = float(max(1.0, low_conf_r_multiplier))

    def smooth(self, trajectory_2d):
        """
        Apply forward-backward RTS smoother to a 2D trajectory.

        Forward pass: standard Kalman filter (predict + update).
        Backward pass: RTS smoother refines estimates using future data.

        The backward pass is what prevents corner-cutting at turn apexes —
        it allows the filter to "look ahead" and adjust positions that a
        forward-only filter would lag on.

        Handles frame gaps by running multiple prediction steps.

        Args:
            trajectory_2d: List of {"frame": int, "x": float, "y": float}.

        Returns:
            Smoothed trajectory with same structure.
        """
        if len(trajectory_2d) < 2:
            return trajectory_2d

        dt = self.dt
        n = len(trajectory_2d)

        # State transition matrix F (constant velocity model)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Process noise covariance Q (discrete white noise acceleration model)
        # sigma_a is in pixels/s², making Q fps-independent.
        # Q = G * G^T * sigma_a^2 where G = [dt^2/2, dt^2/2, dt, dt]
        sigma_a = self.process_noise  # pixels/s²
        G = np.array([dt**2 / 2, dt**2 / 2, dt, dt], dtype=np.float64)
        Q = np.outer(G, G) * sigma_a**2
        # Zero out cross-axis terms (x-accel doesn't affect y)
        Q[0, 1] = Q[1, 0] = 0.0
        Q[0, 3] = Q[3, 0] = 0.0
        Q[1, 2] = Q[2, 1] = 0.0
        Q[2, 3] = Q[3, 2] = 0.0

        # Observation matrix H (we observe position only)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Measurement noise covariance R
        r = self.measurement_noise
        R = np.eye(2, dtype=np.float64) * r**2

        # Initialize state from first measurement
        x0 = trajectory_2d[0]["x"]
        y0 = trajectory_2d[0]["y"]

        # Estimate initial velocity from first two points
        if n >= 2:
            df = max(1, trajectory_2d[1]["frame"] - trajectory_2d[0]["frame"])
            vx0 = (trajectory_2d[1]["x"] - x0) / (df * dt)
            vy0 = (trajectory_2d[1]["y"] - y0) / (df * dt)
        else:
            vx0, vy0 = 0.0, 0.0

        state = np.array([x0, y0, vx0, vy0], dtype=np.float64)
        P = np.diag([r**2, r**2, (r * 2)**2, (r * 2)**2])

        # ─── Forward pass: standard Kalman filter ───
        # Store predicted and filtered states/covariances for RTS backward pass
        states_filt = []      # filtered state at each measurement
        covs_filt = []        # filtered covariance at each measurement
        states_pred = []      # predicted state (before update) at each measurement
        covs_pred = []        # predicted covariance (before update)
        F_steps = []          # effective F matrix used for each prediction step

        prev_frame = trajectory_2d[0]["frame"]
        gate_rejected = 0
        gate_considered = 0

        for i, pt in enumerate(trajectory_2d):
            frame = pt["frame"]
            z = np.array([pt["x"], pt["y"]], dtype=np.float64)

            # Handle frame gaps: compute effective F^gap and accumulated Q
            frame_gap = max(1, frame - prev_frame) if i > 0 else 0

            if frame_gap <= 1:
                F_eff = F
                Q_eff = Q
            else:
                # F^gap for multi-step prediction
                F_eff = np.linalg.matrix_power(F, frame_gap)
                # Accumulated process noise: sum of F^j Q F^jT for j=0..gap-1
                Q_eff = np.zeros_like(Q)
                F_pow = np.eye(4, dtype=np.float64)
                for _ in range(frame_gap):
                    Q_eff += F_pow @ Q @ F_pow.T
                    F_pow = F_pow @ F

            F_steps.append(F_eff)

            if i > 0:
                # Predict using effective transition
                state = F_eff @ state
                P = F_eff @ P @ F_eff.T + Q_eff

            # Store predicted (before update)
            states_pred.append(state.copy())
            covs_pred.append(P.copy())

            if i > 0:
                # Update with measurement (Mahalanobis innovation gating)
                y_innov = z - H @ state
                det_conf = float(pt.get("confidence", 1.0))
                conf_for_scale = float(np.clip(det_conf, 0.0, 1.0))
                r_scale = 1.0
                if conf_for_scale < self.low_conf_threshold and self.low_conf_threshold > 1e-6:
                    frac = (self.low_conf_threshold - conf_for_scale) / self.low_conf_threshold
                    r_scale = 1.0 + frac * (self.low_conf_r_multiplier - 1.0)
                R_eff = R * (r_scale ** 2)
                S = H @ P @ H.T + R_eff
                use_update = True
                gate_considered += 1

                if self.innovation_gate is not None:
                    try:
                        S_inv = np.linalg.inv(S)
                        mahal_sq = float(y_innov.T @ S_inv @ y_innov)
                        if mahal_sq > self.innovation_gate:
                            use_update = False
                            gate_rejected += 1
                    except np.linalg.LinAlgError:
                        use_update = False

                if use_update:
                    try:
                        K = P @ H.T @ np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        K = np.zeros((4, 2))
                    state = state + K @ y_innov
                    P = (np.eye(4) - K @ H) @ P

            # Store filtered (after update)
            states_filt.append(state.copy())
            covs_filt.append(P.copy())
            prev_frame = frame

        # ─── Backward pass: RTS smoother ───
        # Uses the effective transition matrix from each forward step
        states_smooth = [None] * n
        states_smooth[n - 1] = states_filt[n - 1]

        for k in range(n - 2, -1, -1):
            P_filt = covs_filt[k]
            P_pred_next = covs_pred[k + 1]
            F_k = F_steps[k + 1]  # Transition used from step k to k+1

            # RTS gain: G_k = P_k|k * F_k^T * (P_{k+1|k})^{-1}
            try:
                G = P_filt @ F_k.T @ np.linalg.inv(P_pred_next)
            except np.linalg.LinAlgError:
                G = np.zeros((4, 4))

            states_smooth[k] = states_filt[k] + G @ (states_smooth[k + 1] - states_pred[k + 1])

        # Build output
        smoothed = []
        for i, pt in enumerate(trajectory_2d):
            smoothed.append({
                "frame": pt["frame"],
                "x": float(states_smooth[i][0]),
                "y": float(states_smooth[i][1]),
            })

        if gate_rejected > 0:
            print(f"  ✓ Innovation gating: skipped {gate_rejected}/{gate_considered} "
                  f"updates (mahal_sq > {self.innovation_gate:.1f})")

        # ── Over-smoothing diagnostic (Prof. feedback Phase 3) ──
        # If Q is too low, the filter cuts corners on sharp turns.
        # Compare lateral deviation at turn apexes (x-reversal points)
        # between raw measurements and smoothed output.
        self._check_over_smoothing(trajectory_2d, smoothed)

        return smoothed

    def _check_over_smoothing(self, raw, smoothed):
        """
        Detect if smoothing is cutting corners on sharp turns.

        Finds turn apexes (points where X-direction reverses) and compares
        lateral deviation between raw and smoothed paths. If the smoothed
        path consistently reduces apex deviation by >30%, the filter is
        probably too stiff (process_noise too low).

        Professor's caution: "the Kalman filter will be too stiff — it will
        cut the corners of sharp slalom turns, reporting a turn radius of
        15m when the skier actually turned at 8m."
        """
        if len(raw) < 10:
            return

        # Find turn apexes: points where dx changes sign
        raw_x = [p["x"] for p in raw]
        smooth_x = [p["x"] for p in smoothed]

        apex_indices = []
        for i in range(1, len(raw_x) - 1):
            dx_prev = raw_x[i] - raw_x[i - 1]
            dx_next = raw_x[i + 1] - raw_x[i]
            if dx_prev * dx_next < 0:  # sign change = turn apex
                apex_indices.append(i)

        if len(apex_indices) < 2:
            return

        # Measure how much the smoothed path reduces lateral excursion
        raw_y = [p["y"] for p in raw]

        # Use a wider local baseline: average x of neighbors ±10 frames.
        # This is more robust than ±5 because noise at individual frames
        # can inflate the raw deviation, leading to false positives.
        deviations_raw = []
        deviations_smooth = []
        window = 10

        for idx in apex_indices:
            lo = max(0, idx - window)
            hi = min(len(raw_x) - 1, idx + window)
            # Exclude the apex itself from the baseline
            baseline_x = [raw_x[j] for j in range(lo, hi + 1) if j != idx]
            if not baseline_x:
                continue
            local_avg_x = np.mean(baseline_x)
            deviations_raw.append(abs(raw_x[idx] - local_avg_x))
            deviations_smooth.append(abs(smooth_x[idx] - local_avg_x))

        avg_raw = np.mean(deviations_raw) if deviations_raw else 0
        avg_smooth = np.mean(deviations_smooth) if deviations_smooth else 0

        # Only warn if: (1) meaningful lateral movement at apexes (>10px),
        # and (2) smoothing reduces apex deviation by >50%.
        # 50% threshold is higher than 30% because the RTS smoother naturally
        # adjusts turn peaks slightly, and raw apex deviations include noise.
        if avg_raw > 10.0:
            reduction = 1.0 - avg_smooth / avg_raw if avg_raw > 0 else 0
            if reduction > 0.50:
                print(f"  ⚠️  Over-smoothing detected: turn apex deviation reduced by "
                      f"{reduction * 100:.0f}% ({avg_raw:.1f}px → {avg_smooth:.1f}px). "
                      f"The filter may be cutting corners on sharp turns (Q={self.process_noise}). "
                      f"Increase process_noise or verify with demo video overlay.")


class SkierTracker:
    """
    Track the racing skier through video frames.
    Uses ByteTrack for persistent ID tracking across frames,
    with fallback to temporal consistency when ByteTrack loses the target.
    """

    def __init__(self, model_name=None):
        """
        Args:
            model_name: YOLO model for person detection.
        """
        from ultralytics import YOLO  # Lazy import: not needed for KalmanSmoother

        if model_name is None:
            # Prefer local weights to avoid network downloads
            if Path("models/yolov8s.pt").exists():
                model_name = "models/yolov8s.pt"
            elif Path("models/yolov8n.pt").exists():
                model_name = "models/yolov8n.pt"
            else:
                model_name = "yolov8n.pt"
        self.model = YOLO(model_name)
        self._tracker_cfg = self._resolve_tracker_config_path()
        self._last_tracking_stats = {}

    @staticmethod
    def _resolve_tracker_config_path():
        """Resolve local ByteTrack config path with tuned thresholds."""
        project_root = Path(__file__).resolve().parent.parent
        cfg_path = project_root / "configs" / "bytetrack_tuned.yaml"
        if cfg_path.exists():
            return str(cfg_path)
        return "bytetrack.yaml"

    def get_last_tracking_stats(self):
        return dict(self._last_tracking_stats) if self._last_tracking_stats else {}

    def track_video(
        self,
        video_path,
        method="bytetrack",
        frame_stride=1,
        max_frames=None,
        max_jump=None,
        conf=0.25,
        gates=None,
    ):
        """
        Track skier through an entire video.

        Args:
            video_path: Path to video file.
            method: "bytetrack" (recommended) or "temporal" (fallback).
            frame_stride: Process every Nth frame (temporal mode only).
            max_frames: Optional max number of frames to process from start.
            max_jump: Max pixel jump allowed between frames (temporal mode only).
            conf: Person detection confidence threshold.
            gates: Optional list of gate detections (static). When provided,
                tracking is biased toward the course centerline implied by gates
                to reduce switching to spectators/course workers.

        Returns:
            List of trajectory points: [{"frame": int, "x": float, "y": float}, ...]
        """
        if method == "bytetrack":
            try:
                trajectory = self._track_with_bytetrack_reassociate(
                    video_path,
                    conf=conf,
                    max_jump=max_jump,
                    gates=gates,
                )
                return trajectory
            except ModuleNotFoundError as e:
                # HARD FAIL: Do NOT silently fall back to temporal tracking.
                # The temporal tracker picks the person closest to screen center,
                # which is frequently a spectator or coach rather than the skier.
                # This produces plausible-looking but completely wrong trajectories
                # that are very hard to detect downstream.
                #
                # Fix: pip install lap>=0.5.12
                # If lap fails to build: pip install lapjv
                raise RuntimeError(
                    f"\n\n{'='*60}\n"
                    f"BYTETRACK DEPENDENCY MISSING: {e}\n"
                    f"{'='*60}\n"
                    f"ByteTrack requires the 'lap' package for linear assignment.\n"
                    f"Without it the pipeline cannot track skiers reliably.\n\n"
                    f"Fix with ONE of the following:\n"
                    f"  pip install lap>=0.5.12\n"
                    f"  pip install lapjv          (if lap fails to build)\n\n"
                    f"Then re-run: python scripts/check_env.py\n"
                    f"{'='*60}\n"
                ) from e
            except Exception as e:
                print(f"⚠️  ByteTrack failed ({e}). Falling back to temporal tracking.")
                trajectory = self._track_with_temporal_consistency(
                    video_path,
                    frame_stride=frame_stride,
                    max_frames=max_frames,
                    max_jump=max_jump,
                    conf=conf,
                )
                self._last_tracking_stats["selected_method"] = "temporal"
                self._last_tracking_stats["failure_reason"] = f"Exception: {e}"
                return trajectory
        else:
            trajectory = self._track_with_temporal_consistency(
                video_path,
                frame_stride=frame_stride,
                max_frames=max_frames,
                max_jump=max_jump,
                conf=conf,
            )
            self._last_tracking_stats["selected_method"] = "temporal"
            return trajectory

    @staticmethod
    def _build_course_centerline(gates):
        """
        Build a simple course centerline x(y) function from static gate detections.

        Uses gate center_x vs base_y, linearly interpolated in y.
        Returns None when there isn't enough usable gate data.
        """
        if not gates:
            return None
        pts = []
        for g in gates:
            if not isinstance(g, dict):
                continue
            if "center_x" not in g or "base_y" not in g:
                continue
            try:
                pts.append((float(g["base_y"]), float(g["center_x"])))
            except Exception:
                continue
        if len(pts) < 2:
            return None
        pts.sort(key=lambda t: t[0])
        ys = np.array([p[0] for p in pts], dtype=np.float64)
        xs = np.array([p[1] for p in pts], dtype=np.float64)

        # Drop nearly-duplicate y points to keep interpolation stable.
        keep = [0]
        for i in range(1, len(ys)):
            if abs(float(ys[i]) - float(ys[keep[-1]])) >= 2.0:
                keep.append(i)
        ys = ys[keep]
        xs = xs[keep]
        if len(ys) < 2:
            return None

        def x_at_y(y):
            y = float(y)
            if y <= float(ys[0]):
                return float(xs[0])
            if y >= float(ys[-1]):
                return float(xs[-1])
            j = int(np.searchsorted(ys, y, side="right"))
            y0, y1 = float(ys[j - 1]), float(ys[j])
            x0, x1 = float(xs[j - 1]), float(xs[j])
            t = (y - y0) / (y1 - y0) if y1 > y0 else 0.5
            return float(x0 + t * (x1 - x0))

        def lateral_error_px(x, y):
            return abs(float(x) - x_at_y(y))

        return {"x_at_y": x_at_y, "lateral_error_px": lateral_error_px}

    def _track_with_bytetrack_reassociate(
        self,
        video_path,
        conf=0.25,
        max_jump=None,
        gates=None,
        max_gap=30,
        reacquire_gap=15,
    ):
        """
        Use YOLOv8's built-in ByteTrack detections but reassociate
        targets frame-to-frame to avoid ID switches truncating the run.
        """
        trajectory = []
        centerline = self._build_course_centerline(gates)
        gate_corridor_max_ratio = 0.35
        gate_lateral_weight = 0.70

        # Run tracking with persistence
        results = self.model.track(
            source=video_path,
            classes=[0],  # person class only
            persist=True,
            tracker=self._tracker_cfg,
            conf=conf,
            verbose=False,
            stream=True,
        )

        last_pos = None
        last_vel = (0.0, 0.0)
        last_frame = None
        last_area = None
        last_track_id = None
        missing = 0
        track_id_switches = 0
        track_id_changes_observed = 0
        observed_frames = 0
        track_id_counts = {}
        primary_track_id = None

        for frame_idx, result in enumerate(results):
            observed_frames += 1
            if result.boxes is None or len(result.boxes) == 0:
                missing += 1
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None

            h, w = result.orig_shape[:2] if hasattr(result, "orig_shape") else (None, None)
            if w is None or h is None:
                # Fallback when orig_shape is missing
                h, w = 720, 1280

            detections = []
            for i, box in enumerate(boxes):
                x = float((box[0] + box[2]) / 2)
                y = float(box[3])
                area = float((box[2] - box[0]) * (box[3] - box[1]))
                c = float(confs[i]) if confs is not None else 0.5
                det_id = int(ids[i]) if ids is not None and not np.isnan(ids[i]) else None
                detections.append({
                    "x": x,
                    "y": y,
                    "area": area,
                    "confidence": c,
                    "track_id": det_id,
                })

            if not detections:
                missing += 1
                continue

            # Compute dynamic max jump
            if max_jump is None:
                base = 0.35 * min(w, h)
                max_jump_px = min(base, 0.75 * max(w, h))
            else:
                max_jump_px = float(max_jump)

            if centerline is not None and last_pos is not None:
                corridor_max_px = float(gate_corridor_max_ratio) * float(w)
                in_course = [
                    det for det in detections
                    if centerline["lateral_error_px"](det["x"], det["y"]) <= corridor_max_px
                ]
                if in_course:
                    detections = in_course
                else:
                    # No plausible on-course detections: treat as missing rather than
                    # latching onto spectators far from the gate corridor.
                    missing += 1
                    if missing > max_gap:
                        last_pos = None
                        last_vel = (0.0, 0.0)
                        last_frame = None
                        last_area = None
                        last_track_id = None
                    continue

            chosen = None
            if last_pos is None:
                # First frame: gate-guided initialization when available.
                if centerline is not None:
                    corridor_max_px = float(gate_corridor_max_ratio) * float(w)
                    best = None
                    best_score = float("inf")
                    for det in detections:
                        lat = centerline["lateral_error_px"](det["x"], det["y"])
                        if lat > corridor_max_px:
                            continue
                        prominence = (det["area"] ** 0.5) * det["confidence"]
                        # Lower is better: stay on-course, prefer confident/large targets.
                        score = float(lat) - 1.25 * float(prominence)
                        if score < best_score:
                            best_score = score
                            best = det
                    chosen = best

                if chosen is None:
                    # Fallback: choose detection closest to center
                    cx, cy = w / 2.0, h / 2.0
                    best = None
                    best_dist = float("inf")
                    for det in detections:
                        dist = ((det["x"] - cx) ** 2 + (det["y"] - cy) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best = det
                    chosen = best
            else:
                gap = max(1, frame_idx - (last_frame if last_frame is not None else frame_idx))
                pred_x = last_pos[0] + last_vel[0] * gap
                pred_y = last_pos[1] + last_vel[1] * gap
                corridor_max_px = (
                    float(gate_corridor_max_ratio) * float(w)
                    if centerline is not None
                    else None
                )

                if last_track_id is not None:
                    for det in detections:
                        if det.get("track_id") != last_track_id:
                            continue
                        if corridor_max_px is not None:
                            lat = centerline["lateral_error_px"](det["x"], det["y"])
                            if lat > corridor_max_px:
                                continue
                        dist_same = ((det["x"] - pred_x) ** 2 + (det["y"] - pred_y) ** 2) ** 0.5
                        if dist_same <= max_jump_px * gap:
                            chosen = det
                            break

                if chosen is not None:
                    pass
                else:
                    best = None
                    best_dist = float("inf")
                    best_any = None
                    best_any_dist = float("inf")
                    for det in detections:
                        dist = ((det["x"] - pred_x) ** 2 + (det["y"] - pred_y) ** 2) ** 0.5
                        in_course = True
                        if centerline is not None:
                            lat = centerline["lateral_error_px"](det["x"], det["y"])
                            dist += gate_lateral_weight * lat
                            # Soft guardrail: if we have on-course candidates, avoid extremely off-course ones.
                            in_course = lat <= 0.55 * w
                        det_id = det.get("track_id")
                        if primary_track_id is not None and det_id is not None and det_id != primary_track_id:
                            dist += 150.0
                        # Penalize implausible uphill jumps (helps avoid switching to spectators).
                        if det["y"] < last_pos[1] - 0.08 * h:
                            dist += 10_000.0
                        if dist < best_any_dist:
                            best_any_dist = dist
                            best_any = det
                        if dist < best_dist:
                            if in_course:
                                best_dist = dist
                                best = det

                    if best is None:
                        best = best_any
                        best_dist = best_any_dist

                    if best is not None and best_dist <= max_jump_px * gap:
                        chosen = best
                    elif gap > reacquire_gap:
                        # Reacquire: favor on-course detections and predicted continuity.
                        best = None
                        best_score = float("inf")
                        best_any = None
                        best_any_score = float("inf")
                        for det in detections:
                            prominence = (det["area"] ** 0.5) * det["confidence"]
                            dist_pred = ((det["x"] - pred_x) ** 2 + (det["y"] - pred_y) ** 2) ** 0.5
                            in_course = True
                            lat = 0.0
                            if centerline is not None:
                                lat = centerline["lateral_error_px"](det["x"], det["y"])
                                in_course = lat <= 0.55 * w
                            score = dist_pred + gate_lateral_weight * lat - 1.5 * prominence
                            det_id = det.get("track_id")
                            if primary_track_id is not None and det_id is not None and det_id != primary_track_id:
                                score += 150.0
                            if det["y"] < last_pos[1] - 0.08 * h:
                                score += 10_000.0
                            if score < best_any_score:
                                best_any_score = score
                                best_any = det
                            if score < best_score:
                                if in_course:
                                    best_score = score
                                    best = det

                        if best is None:
                            best = best_any
                        if best is not None:
                            dist = ((best["x"] - pred_x) ** 2 + (best["y"] - pred_y) ** 2) ** 0.5
                            within_pred = dist <= max(200.0, 0.30 * max(w, h) * min(gap, 10))
                            if last_area is not None and last_area > 1e-6:
                                area_ratio = best["area"] / last_area
                                within_size = 0.5 <= area_ratio <= 1.5
                            else:
                                within_size = True
                            if within_pred and within_size:
                                chosen = best
                    else:
                        missing += 1
                        if missing > max_gap:
                            last_pos = None
                            last_vel = (0.0, 0.0)
                            last_frame = None
                            last_area = None
                            last_track_id = None
                        continue

            if chosen is None:
                missing += 1
                continue
            if centerline is not None:
                corridor_max_px = float(gate_corridor_max_ratio) * float(w)
                # If there are any plausible on-course detections this frame,
                # reject off-course picks (prevents latching onto spectators).
                has_in_course = any(
                    centerline["lateral_error_px"](det["x"], det["y"]) <= corridor_max_px
                    for det in detections
                )
                if has_in_course:
                    lat = centerline["lateral_error_px"](chosen["x"], chosen["y"])
                    if lat > corridor_max_px:
                        missing += 1
                        if missing > max_gap:
                            last_pos = None
                            last_vel = (0.0, 0.0)
                            last_frame = None
                            last_area = None
                            last_track_id = None
                        continue

            x, y = chosen["x"], chosen["y"]
            gap_to_prev = max(1, frame_idx - last_frame) if last_frame is not None else 1
            if last_pos is not None and last_frame is not None:
                gap = max(1, frame_idx - last_frame)
                last_vel = ((x - last_pos[0]) / gap, (y - last_pos[1]) / gap)
            current_track_id = chosen.get("track_id")
            if (last_track_id is not None and current_track_id is not None
                    and current_track_id != last_track_id):
                track_id_changes_observed += 1
                jump_px = ((x - last_pos[0]) ** 2 + (y - last_pos[1]) ** 2) ** 0.5 if last_pos is not None else 0.0
                # Count only continuity-breaking switches.
                if gap_to_prev <= 2 and jump_px > 500.0:
                    track_id_switches += 1
            last_pos = (x, y)
            last_frame = frame_idx
            last_area = float(chosen["area"])
            last_track_id = current_track_id
            missing = 0
            if current_track_id is not None:
                track_id_counts[current_track_id] = track_id_counts.get(current_track_id, 0) + 1
                primary_track_id = max(track_id_counts, key=track_id_counts.get)

            trajectory.append({
                "frame": frame_idx,
                "x": x,
                "y": y,
                "confidence": float(chosen["confidence"]),
                "track_id": current_track_id,
                "bbox_area": float(chosen["area"]),
            })

        print(f"✓ ByteTrack reassociation: tracked {len(trajectory)} frames")

        # If coverage is too low, fall back to temporal tracking
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:
            total_frames = 0

        bytetrack_valid_frames = len(trajectory)
        coverage = (bytetrack_valid_frames / total_frames) if total_frames > 0 else 0.0
        self._last_tracking_stats = {
            "method_requested": "bytetrack",
            "selected_method": "bytetrack",
            "total_frames": int(total_frames) if total_frames > 0 else int(observed_frames),
            "frames_with_valid_track": int(bytetrack_valid_frames),
            "bytetrack_coverage": float(coverage),
            "frames_fallback_used": 0,
            "fallback_used": False,
            "track_id_switches": int(track_id_switches),
            "track_id_changes_observed": int(track_id_changes_observed),
            "tracker_config": self._tracker_cfg,
        }

        if total_frames > 0 and coverage < 0.4:
            print(f"⚠️  Low ByteTrack coverage ({coverage:.0%}). Falling back to temporal tracking.")
            fallback_trajectory = self._track_with_temporal_consistency(
                video_path,
                frame_stride=1,
                max_frames=None,
                max_jump=max_jump,
                conf=conf,
                update_stats=False,
            )
            self._last_tracking_stats["selected_method"] = "temporal_fallback"
            self._last_tracking_stats["frames_fallback_used"] = int(len(fallback_trajectory))
            self._last_tracking_stats["fallback_used"] = True
            return fallback_trajectory

        return trajectory

    def _track_with_temporal_consistency(
        self,
        video_path,
        frame_stride=1,
        max_frames=None,
        max_jump=None,
        conf=0.25,
        update_stats=True,
    ):
        """
        Fallback tracking using temporal consistency.
        Tracks the person closest to the previous frame's position,
        with a maximum jump distance to prevent tracking switches.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        trajectory = []
        prev_position = None
        frame_num = 0
        # pixels - racer shouldn't teleport between frames
        max_jump_px = max_jump

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_num >= max_frames:
                break
            if frame_stride > 1 and (frame_num % frame_stride != 0):
                frame_num += 1
                continue

            if max_jump_px is None:
                h, w = frame.shape[:2]
                base = 0.35 * min(w, h)
                max_jump_px = min(base * frame_stride, 0.75 * max(w, h))

            results = self.model(frame, classes=[0], conf=conf, verbose=False)

            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None

                if prev_position is not None:
                    # Find detection closest to previous position
                    best_box = None
                    min_dist = float("inf")

                    for i, box in enumerate(boxes):
                        x = (box[0] + box[2]) / 2
                        y = box[3]
                        dist = ((x - prev_position[0]) ** 2 + (y - prev_position[1]) ** 2) ** 0.5

                        if dist < max_jump_px and dist < min_dist:
                            min_dist = dist
                            c = float(confs[i]) if confs is not None else 0.5
                            area = float((box[2] - box[0]) * (box[3] - box[1]))
                            best_box = (float(x), float(y), c, area)

                    if best_box:
                        prev_position = (best_box[0], best_box[1])
                        trajectory.append({
                            "frame": frame_num,
                            "x": best_box[0],
                            "y": best_box[1],
                            "confidence": float(best_box[2]),
                            "track_id": None,
                            "bbox_area": float(best_box[3]),
                        })
                else:
                    # First frame: pick center-most person (racer usually centered)
                    frame_center_x = frame.shape[1] / 2
                    best_box = None
                    min_dist = float("inf")

                    for i, box in enumerate(boxes):
                        x = (box[0] + box[2]) / 2
                        y = box[3]
                        dist = abs(x - frame_center_x)

                        if dist < min_dist:
                            min_dist = dist
                            c = float(confs[i]) if confs is not None else 0.5
                            area = float((box[2] - box[0]) * (box[3] - box[1]))
                            best_box = (float(x), float(y), c, area)

                    if best_box:
                        prev_position = (best_box[0], best_box[1])
                        trajectory.append({
                            "frame": frame_num,
                            "x": best_box[0],
                            "y": best_box[1],
                            "confidence": float(best_box[2]),
                            "track_id": None,
                            "bbox_area": float(best_box[3]),
                        })

            frame_num += 1

        cap.release()
        if update_stats:
            self._last_tracking_stats = {
                "method_requested": "temporal",
                "selected_method": "temporal",
                "total_frames": int(frame_num),
                "frames_with_valid_track": int(len(trajectory)),
                "bytetrack_coverage": 0.0,
                "frames_fallback_used": int(frame_num),
                "fallback_used": False,
                "track_id_switches": 0,
                "track_id_changes_observed": 0,
            }
        return trajectory

    def get_video_info(self, video_path):
        """Get basic video info for downstream processing."""
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info


# ---------------------------------------------------------------------------
# v2.1 additions — VFR-aware gate tracker ported from Track D (Wave 3)
# Manager-applied 2026-02-19 from:
#   tracks/D_tracking_outlier/scripts/vfr_bev_tracker.py
# ---------------------------------------------------------------------------

import math as _math_t
from dataclasses import dataclass as _dataclass
from typing import Dict as _Dict, List as _List, Optional as _Optional
from typing import Sequence as _Sequence, Tuple as _Tuple


def _project_point_h(H: np.ndarray, x: float, y: float) -> _Tuple[float, float]:
    """Apply a 3x3 homography to a 2D point."""
    p = np.array([x, y, 1.0], dtype=np.float64)
    out = H @ p
    if abs(out[2]) < 1e-12:
        return float(x), float(y)
    return float(out[0] / out[2]), float(out[1] / out[2])


def _project_bbox_to_bev(H: np.ndarray, bbox_xyxy: _Sequence[float]) -> _Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    corners = [
        _project_point_h(H, x1, y1),
        _project_point_h(H, x2, y1),
        _project_point_h(H, x2, y2),
        _project_point_h(H, x1, y2),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _iou_xyxy(
    a: _Tuple[float, float, float, float],
    b: _Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else float(inter / denom)


@_dataclass
class GateObservation:
    """Per-detection input to BEVByteTracker."""
    frame_idx: int
    detection_id: str
    conf_class: float
    is_degraded: bool
    class_label: str
    geom_ok: bool
    bev_x: float
    bev_y: float
    bev_bbox: _Tuple[float, float, float, float]
    scale_s: float
    aspect_ratio: _Optional[float]
    colour_hist: _Optional[np.ndarray]
    image_base_x: float
    image_base_y: float


@_dataclass
class GateFrameTrack:
    """Per-track output from BEVByteTracker for one frame."""
    track_id: int
    bev_x: float
    bev_y: float
    bev_vx: float
    bev_vy: float
    innovation_magnitude: _Optional[float]
    frames_since_observation: int
    base_px: _Dict[str, float]


class _GateKalmanTrack:
    """
    6-state Kalman track for a single gate in BEV space.
    State: [x, y, vx, vy, s, ds] where s = sqrt(bbox area).
    Uses PTS-driven delta_t (VFR fix) rather than 1/fps_nominal.
    """

    def __init__(self, track_id: int, obs: GateObservation, frame_idx: int):
        self.track_id = int(track_id)
        self.x = np.array(
            [obs.bev_x, obs.bev_y, 0.0, 0.0, obs.scale_s, 0.0], dtype=np.float64
        )
        self.P = np.diag([1.0, 1.0, 10.0, 10.0, 1.0, 10.0]).astype(np.float64)
        self.last_frame_idx = int(frame_idx)
        self.frames_since_observation = 0
        self.hits = 1
        self.age = 1
        self.innovation_magnitude: _Optional[float] = None
        self.aspect_ratio = obs.aspect_ratio if obs.aspect_ratio is not None else 1.0
        self.colour_hist = obs.colour_hist

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _Q(dt: float, s_pos: float = 0.6, s_scale: float = 0.25) -> np.ndarray:
        q = np.zeros((6, 6), dtype=np.float64)
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        bp = np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]], dtype=np.float64) * s_pos ** 2
        bs = np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]], dtype=np.float64) * s_scale ** 2
        q[np.ix_([0, 2], [0, 2])] = bp
        q[np.ix_([1, 3], [1, 3])] = bp
        q[np.ix_([4, 5], [4, 5])] = bs
        return q

    @staticmethod
    def _H_mat() -> np.ndarray:
        return np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

    def predict(self, dt: float) -> None:
        dt = max(1e-6, float(dt))
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.frames_since_observation += 1
        self.age += 1

    def _innovation(
        self, obs: GateObservation, degraded_boost: float = 4.0
    ) -> _Tuple[np.ndarray, np.ndarray, float]:
        H = self._H_mat()
        z = np.array([obs.bev_x, obs.bev_y, obs.scale_s], dtype=np.float64)
        y = z - H @ self.x
        r_xy = 0.2 * (degraded_boost if obs.is_degraded else 1.0)
        r_s = 0.2 * (degraded_boost * 0.75 if obs.is_degraded else 1.0)
        R = np.diag([r_xy ** 2, r_xy ** 2, r_s ** 2]).astype(np.float64)
        S = H @ self.P @ H.T + R
        try:
            mahal = float(_math_t.sqrt(max(0.0, float(y.T @ np.linalg.inv(S) @ y))))
        except np.linalg.LinAlgError:
            mahal = float("inf")
        return y, S, mahal

    def mahalanobis(self, obs: GateObservation, degraded_boost: float = 4.0) -> float:
        return self._innovation(obs, degraded_boost)[2]

    def update(self, obs: GateObservation, degraded_boost: float = 4.0) -> float:
        H = self._H_mat()
        y, S, mahal = self._innovation(obs, degraded_boost)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((6, 3), dtype=np.float64)
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float64) - K @ H) @ self.P
        self.frames_since_observation = 0
        self.hits += 1
        self.innovation_magnitude = float(mahal)
        if obs.aspect_ratio is not None:
            self.aspect_ratio = float(obs.aspect_ratio)
        if obs.colour_hist is not None:
            self.colour_hist = obs.colour_hist
        self.last_frame_idx = obs.frame_idx
        return float(mahal)

    def predicted_bbox(self) -> _Tuple[float, float, float, float]:
        s = max(1e-4, float(self.x[4]))
        ar = max(1e-3, float(self.aspect_ratio or 1.0))
        w = s / _math_t.sqrt(ar)
        h = s * _math_t.sqrt(ar)
        cx, cy = float(self.x[0]), float(self.x[1])
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class BEVByteTracker:
    """
    ByteTrack-style gate tracker operating in Topological BEV space (v2.1).

    Key differences from the legacy SkierTracker:
      - Operates on gate detections (not skier bounding boxes)
      - Uses PTS-derived delta_t_s (VFR fix) for Kalman prediction
      - Inflates measurement noise when is_degraded=True (Tier-3 fallback)
      - Two-pass association: Pass 1 = Mahalanobis + appearance on high-conf
        detections; Pass 2 = IoU rescue on low-conf + unmatched high-conf
      - Down-weights appearance cost in flat-light conditions

    Usage::

        tracker = BEVByteTracker()
        for frame in frames:
            obs = build_observations(frame)       # list of GateObservation
            tracks = tracker.step(
                frame_idx=frame["frame_idx"],
                delta_t_s=frame["delta_t_s"],     # from PTS sidecar
                fps_nominal=frame["fps_nominal"],
                observations=obs,
                condition_light=frame.get("light", "normal"),
                H_inv_for_output=H_inv,           # optional: back-project BEV→image
            )
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        max_lost: int = 30,
        maha_gate: float = 14.0,
        degraded_boost: float = 4.0,
    ):
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.max_lost = int(max_lost)
        self.maha_gate = float(maha_gate)
        self.degraded_boost = float(degraded_boost)
        self._tracks: _List[_GateKalmanTrack] = []
        self._next_id = 1

    @staticmethod
    def _appearance_cost(track: _GateKalmanTrack, obs: GateObservation) -> float:
        costs: _List[float] = []
        if track.colour_hist is not None and obs.colour_hist is not None:
            a = track.colour_hist.astype(np.float64)
            b = obs.colour_hist.astype(np.float64)
            if a.size == b.size > 0:
                costs.append(float(0.5 * np.sum(np.abs(a / (np.sum(a) + 1e-9) - b / (np.sum(b) + 1e-9)))))
        if track.aspect_ratio is not None and obs.aspect_ratio is not None:
            costs.append(float(min(2.0, abs(float(obs.aspect_ratio) - float(track.aspect_ratio)) / (abs(float(track.aspect_ratio)) + 1e-6))))
        return float(sum(costs) / len(costs)) if costs else 0.0

    @staticmethod
    def _greedy_assign(
        cost_pairs: _List[_Tuple[float, int, int]],
    ) -> _List[_Tuple[int, int]]:
        seen_t, seen_d, matches = set(), set(), []
        for _, ti, di in sorted(cost_pairs):
            if ti not in seen_t and di not in seen_d:
                seen_t.add(ti)
                seen_d.add(di)
                matches.append((ti, di))
        return matches

    def step(
        self,
        frame_idx: int,
        delta_t_s: float,
        fps_nominal: float,
        observations: _Sequence[GateObservation],
        condition_light: str = "normal",
        H_inv_for_output: _Optional[np.ndarray] = None,
    ) -> _List[GateFrameTrack]:
        """
        Process one frame.

        Args:
            delta_t_s:        Per-frame PTS delta from Track A sidecar (VFR fix).
            fps_nominal:      Fallback if delta_t_s <= 0.
            H_inv_for_output: Inverse homography to back-project BEV positions
                              to image coordinates for base_px output.
        """
        dt = float(delta_t_s)
        if dt <= 0.0:
            dt = 1.0 / max(1e-6, float(fps_nominal))

        for tr in self._tracks:
            tr.predict(dt)

        high = [o for o in observations if o.conf_class >= self.high_thresh]
        low  = [o for o in observations if self.low_thresh <= o.conf_class < self.high_thresh]
        app_w = 0.05 if str(condition_light).lower() == "flat" else 0.2

        # Pass 1: Mahalanobis + appearance on high-conf detections
        p1_costs: _List[_Tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            for di, obs in enumerate(high):
                mahal = tr.mahalanobis(obs, self.degraded_boost)
                if not _math_t.isfinite(mahal) or mahal > self.maha_gate:
                    continue
                penalty = 0.15 if not obs.geom_ok else 0.0
                p1_costs.append((0.8 * mahal + app_w * self._appearance_cost(tr, obs) + penalty, ti, di))

        p1 = self._greedy_assign(p1_costs)
        matched_t = {ti for ti, _ in p1}
        matched_h = {di for _, di in p1}

        for ti, di in p1:
            self._tracks[ti].update(high[di], self.degraded_boost)

        # Pass 2: IoU rescue on low-conf + unmatched high-conf
        p2_pool = low + [o for di, o in enumerate(high) if di not in matched_h]
        unmatched_t = [ti for ti in range(len(self._tracks)) if ti not in matched_t]
        p2_costs: _List[_Tuple[float, int, int]] = []
        for ui, ti in enumerate(unmatched_t):
            pred = self._tracks[ti].predicted_bbox()
            for di, obs in enumerate(p2_pool):
                ov = _iou_xyxy(pred, obs.bev_bbox)
                if ov > 0.0:
                    p2_costs.append((1.0 - ov, ui, di))

        p2_matched_pool: set = set()
        for ui, di in self._greedy_assign(p2_costs):
            self._tracks[unmatched_t[ui]].update(p2_pool[di], self.degraded_boost)
            p2_matched_pool.add(di)

        # Spawn new tracks from unmatched high-conf detections
        n_low = len(low)
        for di, obs in enumerate(high):
            if di not in matched_h:
                pool_idx = n_low + sum(1 for j in range(di) if j not in matched_h)
                if pool_idx not in p2_matched_pool:
                    tr = _GateKalmanTrack(self._next_id, obs, frame_idx)
                    self._tracks.append(tr)
                    self._next_id += 1

        # Prune lost tracks
        self._tracks = [tr for tr in self._tracks if tr.frames_since_observation <= self.max_lost]

        # Build output
        result: _List[GateFrameTrack] = []
        for tr in sorted(self._tracks, key=lambda t: t.track_id):
            bev_x, bev_y = float(tr.x[0]), float(tr.x[1])
            if H_inv_for_output is not None:
                img_x, img_y = _project_point_h(H_inv_for_output, bev_x, bev_y)
            else:
                img_x, img_y = bev_x, bev_y
            result.append(GateFrameTrack(
                track_id=int(tr.track_id),
                bev_x=bev_x,
                bev_y=bev_y,
                bev_vx=float(tr.x[2]),
                bev_vy=float(tr.x[3]),
                innovation_magnitude=float(tr.innovation_magnitude) if tr.innovation_magnitude is not None else None,
                frames_since_observation=int(tr.frames_since_observation),
                base_px={"x_px": float(img_x), "y_px": float(img_y)},
            ))
        return result
