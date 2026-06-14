"""
Perspective transformation module.
Converts 2D pixel coordinates to real-world 3D coordinates (bird's-eye view)
using homography based on known gate positions.

Includes camera motion compensation (Phase 2) and dynamic per-frame
scaling (Phase 4) for handling moving cameras in ski race footage.
"""
import cv2
import numpy as np


class CameraMotionCompensator:
    """
    Estimate and compensate for camera motion using detected gates as
    fixed-world anchor points.

    Since gates are physically fixed on the mountain, any movement of a
    gate's pixel position between frames is caused by camera panning/tilting.
    We compute the median gate displacement per frame relative to a baseline
    and subtract it from skier coordinates to get stabilized positions.
    """

    def __init__(self, baseline_gates, frame_gate_history, mode="translation"):
        """
        Args:
            baseline_gates: Dict {gate_id: (center_x, base_y)} — averaged
                           gate positions from temporal tracking (the "true" positions).
            frame_gate_history: Dict {frame_idx: {gate_id: (center_x, base_y)}} —
                               per-frame gate positions from TemporalGateTracker.
            mode: "translation" (default) or "affine" (rotation+translation+scale).
        """
        self.baseline = baseline_gates
        self.history = frame_gate_history
        self.mode = mode
        self.offsets = {}  # frame_idx -> (dx, dy)
        self.affine = {}   # frame_idx -> 2x3 affine matrix (baseline -> frame)
        self.affine_inv = {}  # frame_idx -> 2x3 inverse (frame -> baseline)

    def estimate_motion(self):
        """
        For each frame, compute camera motion relative to baseline gates.

        Translation mode:
            Median displacement of visible gates (dx, dy), with interpolation.
        Affine mode:
            Estimate affine transform (rotation + translation + scale) using
            matched gates. Missing frames fall back to nearest known transform.
        """
        if self.mode == "affine":
            self._estimate_affine_motion()
            return

        # Translation-only (default)
        raw_offsets = {}
        rejected_spike_frames = []
        spike_thresh = None

        for frame_idx in sorted(self.history.keys()):
            frame_gates = self.history[frame_idx]
            dx_list = []
            dy_list = []

            for gate_id, (cx, by) in frame_gates.items():
                if gate_id in self.baseline:
                    bx, bby = self.baseline[gate_id]
                    dx_list.append(cx - bx)
                    dy_list.append(by - bby)

            if len(dx_list) >= 2:
                raw_offsets[frame_idx] = (
                    float(np.median(dx_list)),
                    float(np.median(dy_list)),
                )
            elif len(dx_list) == 1:
                raw_offsets[frame_idx] = (dx_list[0], dy_list[0])

        if not raw_offsets:
            self.offsets = {}
            return

        # Robust single-frame spike rejection: reject frames whose (dx,dy)
        # step is wildly inconsistent with the typical motion.
        all_frames_full = sorted(raw_offsets.keys())
        min_f_full, max_f_full = int(all_frames_full[0]), int(all_frames_full[-1])

        frames_sorted = list(all_frames_full)
        if len(frames_sorted) >= 3:
            offsets_arr = np.asarray([raw_offsets[f] for f in frames_sorted], dtype=np.float64)
            steps = np.linalg.norm(offsets_arr[1:] - offsets_arr[:-1], axis=1)
            frame_gaps = np.asarray(
                [max(1, int(frames_sorted[i + 1] - frames_sorted[i])) for i in range(len(frames_sorted) - 1)],
                dtype=np.float64,
            )
            # Normalize step by frame gap so missing-gate gaps don't look like spikes.
            steps_per_frame = steps / np.maximum(1.0, frame_gaps)

            med = float(np.median(steps_per_frame)) if len(steps_per_frame) else 0.0
            mad = float(np.median(np.abs(steps_per_frame - med))) if len(steps_per_frame) else 0.0
            spike_thresh = float(max(50.0, med + 6.0 * mad))

            filtered = {}
            last_kept = None
            for f in frames_sorted:
                if last_kept is None:
                    filtered[f] = raw_offsets[f]
                    last_kept = f
                    continue
                dx0, dy0 = raw_offsets[last_kept]
                dx1, dy1 = raw_offsets[f]
                gap = max(1, int(f - last_kept))
                step = float(((dx1 - dx0) ** 2 + (dy1 - dy0) ** 2) ** 0.5)
                step_per_frame = step / gap
                if step_per_frame > spike_thresh:
                    rejected_spike_frames.append(int(f))
                    continue
                filtered[f] = raw_offsets[f]
                last_kept = f

            # Only apply filtering if we still have enough anchors for interpolation.
            if len(filtered) >= 2:
                raw_offsets = filtered
                frames_sorted = sorted(raw_offsets.keys())

        # Interpolate missing frames between known offsets
        all_frames = frames_sorted
        min_f, max_f = min_f_full, max_f_full

        for f in range(min_f, max_f + 1):
            if f in raw_offsets:
                self.offsets[f] = raw_offsets[f]
            else:
                # Find nearest known frames before and after
                prev_f = max(k for k in all_frames if k <= f) if any(k <= f for k in all_frames) else None
                next_f = min(k for k in all_frames if k >= f) if any(k >= f for k in all_frames) else None

                if prev_f is not None and next_f is not None and next_f != prev_f:
                    t = (f - prev_f) / (next_f - prev_f)
                    dx = raw_offsets[prev_f][0] + t * (raw_offsets[next_f][0] - raw_offsets[prev_f][0])
                    dy = raw_offsets[prev_f][1] + t * (raw_offsets[next_f][1] - raw_offsets[prev_f][1])
                    self.offsets[f] = (dx, dy)
                elif prev_f is not None:
                    self.offsets[f] = raw_offsets[prev_f]
                elif next_f is not None:
                    self.offsets[f] = raw_offsets[next_f]

        # Rolling-median smoothing over the dense interpolated offsets.
        smooth_window = 15
        smooth_applied = False
        # Only smooth when we have enough frames to support the chosen window.
        # Smoothing on very short sequences can incorrectly "pull" early frames
        # toward the global median (e.g. 3 frames -> median over all frames),
        # which breaks the expectation that the baseline frame has ~0 offset.
        if smooth_window >= 3 and len(self.offsets) >= smooth_window:
            if smooth_window % 2 == 0:
                smooth_window += 1
            dense_frames = list(range(min_f, max_f + 1))
            dxs = np.asarray([self.offsets[f][0] for f in dense_frames], dtype=np.float64)
            dys = np.asarray([self.offsets[f][1] for f in dense_frames], dtype=np.float64)
            half = smooth_window // 2
            dxs_s = []
            dys_s = []
            for i in range(len(dense_frames)):
                lo = max(0, i - half)
                hi = min(len(dense_frames), i + half + 1)
                dxs_s.append(float(np.median(dxs[lo:hi])))
                dys_s.append(float(np.median(dys[lo:hi])))
            self.offsets = {
                int(f): (float(dx), float(dy))
                for f, dx, dy in zip(dense_frames, dxs_s, dys_s)
            }
            smooth_applied = True

        n_raw = len(raw_offsets)
        n_total = len(self.offsets)
        extra = []
        if rejected_spike_frames:
            extra.append(f"rejected_spikes={len(rejected_spike_frames)}")
        if spike_thresh is not None:
            extra.append(f"spike_thresh≈{spike_thresh:.0f}px")
        extra.append(f"median_smooth={smooth_window}" if smooth_applied else "median_smooth=off")
        extra_text = ", ".join(extra)
        print(f"✓ Camera motion (translation): {n_raw} measured frames, {n_total} total ({extra_text})")

        # Phase 2 diagnostic (Prof. feedback): Translation-only assumes the
        # camera slides without rotating. For broadcast/pan cameras this
        # introduces edge distortion — pixels on the left move faster than
        # the right during panning, biasing turn-radius calculations at
        # frame edges. Recommend affine mode for panning footage.
        if n_raw > 10:
            sorted_frames = sorted(raw_offsets.keys())
            dxs = [raw_offsets[f][0] for f in sorted_frames]
            dys = [raw_offsets[f][1] for f in sorted_frames]
            range_dx = max(dxs) - min(dxs) if dxs else 0
            # Large horizontal drift suggests panning camera
            if range_dx > 100:
                print(f"  ⚠️  Large horizontal camera drift detected ({range_dx:.0f}px). "
                      f"This suggests a panning camera. Translation-only compensation "
                      f"distorts turn radii at frame edges. Consider --camera-mode affine.")

    def _estimate_affine_motion(self):
        """
        Estimate rigid (Euclidean) motion per frame: rotation + translation ONLY.

        CRITICAL FIX: Previously used estimateAffinePartial2D which includes
        a scale component. In snowy/low-texture environments, the algorithm
        hallucinates zoom changes. When combined with DynamicScaleTransform,
        this creates a feedback loop:
          1. Affine "zoom" shrinks gate pixel distances
          2. DynamicScale sees smaller gate gaps → computes huge meters/pixel
          3. Small pixel movements → supersonic speeds (11,000 km/h)

        Now uses estimateAffinePartial2D but strips the scale component,
        keeping only rotation + translation (Euclidean rigid motion).
        Falls back to translation for frames with <3 gates.
        Interpolates missing frames between measured frames for continuity.
        """
        raw_affine = {}      # frame_idx -> 2x3 matrix (measured only)
        raw_affine_inv = {}  # frame_idx -> 2x3 inverse (measured only)
        n_scale_rejected = 0

        for frame_idx in sorted(self.history.keys()):
            frame_gates = self.history[frame_idx]
            src = []
            dst = []

            for gate_id, (cx, by) in frame_gates.items():
                if gate_id in self.baseline:
                    bx, bby = self.baseline[gate_id]
                    src.append([bx, bby])  # baseline
                    dst.append([cx, by])   # observed

            if len(src) >= 3:
                src_pts = np.array(src, dtype=np.float32)
                dst_pts = np.array(dst, dtype=np.float32)
                M, inliers = cv2.estimateAffinePartial2D(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0,
                )
                if M is not None:
                    # Extract scale from the affine matrix.
                    # estimateAffinePartial2D returns:
                    #   [[s*cos(θ), -s*sin(θ), tx],
                    #    [s*sin(θ),  s*cos(θ), ty]]
                    # Scale = sqrt(M[0,0]^2 + M[1,0]^2)
                    scale = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))

                    # SAFETY: Strip scale component, keep only rotation+translation.
                    # This prevents the "affine zoom hallucination" in low-texture
                    # snow scenes that causes DynamicScale to explode.
                    if abs(scale - 1.0) > 0.01:
                        # Normalize to unit scale (Euclidean only)
                        M[0, 0] /= scale
                        M[0, 1] /= scale
                        M[1, 0] /= scale
                        M[1, 1] /= scale
                        n_scale_rejected += 1

                    raw_affine[frame_idx] = M
                    A = np.vstack([M, [0, 0, 1]])
                    try:
                        A_inv = np.linalg.inv(A)
                        raw_affine_inv[frame_idx] = A_inv[:2, :]
                    except np.linalg.LinAlgError:
                        pass
                    continue

            # Fallback: translation-only if we don't have enough gates
            if len(src) >= 1:
                dx = float(np.median([dst[i][0] - src[i][0] for i in range(len(src))]))
                dy = float(np.median([dst[i][1] - src[i][1] for i in range(len(src))]))
                M = np.array([[1.0, 0.0, dx],
                              [0.0, 1.0, dy]], dtype=np.float32)
                raw_affine[frame_idx] = M
                raw_affine_inv[frame_idx] = np.array([[1.0, 0.0, -dx],
                                                       [0.0, 1.0, -dy]], dtype=np.float32)

        if not raw_affine:
            return

        # Interpolate missing frames between measured frames
        all_frames = sorted(raw_affine.keys())
        min_f, max_f = all_frames[0], all_frames[-1]

        for f in range(min_f, max_f + 1):
            if f in raw_affine:
                self.affine[f] = raw_affine[f]
                if f in raw_affine_inv:
                    self.affine_inv[f] = raw_affine_inv[f]
            else:
                # Linear interpolation of affine matrix components
                prev_f = max(k for k in all_frames if k <= f) if any(k <= f for k in all_frames) else None
                next_f = min(k for k in all_frames if k >= f) if any(k >= f for k in all_frames) else None

                if prev_f is not None and next_f is not None and next_f != prev_f:
                    t = (f - prev_f) / (next_f - prev_f)
                    M_interp = (1 - t) * raw_affine[prev_f] + t * raw_affine[next_f]
                    self.affine[f] = M_interp
                    # Compute inverse of interpolated matrix
                    A = np.vstack([M_interp, [0, 0, 1]])
                    try:
                        A_inv = np.linalg.inv(A)
                        self.affine_inv[f] = A_inv[:2, :].astype(np.float32)
                    except np.linalg.LinAlgError:
                        # Fallback: interpolate inverses directly
                        if prev_f in raw_affine_inv and next_f in raw_affine_inv:
                            self.affine_inv[f] = (
                                (1 - t) * raw_affine_inv[prev_f] + t * raw_affine_inv[next_f]
                            )
                elif prev_f is not None:
                    self.affine[f] = raw_affine[prev_f]
                    if prev_f in raw_affine_inv:
                        self.affine_inv[f] = raw_affine_inv[prev_f]
                elif next_f is not None:
                    self.affine[f] = raw_affine[next_f]
                    if next_f in raw_affine_inv:
                        self.affine_inv[f] = raw_affine_inv[next_f]

        n_raw = len(raw_affine)
        n_total = len(self.affine)
        print(f"✓ Camera motion (rigid/Euclidean): {n_raw} measured, {n_total} total "
              f"(scale stripped from {n_scale_rejected} frames)")

    def _apply_affine_inv(self, M, x, y):
        """Apply a 2x3 inverse affine matrix to a point."""
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return (float(nx), float(ny))

    def _nearest_frame(self, frame_idx, frame_dict):
        """Find nearest frame key using bisect for O(log n) lookup."""
        if not frame_dict:
            return None
        import bisect
        if not hasattr(self, '_sorted_keys_cache') or self._sorted_keys_id != id(frame_dict):
            self._sorted_keys = sorted(frame_dict.keys())
            self._sorted_keys_id = id(frame_dict)
        keys = self._sorted_keys
        pos = bisect.bisect_left(keys, frame_idx)
        if pos == 0:
            return keys[0]
        if pos >= len(keys):
            return keys[-1]
        before, after = keys[pos - 1], keys[pos]
        return before if (frame_idx - before) <= (after - frame_idx) else after

    def stabilize_point(self, x, y, frame_idx):
        """
        Remove camera motion from a pixel coordinate.

        Returns:
            (x_stabilized, y_stabilized) as if camera never moved.
        """
        if self.mode == "affine":
            if frame_idx in self.affine_inv:
                return self._apply_affine_inv(self.affine_inv[frame_idx], x, y)
            if self.affine_inv:
                nearest = self._nearest_frame(frame_idx, self.affine_inv)
                if nearest is not None:
                    return self._apply_affine_inv(self.affine_inv[nearest], x, y)
            return (x, y)

        if frame_idx in self.offsets:
            dx, dy = self.offsets[frame_idx]
            return (x - dx, y - dy)

        # For frames outside known range, find nearest
        if self.offsets:
            nearest = self._nearest_frame(frame_idx, self.offsets)
            if nearest is not None:
                dx, dy = self.offsets[nearest]
                return (x - dx, y - dy)

        return (x, y)

    def stabilize_trajectory(self, trajectory_2d):
        """
        Apply camera stabilization to an entire 2D trajectory.

        Args:
            trajectory_2d: List of {"frame": int, "x": float, "y": float}

        Returns:
            New list with stabilized coordinates.
        """
        stabilized = []
        for pt in trajectory_2d:
            sx, sy = self.stabilize_point(pt["x"], pt["y"], pt["frame"])
            stabilized.append({"frame": pt["frame"], "x": sx, "y": sy})
        return stabilized


class DynamicScaleTransform:
    """
    Compute per-frame pixel-to-meter scaling from visible gate pairs.

    As the camera moves and zooms, the pixel distance between gates changes.
    By measuring gate-pair distances in each frame, we get a local scale
    that accounts for perspective depth variations.
    """

    def __init__(self, frame_gate_history, gate_spacing_m=12.0, camera_pitch_deg=None):
        """
        Args:
            frame_gate_history: Dict {frame_idx: {gate_id: (cx, by)}}
            gate_spacing_m: Real-world distance between consecutive gates.
            camera_pitch_deg: Camera pitch angle in degrees. If provided,
                              retained for metadata/future corrections.
        """
        self.history = frame_gate_history
        self.gate_spacing_m = float(gate_spacing_m)
        self.camera_pitch_deg = float(camera_pitch_deg) if camera_pitch_deg is not None else None
        self.pitch_estimated = False
        if self.camera_pitch_deg is None:
            estimated = self.estimate_camera_pitch_deg_from_history(self.history)
            if estimated is not None:
                self.camera_pitch_deg = float(estimated)
                self.pitch_estimated = True
                print(f"  ℹ️  Estimated camera pitch: {self.camera_pitch_deg:.1f}° from gate geometry")
        self.effective_spacing_m = self._compute_effective_spacing()
        self.frame_ppm = {}  # frame_idx -> ppm_y
        self.raw_frame_ppm = {}  # measured frame_idx -> raw ppm_y
        self.smoothed_frame_ppm = {}  # measured frame_idx -> median-filtered ppm_y
        self.rejected_frames = []  # measured frame_idx rejected by bounds
        self.overall_median_ppm = None

    def _compute_effective_spacing(self):
        # Camera pitch does NOT compress gate spacing.
        # Perspective effects are handled by per-frame pixel gap measurement.
        return self.gate_spacing_m

    @staticmethod
    def estimate_camera_pitch_deg_from_history(frame_gate_history):
        """
        Estimate camera pitch from gate geometry in a frame with enough gates.

        Uses apparent perspective compression: gates higher in the image are
        farther away and should have smaller apparent spacing / size.
        """
        if not frame_gate_history:
            return None

        frame_items = sorted(
            frame_gate_history.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )

        for _, frame_gates in frame_items:
            if len(frame_gates) < 4:
                continue
            gate_points = []
            for payload in frame_gates.values():
                if isinstance(payload, dict):
                    cx = payload.get("center_x")
                    by = payload.get("base_y")
                    # Optional: pass gate pixel height if available.
                    h = payload.get("height")
                    if cx is None or by is None:
                        continue
                    gate_points.append((float(cx), float(by), float(h) if h is not None else None))
                else:
                    if len(payload) >= 2:
                        gate_points.append((float(payload[0]), float(payload[1]), None))

            estimate = DynamicScaleTransform.estimate_camera_pitch_deg(gate_points)
            if estimate is not None:
                return estimate

        return None

    @staticmethod
    def estimate_camera_pitch_deg(gate_points, min_pitch_deg=2.0, max_pitch_deg=40.0):
        """
        Estimate camera pitch from top-vs-bottom apparent gate scale.

        Preferred signal: ratio of gate sizes (if per-gate pixel heights exist).
        Fallback signal: ratio of consecutive gate spacing in Y (bottom/top).
        """
        if not gate_points or len(gate_points) < 4:
            return None

        # Normalize points to (x, y, size_px_or_none)
        norm = []
        for p in gate_points:
            if isinstance(p, dict):
                x = p.get("center_x")
                y = p.get("base_y")
                size = p.get("height")
            else:
                x = p[0] if len(p) > 0 else None
                y = p[1] if len(p) > 1 else None
                size = p[2] if len(p) > 2 else None
            if x is None or y is None:
                continue
            norm.append((float(x), float(y), float(size) if size is not None else None))

        if len(norm) < 4:
            return None

        norm.sort(key=lambda g: g[1])
        n = len(norm)
        k = max(1, n // 3)
        top = norm[:k]
        bottom = norm[-k:]

        # 1) Size-ratio estimate if gate heights are available
        top_sizes = [g[2] for g in top if g[2] is not None and g[2] > 1e-6]
        bottom_sizes = [g[2] for g in bottom if g[2] is not None and g[2] > 1e-6]
        ratio = None
        if top_sizes and bottom_sizes:
            top_med = float(np.median(top_sizes))
            bottom_med = float(np.median(bottom_sizes))
            if top_med > 1e-6:
                ratio = bottom_med / top_med

        # 2) Fallback: spacing-ratio estimate
        if ratio is None:
            ys = [g[1] for g in norm]
            gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1e-3]
            if len(gaps) < 3:
                return None
            k_gap = max(1, len(gaps) // 3)
            top_gap = float(np.median(gaps[:k_gap]))
            bottom_gap = float(np.median(gaps[-k_gap:]))
            if top_gap <= 1e-6:
                return None
            ratio = bottom_gap / top_gap

        # Map perspective ratio to a practical pitch range.
        # ratio≈1 => shallow pitch; higher ratio => steeper apparent pitch.
        ratio = float(np.clip(ratio, 1.0, 8.0))
        norm_ratio = np.log(ratio) / np.log(8.0)
        pitch = min_pitch_deg + norm_ratio * (max_pitch_deg - min_pitch_deg)
        return float(np.clip(pitch, min_pitch_deg, max_pitch_deg))

    @staticmethod
    def _rolling_median(values, window):
        """Apply an edge-aware rolling median filter."""
        if not values:
            return []
        window = max(3, int(window))
        if window % 2 == 0:
            window += 1
        half = window // 2
        out = []
        for i in range(len(values)):
            lo = max(0, i - half)
            hi = min(len(values), i + half + 1)
            out.append(float(np.median(values[lo:hi])))
        return out

    def compute_scales(self, global_ppm_y, one_gate_px_ref=None,
                        max_change_rate=0.10, median_window=15):
        """
        For each frame with >=2 visible gates, compute local ppm_y
        from the median gate-pair pixel distance.

        Then:
          1) Apply a rolling median filter (default window=15 frames)
          2) Reject outliers outside [0.5x, 2.0x] of overall median
          3) Interpolate rejected values
          4) Densify to every frame between first/last observation

        Args:
            global_ppm_y: Fallback ppm_y from static gate detection.
            one_gate_px_ref: Reference pixel distance for one gate interval.
            max_change_rate: Legacy parameter kept for compatibility.
            median_window: Rolling median window size for smoothing.
        """
        _ = max_change_rate  # kept for backwards compatibility
        raw_ppm = {}  # Unfiltered per-frame ppm

        for frame_idx, frame_gates in self.history.items():
            if len(frame_gates) < 2:
                continue

            # Sort gates by base_y
            sorted_gates = sorted(frame_gates.values(), key=lambda g: g[1])
            ppm_estimates = []

            for i in range(len(sorted_gates) - 1):
                dy_px = sorted_gates[i + 1][1] - sorted_gates[i][1]
                if dy_px < 3.0:
                    continue  # Skip near-duplicate gates

                # Estimate how many real gate intervals this spans
                if one_gate_px_ref and one_gate_px_ref > 1.0:
                    n_gates = max(1, round(dy_px / one_gate_px_ref))
                else:
                    n_gates = 1

                dy_m = n_gates * self.effective_spacing_m
                if dy_m > 0:
                    ppm_estimates.append(dy_px / dy_m)

            if ppm_estimates:
                local_ppm = float(np.median(ppm_estimates))
                # Hard clamp to prevent extreme values (global bounds)
                if global_ppm_y:
                    local_ppm = max(0.3 * global_ppm_y, min(3.0 * global_ppm_y, local_ppm))
                raw_ppm[frame_idx] = local_ppm

        self.raw_frame_ppm = {int(f): float(v) for f, v in raw_ppm.items()}
        self.frame_ppm = {}
        self.smoothed_frame_ppm = {}
        self.rejected_frames = []
        self.overall_median_ppm = None

        if not raw_ppm:
            print("✓ Dynamic scale: 0 frames with valid gate pairs")
            return

        frames = np.asarray(sorted(raw_ppm.keys()), dtype=int)
        raw_vals = [float(raw_ppm[int(f)]) for f in frames]
        used_median_window = int(median_window)
        if used_median_window >= 3 and len(raw_vals) >= used_median_window:
            smooth_vals = np.asarray(self._rolling_median(raw_vals, used_median_window), dtype=float)
        else:
            # For short sequences, a large rolling window collapses everything to the same
            # global median (especially near edges). Keep the raw per-frame measurements
            # so gradual zoom/scale changes are preserved.
            used_median_window = 0
            smooth_vals = np.asarray(raw_vals, dtype=float)
        self.smoothed_frame_ppm = {
            int(f): float(v) for f, v in zip(frames.tolist(), smooth_vals.tolist())
        }

        overall_median = float(np.median(smooth_vals)) if len(smooth_vals) else None
        self.overall_median_ppm = overall_median

        if overall_median is None or overall_median <= 0:
            fallback = float(global_ppm_y) if global_ppm_y and global_ppm_y > 0 else 1.0
            dense_frames = range(int(frames[0]), int(frames[-1]) + 1)
            self.frame_ppm = {int(f): fallback for f in dense_frames}
            print(f"✓ Dynamic scale: fallback-only ({len(self.frame_ppm)} frames)")
            return

        low_bound = 0.5 * overall_median
        high_bound = 2.0 * overall_median
        valid_mask = (smooth_vals >= low_bound) & (smooth_vals <= high_bound)
        self.rejected_frames = [int(frames[i]) for i, is_valid in enumerate(valid_mask) if not is_valid]

        if np.any(valid_mask):
            valid_frames = frames[valid_mask].astype(float)
            valid_vals = smooth_vals[valid_mask]
            # Interpolate rejected measured frames between nearest valid frames.
            filtered_vals = np.interp(frames.astype(float), valid_frames, valid_vals)
        else:
            fallback = float(global_ppm_y) if global_ppm_y and global_ppm_y > 0 else overall_median
            filtered_vals = np.full_like(smooth_vals, float(fallback), dtype=float)

        # Fill full frame range so downstream can inspect/plot per-frame scale.
        dense_frames = np.arange(int(frames[0]), int(frames[-1]) + 1, dtype=int)
        if len(frames) >= 2:
            dense_vals = np.interp(dense_frames.astype(float), frames.astype(float), filtered_vals)
        else:
            dense_vals = np.full(len(dense_frames), float(filtered_vals[0]), dtype=float)
        self.frame_ppm = {
            int(f): float(v) for f, v in zip(dense_frames.tolist(), dense_vals.tolist())
        }

        n_total = int(len(frames))
        n_rejected = int(len(self.rejected_frames))
        n_accepted = n_total - n_rejected
        window_text = str(used_median_window) if used_median_window else "off"
        print(f"✓ Dynamic scale: {n_accepted}/{n_total} measured frames accepted "
              f"({n_rejected} rejected, median_window={window_text}, "
              f"bounds=[0.5x, 2.0x], median={overall_median:.2f} px/m)")

    def to_debug_dict(self):
        """Return dynamic-scale diagnostics for JSON output."""
        return {
            "camera_pitch_deg": float(self.camera_pitch_deg) if self.camera_pitch_deg is not None else None,
            "pitch_estimated": bool(self.pitch_estimated),
            "gate_spacing_m": float(self.gate_spacing_m),
            "effective_spacing_m": float(self.effective_spacing_m),
            "overall_median_ppm": float(self.overall_median_ppm) if self.overall_median_ppm is not None else None,
            "n_raw_frames": int(len(self.raw_frame_ppm)),
            "n_dense_frames": int(len(self.frame_ppm)),
            "rejected_frames": [int(f) for f in self.rejected_frames],
            "frame_ppm_raw": [
                {"frame": int(f), "ppm": float(self.raw_frame_ppm[f])}
                for f in sorted(self.raw_frame_ppm.keys())
            ],
            "frame_ppm_smoothed": [
                {"frame": int(f), "ppm": float(self.smoothed_frame_ppm[f])}
                for f in sorted(self.smoothed_frame_ppm.keys())
            ],
            "frame_ppm": [
                {"frame": int(f), "ppm": float(self.frame_ppm[f])}
                for f in sorted(self.frame_ppm.keys())
            ],
        }

    def get_ppm(self, frame_idx, fallback_ppm):
        """
        Get pixels-per-meter for a specific frame.

        Falls back to nearest known frame or global fallback.
        """
        if frame_idx in self.frame_ppm:
            return self.frame_ppm[frame_idx]

        # Find nearest frame with known ppm using bisect for O(log n)
        if self.frame_ppm:
            if not hasattr(self, '_sorted_frames') or self._sorted_frames_len != len(self.frame_ppm):
                self._sorted_frames = sorted(self.frame_ppm.keys())
                self._sorted_frames_len = len(self.frame_ppm)
            import bisect
            keys = self._sorted_frames
            pos = bisect.bisect_left(keys, frame_idx)
            if pos == 0:
                nearest = keys[0]
            elif pos >= len(keys):
                nearest = keys[-1]
            else:
                before, after = keys[pos - 1], keys[pos]
                nearest = before if (frame_idx - before) <= (after - frame_idx) else after

            if abs(nearest - frame_idx) < 30:  # Within 1 second at 30fps
                return self.frame_ppm[nearest]

        return fallback_ppm


class HomographyTransform:
    """
    Transform 2D pixel coordinates to real-world meters using
    homography calculated from detected gate positions.
    """

    def __init__(self):
        self.H = None  # Homography matrix
        self.H_inv = None  # Inverse (for going 3D -> 2D)
        self.mode = "homography"
        self.y_map = None
        self.ppm_y = None
        self.ppm_bounds = None
        self.ppm_x = None
        self.gate_spacing_m = None
        self.effective_gate_spacing_m = None
        self.camera_pitch_deg = None
        self.centerline_fit = None
        self.camera_compensator = None  # Phase 2: CameraMotionCompensator
        self.dynamic_scale = None       # Phase 4: DynamicScaleTransform
        self.jump_guard_info = None     # Transform-time discontinuity guard

    def calculate_from_gates(self, gates_2d, gate_spacing_m=12.0):
        """
        Calculate homography from detected gate positions.

        ⚠️  WARNING (Prof. feedback Phase 4): This computes a STATIC global
        homography from gates visible in the first frame. The perspective of
        the slope changes constantly as the skier moves down. A static
        homography is incorrect for the entire video.

        Prefer: --projection scale --stabilize (dynamic per-frame scaling).
        If you must use homography, it should be re-calculated per-frame
        using the 4 nearest gates (not yet implemented).

        Args:
            gates_2d: List of gate positions in pixels [(x, y), ...].
                      Should be sorted top-to-bottom (far to near).
            gate_spacing_m: Real-world distance between consecutive gates (meters).
                           Standard: ~12m for slalom, ~25-30m for GS.

        Returns:
            Homography matrix (3x3), or None if insufficient gates.
        """
        print("⚠️  Static homography: calculated ONCE from initial gate positions. "
              "This is incorrect for video where perspective changes as the "
              "skier moves down slope. Use --stabilize for dynamic scaling.")
        if len(gates_2d) < 4:
            print(f"⚠️  Only {len(gates_2d)} gates detected. Need at least 4 for homography.")
            if len(gates_2d) >= 2:
                print("   Using piecewise scale fallback from gate spacing.")
                self._calculate_scale_from_gates(gates_2d, gate_spacing_m)
            else:
                print("   Using identity transform as fallback.")
                self.H = np.eye(3)
                self.mode = "identity"
            return self.H

        # Create idealized 3D positions (bird's eye view)
        # Alternating left-right pattern typical of slalom/GS
        gates_3d = []
        for i in range(len(gates_2d)):
            # Slight lateral offset for alternating gates
            x_3d = (i % 2) * 3.0  # 3m lateral offset
            y_3d = i * gate_spacing_m
            gates_3d.append([x_3d, y_3d])

        # Use up to 8 gate pairs for better fit
        n_points = min(len(gates_2d), 8)
        pts_2d = np.float32(gates_2d[:n_points])
        pts_3d = np.float32(gates_3d[:n_points])

        # Calculate homography with RANSAC for robustness
        self.H, status = cv2.findHomography(pts_2d, pts_3d, cv2.RANSAC, 5.0)

        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
            inliers = status.sum() if status is not None else n_points
            self.mode = "homography"
            print(f"✓ Homography calculated using {inliers}/{n_points} gate pairs")
        else:
            print("⚠️  Homography calculation failed. Falling back to piecewise scale.")
            if len(gates_2d) >= 2:
                self._calculate_scale_from_gates(gates_2d, gate_spacing_m)
            else:
                print("   Using identity transform as fallback.")
                self.H = np.eye(3)
                self.mode = "identity"

        return self.H

    def calculate_scale_from_gates(self, gates_2d, gate_spacing_m=12.0):
        """
        Calculate a piecewise scale mapping from gate positions.
        This is a pseudo-3D projection that avoids global homography.
        """
        if len(gates_2d) < 2:
            print(f"⚠️  Only {len(gates_2d)} gates detected. Need at least 2 for scale mapping.")
            print("   Using identity transform as fallback.")
            self.H = np.eye(3)
            self.mode = "identity"
            return self.H

        self._calculate_scale_from_gates(gates_2d, gate_spacing_m)
        return self.H

    def _transform_point_with_ppm(self, point_2d, frame_idx=None):
        """
        Transform a single 2D pixel coordinate to 3D world coordinates,
        returning the local ppm_y used for the mapping.
        """
        if self.mode == "homography":
            if self.H is None:
                raise ValueError("Must calculate homography first")

            pt = np.array([[point_2d[0], point_2d[1]]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pt, self.H)
            return (float(transformed[0][0][0]), float(transformed[0][0][1]), None)

        if self.mode == "scale":
            if not self.y_map or len(self.y_map) < 2:
                return (float(point_2d[0]), float(point_2d[1]), self.ppm_y)

            y_px = float(point_2d[1])
            # Find enclosing gate segment
            idx = 0
            for i in range(len(self.y_map) - 1):
                if self.y_map[i]["y_px"] <= y_px <= self.y_map[i + 1]["y_px"]:
                    idx = i
                    break
                if y_px < self.y_map[0]["y_px"]:
                    idx = 0
                    break
                if y_px > self.y_map[-1]["y_px"]:
                    idx = len(self.y_map) - 2
                    break

            g0 = self.y_map[idx]
            g1 = self.y_map[idx + 1]
            y0, y1 = g0["y_px"], g1["y_px"]
            # Real meter span of this segment (accounts for skipped gates)
            segment_m = g1["y_m"] - g0["y_m"]
            if segment_m < 1e-6:
                segment_m = self.effective_gate_spacing_m or self.gate_spacing_m or 1.0

            if abs(y1 - y0) < 1e-6:
                ppm_y = self.ppm_y or 1.0
                t = 0.0
            else:
                ppm_y = (y1 - y0) / segment_m
                t = (y_px - y0) / (y1 - y0)

            # Phase 4: Use dynamic per-frame scale if available
            if self.dynamic_scale is not None and frame_idx is not None:
                dyn_ppm = self.dynamic_scale.get_ppm(frame_idx, None)
                if dyn_ppm is not None:
                    ppm_y = dyn_ppm

            # Clamp ppm to avoid extreme scale from near-duplicate gates
            if self.ppm_y is not None:
                low, high = self.ppm_bounds if self.ppm_bounds else (0.5 * self.ppm_y, 2.0 * self.ppm_y)
                if ppm_y < low or ppm_y > high:
                    ppm_y = self.ppm_y

            y_m = g0["y_m"] + t * segment_m
            if self.centerline_fit is not None:
                a, b = self.centerline_fit
                x_ref = a * y_px + b
            else:
                x_ref = (g0["x_px"] + g1["x_px"]) / 2.0
            # Use local ppm_y for X as well (isotropic at each depth)
            x_m = (float(point_2d[0]) - x_ref) / ppm_y
            return (x_m, y_m, ppm_y)

        # Identity fallback
        return (float(point_2d[0]), float(point_2d[1]), None)

    def transform_point(self, point_2d, frame_idx=None):
        """
        Transform a single 2D pixel coordinate to 3D world coordinates.

        Args:
            point_2d: (x, y) in pixels.
            frame_idx: Optional frame index for dynamic scale (Phase 4).

        Returns:
            (x, y) in meters (bird's-eye view).
        """
        x_m, y_m, _ = self._transform_point_with_ppm(point_2d, frame_idx=frame_idx)
        return (x_m, y_m)

    def transform_trajectory(self, trajectory_2d, stabilize=False,
                              max_jump_m=5.0, fps=30.0,
                              jump_guard=False, stabilize_after_scale=True):
        """
        Transform an entire 2D trajectory to 3D coordinates.

        Optionally includes a physical jump guard: if any point-to-point 3D
        distance exceeds max_jump_m, the point is replaced by linear
        interpolation from its neighbors. This can smooth isolated scale
        explosions but may distort acceleration; disabled by default.

        Args:
            trajectory_2d: List of {"frame": int, "x": float, "y": float}.
            stabilize: If True, apply camera motion compensation first.
            max_jump_m: Maximum allowed 3D jump per frame (meters).
            fps: Video frame rate (for scaling max_jump with frame gaps).
            jump_guard: If True, apply multi-pass jump guard after transform.
            stabilize_after_scale: If True, apply camera stabilization in 3D
                                   after scaling raw pixels to meters.

        Returns:
            List of {"frame": int, "x": float, "y": float} in meters.
        """
        if self.H is None:
            raise ValueError("Must calculate homography first")

        # Phase 2: Apply camera stabilization BEFORE scaling only if requested
        if stabilize and self.camera_compensator is not None and not stabilize_after_scale:
            working_trajectory = self.camera_compensator.stabilize_trajectory(trajectory_2d)
        else:
            working_trajectory = trajectory_2d

        trajectory_3d = []
        local_ppm = []
        for point in working_trajectory:
            frame_idx = point.get("frame")
            pt_3d = self._transform_point_with_ppm([point["x"], point["y"]], frame_idx=frame_idx)
            trajectory_3d.append({
                "frame": point["frame"],
                "x": pt_3d[0],
                "y": pt_3d[1],
            })
            local_ppm.append(pt_3d[2])

        # Phase 2 (alternate): stabilize in 3D after scaling raw pixels
        if stabilize and self.camera_compensator is not None and stabilize_after_scale:
            for i, pt in enumerate(trajectory_2d):
                frame_idx = pt.get("frame")
                x_raw = pt["x"]
                y_raw = pt["y"]
                x_stab, y_stab = self.camera_compensator.stabilize_point(x_raw, y_raw, frame_idx)
                ppm_y = local_ppm[i] if i < len(local_ppm) else None
                if ppm_y is None or ppm_y <= 0:
                    ppm_y = self.ppm_y if self.ppm_y else 1.0
                dx_m = (x_stab - x_raw) / ppm_y
                dy_m = (y_stab - y_raw) / ppm_y
                trajectory_3d[i]["x"] += dx_m
                trajectory_3d[i]["y"] += dy_m

        # ── Physical jump guard (multi-pass) ──
        # When many consecutive points have bad scale, a single forward pass
        # can't fix them because neighbors are also bad. We use an iterative
        # approach:
        #   1. Find the longest run of "trusted" (low-jump) points as anchors.
        #   2. Mark all points with jumps > max_jump_m as "suspect".
        #   3. Linearly interpolate suspect points between trusted anchors.
        #   4. Repeat until convergence or max iterations.
        n_clamped = 0
        enabled = bool(jump_guard and max_jump_m is not None)
        if enabled:
            n_clamped = self._jump_guard_multipass(trajectory_3d, max_jump_m, fps)

            if n_clamped > 0:
                print(f"  ⚠️  Jump guard: {n_clamped} points interpolated (multi-pass)")

        self.jump_guard_info = {
            "enabled": bool(enabled),
            "max_jump_m": float(max_jump_m) if max_jump_m is not None else None,
            "interpolated_points": int(n_clamped),
        }

        return trajectory_3d

    @staticmethod
    def _jump_guard_multipass(trajectory_3d, max_jump_m, fps,
                               max_iterations=10):
        """
        Multi-pass jump guard that handles cascading bad scale frames.

        Strategy:
          1. Identify "trusted" points where the jump from the previous point
             is within max_jump_m (scaled by frame gap).
          2. The first and last points are always trusted (anchors).
          3. For each "suspect" run between two trusted anchors, replace all
             suspect points with linear interpolation between the anchors.
          4. Repeat until no more suspects are found (convergence).

        This handles the common case where bad dynamic scale affects many
        consecutive frames — the single-pass approach fails because neighbors
        are themselves bad.

        Returns:
            Total number of points interpolated.
        """
        if len(trajectory_3d) < 3:
            return 0

        total_clamped = 0

        for iteration in range(max_iterations):
            # Mark trusted vs suspect points
            trusted = [True]  # First point is always trusted
            for i in range(1, len(trajectory_3d)):
                prev = trajectory_3d[i - 1]
                curr = trajectory_3d[i]
                frame_gap = max(1, curr["frame"] - prev["frame"])
                allowed = max_jump_m * frame_gap

                dx = curr["x"] - prev["x"]
                dy = curr["y"] - prev["y"]
                jump = (dx**2 + dy**2) ** 0.5

                trusted.append(jump <= allowed)

            # Last point: also check backward
            if len(trajectory_3d) >= 2:
                prev = trajectory_3d[-2]
                curr = trajectory_3d[-1]
                frame_gap = max(1, curr["frame"] - prev["frame"])
                allowed = max_jump_m * frame_gap
                dx = curr["x"] - prev["x"]
                dy = curr["y"] - prev["y"]
                if (dx**2 + dy**2) ** 0.5 > allowed:
                    trusted[-1] = False

            # Count suspects this pass
            n_suspect = sum(1 for t in trusted if not t)
            if n_suspect == 0:
                break

            # Find trusted anchor indices
            anchors = [i for i, t in enumerate(trusted) if t]

            if len(anchors) < 2:
                # Almost everything is suspect; use first and last as anchors
                anchors = [0, len(trajectory_3d) - 1]

            # Interpolate suspect points between consecutive anchors
            n_fixed = 0
            for a_idx in range(len(anchors) - 1):
                start = anchors[a_idx]
                end = anchors[a_idx + 1]

                if end - start <= 1:
                    continue  # No suspects between these anchors

                # Linear interpolate all points between start and end
                f_start = trajectory_3d[start]["frame"]
                f_end = trajectory_3d[end]["frame"]
                x_start = trajectory_3d[start]["x"]
                y_start = trajectory_3d[start]["y"]
                x_end = trajectory_3d[end]["x"]
                y_end = trajectory_3d[end]["y"]

                for j in range(start + 1, end):
                    if not trusted[j]:
                        f_j = trajectory_3d[j]["frame"]
                        if f_end > f_start:
                            t = (f_j - f_start) / (f_end - f_start)
                        else:
                            t = 0.5
                        trajectory_3d[j]["x"] = x_start + t * (x_end - x_start)
                        trajectory_3d[j]["y"] = y_start + t * (y_end - y_start)
                        n_fixed += 1

            total_clamped += n_fixed
            if n_fixed == 0:
                break

        return total_clamped

    def get_reprojection_error(self, gates_2d, gate_spacing_m=12.0):
        """
        Calculate reprojection error to assess homography quality.
        Lower is better. >10 pixels means the transform is suspect.

        Returns:
            Mean reprojection error in pixels.
        """
        if self.H is None or self.H_inv is None:
            return float("inf")

        errors = []
        for i, gate in enumerate(gates_2d):
            # Forward transform: 2D -> 3D
            pt_3d = self.transform_point(gate)

            # Inverse: 3D -> 2D
            pt_3d_arr = np.array([[pt_3d[0], pt_3d[1]]], dtype=np.float32).reshape(-1, 1, 2)
            reprojected = cv2.perspectiveTransform(pt_3d_arr, self.H_inv)
            rx, ry = reprojected[0][0]

            error = ((rx - gate[0]) ** 2 + (ry - gate[1]) ** 2) ** 0.5
            errors.append(error)

        mean_error = np.mean(errors)
        print(f"Reprojection error: {mean_error:.2f} pixels (target: <10)")
        return mean_error

    def _calculate_scale_from_gates(self, gates_2d, gate_spacing_m):
        """
        Piecewise linear scale mapping using gate spacing.
        Maps y using gate spacing and scales x using local y-scale.

        Key fix: we estimate how many real gates are between each pair of
        detected gates using the minimum-gap heuristic, so that skipped
        gates don't compress the meter scale.
        """
        gates_sorted = sorted(gates_2d, key=lambda g: g[1])
        self.gate_spacing_m = float(gate_spacing_m)
        # Camera pitch does NOT modify physical gate spacing.
        effective_spacing_m = self.gate_spacing_m
        self.effective_gate_spacing_m = effective_spacing_m

        # --- Step 1: estimate real gate counts between detected gates ---
        # Compute all Y-pixel gaps between consecutive detected gates
        y_gaps = []
        for i in range(1, len(gates_sorted)):
            dy = gates_sorted[i][1] - gates_sorted[i - 1][1]
            if dy > 1e-3:
                y_gaps.append(dy)

        if y_gaps:
            # Estimate ONE real gate interval from the Y-pixel gaps.
            #
            # Strategy: use the minimum gap that is "real" (not a double-
            # detection artifact). Double-detections produce tiny gaps (<40%
            # of the median gap). We filter those out and use the minimum
            # of the remaining gaps.
            #
            # If all gaps are similar (no outlier-small), the minimum is the
            # one-gate reference. If the detector skips every other gate,
            # the minimum still represents the closest detected pair.
            y_gaps_sorted = sorted(y_gaps)
            median_gap = y_gaps_sorted[len(y_gaps_sorted) // 2]

            # Filter out gaps that are less than 40% of the median
            # (these are likely double-detections of the same gate pole)
            real_gaps = [g for g in y_gaps_sorted if g > 0.4 * median_gap]
            if real_gaps:
                one_gate_px = real_gaps[0]
            else:
                one_gate_px = y_gaps_sorted[0]

            # Sanity: one_gate_px should be at least 15% of the Y-span of all gates.
            # If it's much smaller, we likely picked a within-gate pole gap instead
            # of a between-gate gap. Fall back to average spacing.
            total_y_span = gates_sorted[-1][1] - gates_sorted[0][1]
            if total_y_span > 0 and one_gate_px < 0.15 * total_y_span:
                fallback = total_y_span / max(1, len(gates_sorted) - 1)
                print(f"  ⚠️  one_gate_px ({one_gate_px:.1f}px) too small vs "
                      f"gate field span ({total_y_span:.0f}px), "
                      f"using average: {fallback:.1f}px")
                one_gate_px = fallback
        else:
            one_gate_px = None

        # --- Step 2: build y_map with corrected meter positions ---
        self.y_map = []
        cumulative_m = 0.0
        self.y_map.append({
            "y_px": float(gates_sorted[0][1]),
            "y_m": 0.0,
            "x_px": float(gates_sorted[0][0]),
            "n_gates_from_prev": 0,
        })

        for i in range(1, len(gates_sorted)):
            dy_px = gates_sorted[i][1] - gates_sorted[i - 1][1]
            if one_gate_px is not None and one_gate_px > 1e-3:
                # Estimate how many real gates this gap spans
                n_gates = max(1, round(dy_px / one_gate_px))
            else:
                n_gates = 1
            cumulative_m += n_gates * effective_spacing_m
            self.y_map.append({
                "y_px": float(gates_sorted[i][1]),
                "y_m": cumulative_m,
                "x_px": float(gates_sorted[i][0]),
                "n_gates_from_prev": n_gates,
            })

        # --- Step 3: compute ppm_y from corrected spacing ---
        ppm_vals = []
        for i in range(1, len(self.y_map)):
            dy_px = self.y_map[i]["y_px"] - self.y_map[i - 1]["y_px"]
            real_m = self.y_map[i]["y_m"] - self.y_map[i - 1]["y_m"]
            if dy_px > 1e-3 and real_m > 1e-3:
                ppm_vals.append(dy_px / real_m)
        if ppm_vals:
            self.ppm_y = float(np.median(ppm_vals))
            ppm_vals_sorted = sorted(ppm_vals)
            p10 = ppm_vals_sorted[max(0, int(0.1 * (len(ppm_vals_sorted) - 1)))]
            p90 = ppm_vals_sorted[min(len(ppm_vals_sorted) - 1, int(0.9 * (len(ppm_vals_sorted) - 1)))]
            self.ppm_bounds = (p10, p90)
        else:
            self.ppm_y = None
            self.ppm_bounds = None

        # --- Step 4: ppm_x = local ppm_y (isotropic at each depth) ---
        # In perspective projection, horizontal and vertical scale are the
        # same at any given depth. The old lateral-offset estimation was
        # fragile when detected gates weren't truly alternating.
        self.ppm_x = None  # will use local ppm_y in transform_point

        self.mode = "scale"
        self.H = np.eye(3)

        # Fit a simple centerline x = a*y + b to avoid lateral jumps
        try:
            ys = np.array([g["y_px"] for g in self.y_map], dtype=float)
            xs = np.array([g["x_px"] for g in self.y_map], dtype=float)
            if len(xs) >= 2:
                a, b = np.polyfit(ys, xs, 1)
                self.centerline_fit = (float(a), float(b))
            else:
                self.centerline_fit = None
        except Exception:
            self.centerline_fit = None

        # Print diagnostics
        total_gates_est = sum(g["n_gates_from_prev"] for g in self.y_map)
        ppm_text = f"{self.ppm_y:.2f}" if self.ppm_y else "n/a"
        ref_text = f"{one_gate_px:.1f}" if one_gate_px else "n/a"
        pitch_text = f", pitch={self.camera_pitch_deg:.1f}°" if self.camera_pitch_deg is not None else ""
        eff_text = f"{effective_spacing_m:.2f}" if effective_spacing_m else "n/a"
        print(f"✓ Scale mapping: {len(self.y_map)} detected gates, "
              f"~{total_gates_est} estimated real gates")
        print(f"  one-gate reference: {ref_text} px, "
              f"ppm_y ≈ {ppm_text} px/m, "
              f"course ≈ {cumulative_m:.0f}m{pitch_text}")
        if self.camera_pitch_deg is not None:
            print(f"  effective spacing: {eff_text}m (base {self.gate_spacing_m:.2f}m)")

        # ── PPM sanity check ──
        # A very low ppm means each pixel covers a huge real distance.
        # For typical ski race footage:
        #   - Close camera (slalom): ppm_y ≈ 3-10 px/m
        #   - Medium (GS broadcast): ppm_y ≈ 1.5-5 px/m
        #   - Wide shot (downhill):   ppm_y ≈ 0.5-3 px/m
        # If ppm_y < 0.5, the gates are likely too far or misdetected.
        # A 7px/frame movement at 30fps with ppm=1.0 means 210 m/s (756 km/h).
        if self.ppm_y is not None and self.ppm_y < 2.0:
            meters_per_px = 1.0 / self.ppm_y if self.ppm_y > 0 else float('inf')
            speed_at_7px = 7 * meters_per_px * 30 * 3.6  # km/h at 30fps
            print(f"  ⚠️  LOW PPM WARNING: {self.ppm_y:.2f} px/m means 1px ≈ {meters_per_px:.1f}m")
            print(f"     A 7px/frame movement → {speed_at_7px:.0f} km/h at 30fps")
            print(f"     This will produce unrealistic speeds. Possible causes:")
            print(f"     - Gates are very far from camera (wide-angle shot)")
            print(f"     - Gate spacing ({gate_spacing_m}m) doesn't match actual spacing")
            print(f"     - Gate detections include false positives")
