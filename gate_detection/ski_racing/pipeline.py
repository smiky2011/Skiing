"""
End-to-end ski racing analysis pipeline.
Video in -> gates + 2D skier trajectory out.

Focus: robust gate detection and 2D skier tracking.
3D coordinate estimation and physics validation have been removed because
they depend on gate detection quality that is not yet reliable enough.
Fix gate detection first, then re-introduce 3D/physics when the foundation
is solid.

Usage:
    from ski_racing.pipeline import SkiRacingPipeline
    pipeline = SkiRacingPipeline("models/gate_detector_best.pt")
    results = pipeline.process_video("race.mp4")
"""
import cv2
import json
import numpy as np
import subprocess
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

from .detection import GateDetector, TemporalGateTracker, CourseGateCounter
from .tracking import SkierTracker, KalmanSmoother, TrajectoryOutlierFilter


_GIT_INFO_CACHE = None


def _get_git_info():
    """
    Best-effort git metadata for traceability.

    Returns:
        (commit_sha_or_None, is_dirty_or_None)
    """
    global _GIT_INFO_CACHE
    if _GIT_INFO_CACHE is not None:
        return _GIT_INFO_CACHE

    repo_root = Path(__file__).resolve().parent.parent

    commit = None
    dirty = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = None

    try:
        # Dirty means tracked file changes (staged or unstaged). Untracked outputs
        # should not force dirty=True.
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).returncode
        dirty = bool(unstaged != 0 or staged != 0)
    except Exception:
        dirty = None

    _GIT_INFO_CACHE = (commit, dirty)
    return _GIT_INFO_CACHE


def _apply_kalman_oob_safety(trajectory, trajectory_raw, frame_width, frame_height, threshold=0.02):
    """
    Guard Kalman output against divergence.

    If the out-of-bounds (OOB) ratio exceeds threshold, revert to pre-Kalman
    trajectory. Otherwise clamp all points into frame bounds.
    """
    if not trajectory:
        return trajectory, False, 0.0

    oob_points = [
        pt for pt in trajectory
        if pt["x"] < 0 or pt["x"] >= frame_width or pt["y"] < 0 or pt["y"] >= frame_height
    ]
    oob_ratio = len(oob_points) / len(trajectory)
    if oob_ratio > threshold:
        print(
            f"  WARNING Kalman OOB safety: {len(oob_points)}/{len(trajectory)} points OOB "
            f"({oob_ratio:.1%}). Reverting to pre-Kalman trajectory."
        )
        return trajectory_raw, True, oob_ratio

    for pt in trajectory:
        pt["x"] = max(0.0, min(float(frame_width - 1), pt["x"]))
        pt["y"] = max(0.0, min(float(frame_height - 1), pt["y"]))
    return trajectory, False, oob_ratio


def _gate_stats_from_frame_history(frame_gate_history_full):
    """Build per-gate interpolation stats from frame-level gate history."""
    stats = {}
    for frame_payload in frame_gate_history_full.values():
        for gate_id, gate_info in frame_payload.items():
            gid = int(gate_id)
            entry = stats.setdefault(
                gid,
                {
                    "total_count": 0,
                    "interpolated_count": 0,
                    "observed_count": 0,
                    "positions": set(),
                },
            )
            entry["total_count"] += 1
            is_interp = bool(gate_info.get("is_interpolated", False))
            if is_interp:
                entry["interpolated_count"] += 1
            else:
                entry["observed_count"] += 1
            entry["positions"].add(
                (
                    round(float(gate_info.get("center_x", 0.0)), 3),
                    round(float(gate_info.get("base_y", 0.0)), 3),
                )
            )

    for entry in stats.values():
        total_count = int(entry["total_count"])
        interp_count = int(entry["interpolated_count"])
        entry["interp_rate"] = float(interp_count / total_count) if total_count > 0 else 0.0
        entry["unique_positions"] = int(len(entry["positions"]))
    return stats


def _ghost_gate_count_from_stats(gate_stats):
    """Count ghost gates: never observed OR fully interpolated at one fixed position."""
    ghost_count = 0
    for entry in gate_stats.values():
        real_obs = int(entry.get("observed_count", 0))
        interp_rate = float(entry.get("interp_rate", 0.0))
        unique_positions = int(entry.get("unique_positions", 0))
        if real_obs == 0 or (interp_rate == 1.0 and unique_positions == 1):
            ghost_count += 1
    return int(ghost_count)


def _filter_frame_gate_history_full(frame_gate_history_full, confirmed_ids):
    """Keep only confirmed gate IDs in frame-level full history."""
    filtered = {}
    for frame_idx, gates in frame_gate_history_full.items():
        kept = {gid: ginfo for gid, ginfo in gates.items() if int(gid) in confirmed_ids}
        if kept:
            filtered[int(frame_idx)] = kept
    return filtered


def _interp_rate_overall(frame_gate_history_full):
    """Compute interpolation rate over all exported gate-frame entries."""
    total = 0
    interpolated = 0
    for gates in frame_gate_history_full.values():
        for gate_info in gates.values():
            total += 1
            if bool(gate_info.get("is_interpolated", False)):
                interpolated += 1
    return float(interpolated / total) if total > 0 else 0.0


class SkiRacingPipeline:
    """
    Pipeline: video -> gates (2D pixel positions) + 2D skier trajectory.

    3D coordinate estimation and physics validation have been removed.
    The focus is on getting gate detection and 2D tracking right first.
    """

    SUPPORTED_DISCIPLINES = ("slalom", "giant_slalom")

    def __init__(
        self,
        gate_model_path,
        discipline=None,
        gate_full_track=False,
        outlier_filter=False,
        kalman_smooth=False,
        gate_init_mode="single_best",
        gate_consensus_min_support=3,
        stabilize=None,
    ):
        """
        Args:
            gate_model_path: Path to trained YOLOv8 gate detection weights.
            discipline: "slalom" or "giant_slalom". If None, auto-detected
                        from gate geometry per video.
            gate_full_track: Track gates across the full video (temporal refinement).
            outlier_filter: Apply MAD-based outlier rejection before Kalman.
            kalman_smooth: Apply Kalman smoothing to the 2D trajectory.
            gate_init_mode: Gate init strategy ("single_best" or "consensus").
            gate_consensus_min_support: Min sampled-frame support for consensus seed.
            stabilize: [DEPRECATED] Sets all three flags above.
        """
        self.gate_detector = GateDetector(gate_model_path)
        self.skier_tracker = SkierTracker()
        if discipline is not None and discipline not in self.SUPPORTED_DISCIPLINES:
            raise ValueError(
                f"Unsupported discipline '{discipline}'. "
                f"Supported: {', '.join(self.SUPPORTED_DISCIPLINES)}"
            )
        self.discipline = discipline
        self._discipline_explicit = discipline is not None
        self.discipline_source = "explicit" if self._discipline_explicit else "auto_detected"
        if stabilize is not None:
            warnings.warn(
                "stabilize= is deprecated; use gate_full_track=, outlier_filter=, "
                "kalman_smooth= instead. Will be removed after 2026-04-30.",
                DeprecationWarning,
                stacklevel=2,
            )
            gate_full_track = stabilize
            outlier_filter = stabilize
            kalman_smooth = stabilize
        self.gate_full_track = bool(gate_full_track)
        self.outlier_filter = bool(outlier_filter)
        self.kalman_smooth = bool(kalman_smooth)
        gate_init_mode = str(gate_init_mode)
        if gate_init_mode not in {"single_best", "consensus"}:
            raise ValueError(
                f"Unsupported gate_init_mode '{gate_init_mode}'. "
                "Supported: single_best, consensus"
            )
        self.gate_init_mode = gate_init_mode
        self.gate_consensus_min_support = max(1, int(gate_consensus_min_support))

    @staticmethod
    def classify_discipline(gates, frame_height):
        """
        Classify discipline from detected gates in the best frame.

        Rules:
          - >=6 visible gates: slalom
          - <=3 visible gates: giant_slalom
          - 4-5 gates: use median consecutive Y-gap
              gap < 15% of frame height -> slalom
              else -> giant_slalom
        """
        gates = gates or []
        gate_count = len(gates)
        sorted_gates = sorted(gates, key=lambda g: float(g.get("base_y", 0.0)))
        ys = [float(g.get("base_y", 0.0)) for g in sorted_gates]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 1e-6]
        median_gap_px = float(np.median(gaps)) if gaps else None
        gap_ratio = None
        if median_gap_px is not None and frame_height and frame_height > 0:
            gap_ratio = float(median_gap_px / frame_height)

        if gate_count >= 6:
            discipline = "slalom"
            rule = "gate_count>=6"
        elif gate_count <= 3:
            discipline = "giant_slalom"
            rule = "gate_count<=3"
        else:
            threshold_ratio = 0.15
            if gap_ratio is not None and gap_ratio < threshold_ratio:
                discipline = "slalom"
                rule = "median_gap<15%_height"
            else:
                discipline = "giant_slalom"
                rule = "median_gap>=15%_height"

        return {
            "discipline": discipline,
            "gate_count": int(gate_count),
            "median_gap_px": median_gap_px,
            "median_gap_ratio": gap_ratio,
            "rule": rule,
        }

    def process_video(
        self,
        video_path,
        output_dir="artifacts/outputs",
        gate_conf=0.25,
        gate_iou=0.45,
        skier_conf=0.25,
        gate_search_frames=300,
        gate_search_stride=3,
        gate_track_frames=120,
        gate_track_stride=3,
        gate_track_min_obs=3,
        frame_stride=1,
        max_frames=None,
        max_jump=None,
        kalman_process_noise=None,
        course_gate_count=True,
        course_gate_conf=None,
        course_gate_stride=None,
        course_gate_min_hits=3,
        course_gate_track_missing_max=8,
        course_gate_fragment_merge_gap_max=45,
        course_gate_match_thresh_ratio=0.06,
    ):
        """
        Run gate detection and 2D skier tracking on a video.

        Note: 3D coordinate transformation and physics validation have been
        removed. The output contains only 2D pixel-space gate positions and
        skier trajectory. Fix gate detection quality first before re-adding
        metric-space outputs.

        Args:
            video_path: Path to race video.
            output_dir: Directory for saving results.
            gate_conf: Gate detection confidence threshold (default 0.25).
            gate_iou: Gate detection NMS IoU threshold (default 0.45).
            skier_conf: Skier detection confidence threshold.
            gate_search_frames: Max frames to scan for best gate frame.
            gate_search_stride: Frame stride during gate search.
            gate_track_frames: Frames to use for gate temporal refinement.
            gate_track_stride: Stride for gate tracking pass.
            gate_track_min_obs: Min observations to keep a tracked gate.
            frame_stride: Process every Nth frame for skier tracking.
            max_frames: Max frames to process for skier tracking.
            max_jump: Max pixel jump between consecutive skier positions.
            kalman_process_noise: Override Kalman process noise sigma.

        Returns:
            Dictionary with gate detections and 2D skier trajectory.
        """
        video_path = str(video_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get video info
        video_info = self.skier_tracker.get_video_info(video_path)
        fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        frame_width = int(video_info["width"])
        frame_height = int(video_info["height"])

        mode_flags = []
        if self.gate_full_track:
            mode_flags.append("gate-full-track")
        if self.outlier_filter:
            mode_flags.append("outlier-filter")
        if self.kalman_smooth:
            mode_flags.append("kalman")
        mode_label = f" [{' '.join(mode_flags)}]" if mode_flags else ""
        print(f"Processing {Path(video_path).name}...{mode_label}")
        print(f"  Video: {video_info['width']}x{video_info['height']} @ {fps:.1f} fps, "
              f"{total_frames} frames")

        n_steps = 4 if self.kalman_smooth else 3
        phase_timings = {
            "gate_detection_initial_frame_search": 0.0,
            "gate_tracking_pass": 0.0,
            "skier_tracking_full_video_pass": 0.0,
            "kalman_smoothing": 0.0,
        }

        # ─── Step 1: Detect gates from first/best frame ───
        phase_t0 = time.time()
        print(f"  [1/{n_steps}] Detecting gates...")
        gates = self.gate_detector.detect_from_first_frame(
            video_path,
            conf=gate_conf,
            iou=gate_iou,
        )
        if len(gates) < 4 and gate_search_frames > 0:
            print("         Low gate count; scanning additional frames...")
            if self.gate_init_mode == "consensus":
                gates = self.gate_detector.detect_from_consensus(
                    video_path,
                    conf=gate_conf,
                    iou=gate_iou,
                    max_frames=gate_search_frames,
                    stride=gate_search_stride,
                    min_support=self.gate_consensus_min_support,
                    frame_height=frame_height,
                )
            else:
                gates = self.gate_detector.detect_from_best_frame(
                    video_path,
                    conf=gate_conf,
                    iou=gate_iou,
                    max_frames=gate_search_frames,
                    stride=gate_search_stride,
                )

        # Cluster duplicate pole detections
        if len(gates) >= 2:
            gates = self._cluster_gates_by_y(
                gates,
                y_thresh=15.0,
                frame_height=frame_height,
                frame_width=frame_width,
            )
        gates_initial = [dict(g) for g in gates]
        phase_timings["gate_detection_initial_frame_search"] = float(time.time() - phase_t0)

        print(f"         Found {len(gates)} gates after clustering")

        # ─── Discipline auto-detection ───
        discipline_info = None
        if not self._discipline_explicit:
            discipline_info = self.classify_discipline(gates_initial, video_info["height"])
            self.discipline = discipline_info["discipline"]
            self.discipline_source = "auto_detected"
            gate_count = discipline_info["gate_count"]
            median_gap_px = discipline_info["median_gap_px"]
            gap_ratio = discipline_info["median_gap_ratio"]
            if median_gap_px is not None and gap_ratio is not None:
                print(f"         Auto-detected: {self.discipline} "
                      f"({gate_count} gates visible, median gap {median_gap_px:.0f}px "
                      f"= {gap_ratio * 100:.1f}% of frame height)")
            else:
                print(f"         Auto-detected: {self.discipline} ({gate_count} gates visible)")
        else:
            self.discipline_source = "explicit"

        # ─── Step 1b: Per-frame gate tracking ───
        frame_gate_history = None
        frame_gate_history_full = None
        gate_tracking_quality = {
            "confirmed_gate_count": 0,
            "ghost_gate_count_raw": 0,
            "ghost_gate_count": 0,
            "interp_rate_overall": 0.0,
            "provisional_dropped_count": 0,
        }

        phase_t0 = time.time()
        if self.gate_full_track and len(gates) >= 2:
            print(f"  [2/{n_steps}] Tracking gates across all frames...")
            gates, frame_gate_history, frame_gate_history_full, _, gate_tracking_quality = self._track_gates_full_video(
                video_path,
                gates,
                conf=gate_conf,
                iou=gate_iou,
                stride=gate_track_stride,
                min_obs=gate_track_min_obs,
            )
            phase_timings["gate_tracking_pass"] = float(time.time() - phase_t0)
            if len(gates) >= 2:
                gates = self._cluster_gates_by_y(
                    gates,
                    y_thresh=15.0,
                    frame_height=frame_height,
                    frame_width=frame_width,
                )
            print(f"         {len(gates)} stable gates after temporal tracking")
        elif gate_track_frames > 0 and len(gates) >= 2:
            print("         Refining gates with temporal tracking...")
            gates = self._refine_gates_with_tracking(
                video_path,
                gates,
                conf=gate_conf,
                iou=gate_iou,
                max_frames=gate_track_frames,
                stride=gate_track_stride,
                min_obs=gate_track_min_obs,
            )
            phase_timings["gate_tracking_pass"] = float(time.time() - phase_t0)
            if len(gates) >= 2:
                gates = self._cluster_gates_by_y(
                    gates,
                    y_thresh=15.0,
                    frame_height=frame_height,
                    frame_width=frame_width,
                )
            print(f"         {len(gates)} stable gates after temporal tracking")

        # ─── Step 1c: Course-wide gate counting ───
        _course_conf = max(gate_conf - 0.10, 0.20) if course_gate_conf is None else course_gate_conf
        _course_stride = (2 if frame_height <= 1080 else 3) if course_gate_stride is None else course_gate_stride
        course_gate_result = {"course_gates": [], "course_gates_count": 0, "course_gate_counting": {}}
        if course_gate_count:
            counter = CourseGateCounter(
                detector=self.gate_detector,
                conf=float(_course_conf),
                iou=float(gate_iou),
                stride=int(_course_stride),
                min_hits=int(course_gate_min_hits),
                track_missing_max=int(course_gate_track_missing_max),
                fragment_merge_gap_max=int(course_gate_fragment_merge_gap_max),
                match_thresh_ratio=float(course_gate_match_thresh_ratio),
            )
            _cr = counter.count(video_path, frame_width, frame_height)
            course_gate_result = {
                "course_gates": _cr["course_gates"],
                "course_gates_count": int(_cr["course_gates_count"]),
                "course_gate_counting": _cr["diagnostics"],
            }
            print(f"  Estimated total course gates: {_cr['course_gates_count']}")

        # NOTE: We no longer hard-fail on <2 gates. Gate detection is a work
        # in progress. We record however many gates we found and continue
        # with skier tracking so at least the 2D trajectory is saved.
        if len(gates) < 2:
            print(f"  ⚠️  Only {len(gates)} gate(s) detected. "
                  f"Gate detection needs improvement. Continuing with skier tracking only.")

        # ─── Step 2/3: Track skier ───
        step_n = 3 if self.kalman_smooth else 2
        phase_t0 = time.time()
        print(f"  [{step_n}/{n_steps}] Tracking skier...")
        trajectory_2d = self.skier_tracker.track_video(
            video_path,
            method="bytetrack",
            frame_stride=frame_stride,
            max_frames=max_frames,
            max_jump=max_jump,
            conf=skier_conf,
            gates=gates,
        )
        phase_timings["skier_tracking_full_video_pass"] = float(time.time() - phase_t0)
        tracking_diag = self.skier_tracker.get_last_tracking_stats()
        bytetrack_cov = float(tracking_diag.get("bytetrack_coverage", 0.0))
        track_switches = int(tracking_diag.get("track_id_switches", 0))
        print(f"         Tracked {len(trajectory_2d)} positions "
              f"({len(trajectory_2d)/total_frames*100:.0f}% coverage)")
        if tracking_diag.get("method_requested") == "bytetrack":
            print(f"         ByteTrack coverage: {bytetrack_cov:.1%}, "
                  f"ID switches: {track_switches}")

        trajectory_2d_original = [dict(pt) for pt in trajectory_2d]
        outlier_info = {"outlier_count": 0, "outlier_frames": []}

        # Pre-smoothing outlier rejection
        if self.outlier_filter and len(trajectory_2d) > 0:
            outlier_filter = TrajectoryOutlierFilter(window=5, mad_threshold=3.0)
            trajectory_2d, outlier_info = outlier_filter.filter(trajectory_2d)
            outlier_count = int(outlier_info.get("outlier_count", 0))
            if outlier_count > 0:
                print(f"         Outlier filter: corrected {outlier_count} frames")
            if total_frames > 0 and outlier_count > 0.05 * total_frames:
                print(f"  ⚠️  Tracking quality warning: {outlier_count}/{total_frames} "
                      f"frames flagged as outliers (>5%).")

        # ─── Kalman smoothing ───
        trajectory_2d_raw = None
        kalman_reverted = False
        kalman_oob_ratio = 0.0
        if self.kalman_smooth and len(trajectory_2d) > 2:
            step_n = 4
            phase_t0 = time.time()
            print(f"  [{step_n}/{n_steps}] Smoothing 2D trajectory with Kalman filter...")
            trajectory_2d_raw = [dict(pt) for pt in trajectory_2d]
            kf = KalmanSmoother(
                fps=fps,
                discipline=self.discipline,
                process_noise=kalman_process_noise,
            )
            trajectory_2d = kf.smooth(trajectory_2d)
            trajectory_2d, kalman_reverted, kalman_oob_ratio = _apply_kalman_oob_safety(
                trajectory=trajectory_2d,
                trajectory_raw=trajectory_2d_raw,
                frame_width=frame_width,
                frame_height=frame_height,
                threshold=0.02,
            )
            phase_timings["kalman_smoothing"] = float(time.time() - phase_t0)

        print("  Runtime profile (seconds):")
        print(f"    Gate detection (initial frame search): {phase_timings['gate_detection_initial_frame_search']:.3f}")
        print(f"    Gate tracking pass: {phase_timings['gate_tracking_pass']:.3f}")
        print(f"    Skier tracking (full video pass): {phase_timings['skier_tracking_full_video_pass']:.3f}")
        print(f"    Kalman smoothing: {phase_timings['kalman_smoothing']:.3f}")

        # Compile results
        git_commit, git_dirty = _get_git_info()
        pipeline_params = {
            "gate_full_track": bool(self.gate_full_track),
            "outlier_filter": bool(self.outlier_filter),
            "kalman_smooth": bool(self.kalman_smooth),
            "gate_init_mode": self.gate_init_mode,
            "gate_consensus_min_support": int(self.gate_consensus_min_support),
            "stabilize": bool(
                self.gate_full_track and self.outlier_filter and self.kalman_smooth
            ),
            "gate_conf": float(gate_conf),
            "gate_iou": float(gate_iou),
            "skier_conf": float(skier_conf),
            "gate_search_frames": int(gate_search_frames),
            "gate_search_stride": int(gate_search_stride),
            "gate_track_frames": int(gate_track_frames),
            "gate_track_stride": int(gate_track_stride),
            "gate_track_min_obs": int(gate_track_min_obs),
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames) if max_frames is not None else None,
            "max_jump_px": float(max_jump) if max_jump is not None else None,
            "kalman_process_noise": float(kalman_process_noise) if kalman_process_noise is not None else None,
            "course_gate_count": bool(course_gate_count),
            "course_gate_conf": float(_course_conf),
            "course_gate_stride": int(_course_stride),
            "course_gate_min_hits": int(course_gate_min_hits),
            "course_gate_track_missing_max": int(course_gate_track_missing_max),
            "course_gate_fragment_merge_gap_max": int(course_gate_fragment_merge_gap_max),
            "course_gate_match_thresh_ratio": float(course_gate_match_thresh_ratio),
            # Removed: 3D transform, physics, camera motion params
            "trajectory_3d": "disabled",
            "physics_validation": "disabled",
        }
        results = {
            "video": video_path,
            "video_info": video_info,
            "discipline": self.discipline,
            "discipline_source": self.discipline_source,
            "discipline_detection": discipline_info,
            "gate_conf": float(gate_conf),
            "gate_iou": float(gate_iou),
            "stabilized": bool(self.kalman_smooth and not kalman_reverted),
            "kalman_attempted": bool(self.kalman_smooth),
            "kalman_reverted_oob": bool(kalman_reverted),
            "kalman_oob_ratio": round(float(kalman_oob_ratio), 4),
            "gates": gates,
            "gates_count": int(len(gates)),
            "course_gates": course_gate_result["course_gates"],
            "course_gates_count": course_gate_result["course_gates_count"],
            "course_gate_counting": course_gate_result["course_gate_counting"],
            # 2D-first sprint: 3D transform and physics validation are disabled.
            # These sentinel strings signal "not computed" to downstream consumers
            # (visualize.py, run_eval.py).  When 3D is re-enabled the values will
            # be replaced with a list of {frame, x, y} dicts / a metrics dict.
            "trajectory_3d": "disabled",
            "physics_validation": "disabled",
            "trajectory_2d": trajectory_2d,
            "timestamp": datetime.now().isoformat(),
            "outlier_count": int(outlier_info.get("outlier_count", 0)),
            "outlier_frames": [int(f) for f in outlier_info.get("outlier_frames", [])],
            "bytetrack_coverage": float(tracking_diag.get("bytetrack_coverage", 0.0)),
            "track_id_switches": int(tracking_diag.get("track_id_switches", 0)),
            "tracking_diagnostics": tracking_diag,
            "gate_tracking_quality": gate_tracking_quality,
            "runtime_profile_sec": phase_timings,
            "argv": list(sys.argv) if hasattr(sys, "argv") else None,
            "pipeline_params": pipeline_params,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        }

        # Add stabilization-specific data
        if self.kalman_smooth:
            results["kalman_smoothed"] = bool(not kalman_reverted)
            if trajectory_2d_raw is not None:
                results["trajectory_2d_raw"] = trajectory_2d_raw
        if self.outlier_filter or self.kalman_smooth:
            if trajectory_2d_original is not None:
                results["trajectory_2d_original"] = trajectory_2d_original
        if self.gate_full_track:
            if frame_gate_history:
                results["frames_detected"] = self._build_frame_records(frame_gate_history)
                results["gate_track_stride"] = gate_track_stride
            if frame_gate_history_full:
                results["frames"] = self._build_frame_records(frame_gate_history_full)

        # Save results
        output_path = Path(output_dir) / f"{Path(video_path).stem}_analysis.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✓ Results saved to {output_path}")
        return results

    def _build_frame_records(self, frame_gate_history):
        """
        Build per-frame gate records for JSON output.
        """
        frames = []
        for frame_idx in sorted(frame_gate_history.keys()):
            gates = []
            for gate_id, payload in frame_gate_history[frame_idx].items():
                if isinstance(payload, dict):
                    gate_info = dict(payload)
                    gate_info.setdefault("gate_id", int(gate_id))
                    # Normalize numeric fields
                    if "center_x" in gate_info:
                        gate_info["center_x"] = float(gate_info["center_x"])
                    if "base_y" in gate_info:
                        gate_info["base_y"] = float(gate_info["base_y"])
                    if "confidence" in gate_info:
                        gate_info["confidence"] = float(gate_info["confidence"])
                    if "missing_frames" in gate_info:
                        gate_info["missing_frames"] = int(gate_info["missing_frames"])
                    gates.append(gate_info)
                else:
                    cx, by = payload
                    gates.append({
                        "gate_id": int(gate_id),
                        "center_x": float(cx),
                        "base_y": float(by),
                    })
            frames.append({
                "frame": int(frame_idx),
                "gates": gates,
            })
        return frames

    def _track_gates_full_video(self, video_path, initial_gates, conf=0.35, iou=0.55, stride=3, min_obs=3):
        """
        Track gates across the entire video to build per-frame gate history.

        Returns:
            Tuple of (
                refined_gates,
                frame_gate_history,
                frame_gate_history_full,
                baseline_gates,
                gate_tracking_quality
            )
            - refined_gates: List of gate dicts with median positions
            - frame_gate_history: {frame_idx: {gate_id: (cx, by)}} (detected-only)
            - frame_gate_history_full: {frame_idx: {gate_id: gate_info}} (includes interpolated)
            - baseline_gates: {gate_id: (cx, by)} for camera compensation
            - gate_tracking_quality: dict with ghost/interpolation summary
        """
        tracker = TemporalGateTracker(max_missing_frames=30, match_threshold=60.0)
        tracker.initialize(initial_gates)
        min_obs = max(1, int(min_obs))
        n_initial = len(initial_gates)

        history_for_median = {}  # gate_id -> [(cx, by, class, name)]
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        stride = max(1, int(stride))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            detections = self.gate_detector.detect_in_frame(frame, conf=conf, iou=iou)
            tracked = tracker.update_with_frame_idx(detections, frame_idx)

            for g in tracked:
                if not g.get("is_interpolated", False):
                    history_for_median.setdefault(g["gate_id"], []).append(
                        (g["center_x"], g["base_y"], g["class"],
                         g.get("class_name", "gate"), g.get("confidence", 0.0))
                    )

            processed += 1
            frame_idx += 1

        cap.release()

        # Build refined gates (median positions)
        refined = []
        baseline = {}
        for gate_id, items in history_for_median.items():
            if len(items) < min_obs:
                continue
            xs = [i[0] for i in items]
            ys = [i[1] for i in items]
            cls = items[0][2]
            name = items[0][3]
            confs = [i[4] for i in items if len(i) > 4]
            med_x = float(np.median(xs))
            med_y = float(np.median(ys))
            med_conf = float(np.median(confs)) if confs else 0.0
            refined.append({
                "gate_id": gate_id,
                "center_x": med_x,
                "base_y": med_y,
                "class": int(cls),
                "class_name": name,
                "confidence": med_conf,
            })
            baseline[gate_id] = (med_x, med_y)

        refined.sort(key=lambda g: g["base_y"])
        frame_gate_history = tracker.get_frame_history()
        frame_gate_history_full_raw = tracker.get_frame_history_full()

        confirmed_ids = {
            int(gate_id)
            for gate_id, items in history_for_median.items()
            if len(items) >= min_obs
        }
        frame_gate_history_full = _filter_frame_gate_history_full(
            frame_gate_history_full_raw,
            confirmed_ids=confirmed_ids,
        )
        raw_stats = _gate_stats_from_frame_history(frame_gate_history_full_raw)
        filtered_stats = _gate_stats_from_frame_history(frame_gate_history_full)
        gate_tracking_quality = {
            "confirmed_gate_count": int(len(confirmed_ids)),
            "ghost_gate_count_raw": int(_ghost_gate_count_from_stats(raw_stats)),
            "ghost_gate_count": int(_ghost_gate_count_from_stats(filtered_stats)),
            "interp_rate_overall": float(_interp_rate_overall(frame_gate_history_full)),
            "provisional_dropped_count": int(max(0, len(initial_gates) - len(confirmed_ids))),
        }

        print(f"         Tracked {len(refined)} gates across {processed} frames "
              f"({len(frame_gate_history)} frames with detections)")
        if gate_tracking_quality["ghost_gate_count_raw"] > 0:
            print(
                f"         Filtered {gate_tracking_quality['ghost_gate_count_raw']} "
                "ghost gate track(s) from frame history export"
            )

        if not refined:
            return initial_gates, frame_gate_history, frame_gate_history_full, {}, gate_tracking_quality

        # Safety: if pruning dropped >50% of initial gates, fall back to
        # initial seeds merged with confirmed gates to avoid over-pruning
        # sparse observations.
        if n_initial > 0 and len(refined) < n_initial * 0.5:
            print(
                f"         Full-track safety: {len(refined)}/{n_initial} gates survived "
                f"(< 50%). Recovering initial seeds."
            )
            confirmed_ys = {round(g["base_y"], 0) for g in refined}
            for seed in initial_gates:
                seed_y = round(seed["base_y"], 0)
                if not any(abs(seed_y - cy) < 20 for cy in confirmed_ys):
                    refined.append(dict(seed))
                    confirmed_ys.add(seed_y)
            refined.sort(key=lambda g: g["base_y"])

        return refined, frame_gate_history, frame_gate_history_full, baseline, gate_tracking_quality

    def _refine_gates_with_tracking(
        self,
        video_path,
        initial_gates,
        conf=0.35,
        iou=0.55,
        max_frames=120,
        stride=3,
        min_obs=3,
    ):
        """
        Run temporal gate tracking over early frames to stabilize detections.
        Returns median gate positions for each tracked gate.
        """
        if not initial_gates:
            return initial_gates

        tracker = TemporalGateTracker()
        tracker.initialize(initial_gates)

        history = {}
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        stride = max(1, int(stride))
        max_frames = max(1, int(max_frames))

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            detections = self.gate_detector.detect_in_frame(frame, conf=conf, iou=iou)
            tracked = tracker.update(detections, frame_idx=frame_idx)
            for g in tracked:
                history.setdefault(g["gate_id"], []).append(
                    (g["center_x"], g["base_y"], g["class"],
                     g.get("class_name", "gate"), g.get("confidence", 0.0))
                )

            frame_idx += 1

        cap.release()

        refined = []
        for gate_id, items in history.items():
            if len(items) < min_obs:
                continue
            xs = [i[0] for i in items]
            ys = [i[1] for i in items]
            cls = items[0][2]
            name = items[0][3]
            confs = [i[4] for i in items if len(i) > 4]
            refined.append({
                "gate_id": gate_id,
                "center_x": float(np.median(xs)),
                "base_y": float(np.median(ys)),
                "class": int(cls),
                "class_name": name,
                "confidence": float(np.median(confs)) if confs else 0.0,
            })

        if not refined:
            return initial_gates

        refined.sort(key=lambda g: g["base_y"])
        return refined

    def _cluster_gates_by_y(self, gates, y_thresh=12.0, frame_height=None, frame_width=None):
        """
        Cluster detections with similar base_y AND center_x to avoid
        duplicate gate poles while preserving distinct gates.

        Uses both Y and X proximity checks. Two detections are merged only
        if they are close in Y *and* not too far apart in X. The X threshold
        is scaled from frame width to avoid aspect-ratio sensitivity.
        A max-cluster-size guard prevents long gate chains from collapsing.
        """
        if not gates:
            return gates

        gates_sorted = sorted(gates, key=lambda g: g["base_y"])

        # Adaptive Y threshold: 3% of frame height (conservative — two
        # poles of the same gate are typically 10-25px apart, even at 2160p
        # that is ~65px = 3%).
        if frame_height and frame_height > 0:
            adaptive_y = 0.03 * frame_height
        else:
            adaptive_y = y_thresh

        # Dynamic threshold from gap statistics (30% of median gap)
        gaps = []
        for i in range(1, len(gates_sorted)):
            dy = gates_sorted[i]["base_y"] - gates_sorted[i - 1]["base_y"]
            if dy > 1e-3:
                gaps.append(dy)
        if gaps:
            median_gap = float(np.median(gaps))
            gap_thresh = 0.30 * median_gap
        else:
            gap_thresh = y_thresh

        # Use the SMALLER of the two thresholds (conservative merge)
        y_merge_thresh = min(adaptive_y, gap_thresh)

        # X threshold: gates at the same Y but very different X are distinct
        # poles, not duplicates. Scale from frame width (0.025xw -> 32px at
        # 720p, 96px at 4K). Floor of 20px guards very narrow inputs.
        # Same-pole YOLO duplicates are typically dx < 15px; distinct poles
        # on 720p footage are ~40-55px apart - well above the 32px threshold.
        if frame_width and frame_width > 0:
            x_merge_thresh = max(0.025 * frame_width, 20.0)
        else:
            x_merge_thresh = 50.0

        # Max cluster size: prevent chains of many gates from collapsing
        max_cluster_size = 3

        clusters = []
        current = [gates_sorted[0]]

        for g in gates_sorted[1:]:
            avg_y = sum(x["base_y"] for x in current) / len(current)
            avg_x = sum(x["center_x"] for x in current) / len(current)
            y_close = abs(g["base_y"] - avg_y) <= y_merge_thresh
            x_close = abs(g["center_x"] - avg_x) <= x_merge_thresh
            under_limit = len(current) < max_cluster_size
            if y_close and x_close and under_limit:
                current.append(g)
            else:
                clusters.append(current)
                current = [g]

        if current:
            clusters.append(current)

        merged = []
        for group in clusters:
            xs = [g["center_x"] for g in group]
            ys = [g["base_y"] for g in group]
            cls_list = [g.get("class", 0) for g in group]
            cls = max(set(cls_list), key=cls_list.count)
            name = next((g.get("class_name", "gate") for g in group if g.get("class", 0) == cls), "gate")
            confs = [g.get("confidence", 0.0) for g in group if g.get("confidence") is not None]
            merged.append({
                "center_x": float(np.median(xs)),
                "base_y": float(np.median(ys)),
                "class": int(cls),
                "class_name": name,
                "confidence": float(np.median(confs)) if confs else 0.0,
            })

        return merged


def main():
    """CLI entry point for processing videos."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ski racing video (gate detection + 2D tracking)"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--gate-model", required=True, help="Path to gate detector model")
    parser.add_argument("--discipline", default=None,
                        choices=["slalom", "giant_slalom"],
                        help="Discipline. If omitted, auto-detects from gates")
    parser.add_argument("--output-dir", default="artifacts/outputs", help="Output directory")
    parser.add_argument("--gate-conf", type=float, default=0.25,
                        help="Gate detection confidence threshold (default 0.25)")
    parser.add_argument("--gate-iou", type=float, default=0.45,
                        help="Gate detection NMS IoU threshold (default 0.45)")
    parser.add_argument("--skier-conf", type=float, default=0.25,
                        help="Skier detection confidence threshold")
    parser.add_argument("--gate-init-mode", default="single_best",
                        choices=["single_best", "consensus"],
                        help="Gate initialization mode: single best frame or cross-frame consensus.")
    parser.add_argument("--gate-consensus-min-support", type=int, default=3,
                        help="Minimum sampled-frame support for consensus gate seeds (default 3).")
    parser.add_argument("--gate-full-track", action="store_true",
                        help="Track gates across the full video (temporal refinement over all frames).")
    parser.add_argument("--outlier-filter", action="store_true",
                        help="Apply MAD-based outlier rejection to trajectory before Kalman smoothing.")
    parser.add_argument("--kalman", action="store_true",
                        help="Apply Kalman smoothing to 2D trajectory. Avoid on phone clips prone to OOB drift.")
    parser.add_argument("--stabilize", action="store_true",
                        help="[DEPRECATED] Alias for --gate-full-track --outlier-filter --kalman. "
                             "Use individual flags. Will be removed after 2026-04-30.")
    parser.add_argument("--kalman-q", type=float, default=None,
                        help="Override Kalman process noise (Q sigma_a in px/s^2)")
    parser.add_argument("--gate-search-frames", type=int, default=300,
                        help="Max frames to scan for best gate detection frame (default 300)")
    parser.add_argument("--gate-search-stride", type=int, default=3,
                        help="Frame stride during gate search scan (default 3)")
    args = parser.parse_args()

    if args.stabilize:
        print(
            "WARNING --stabilize is deprecated; use --gate-full-track, --outlier-filter, "
            "--kalman instead. Removal target: 2026-04-30."
        )
        args.gate_full_track = True
        args.outlier_filter = True
        args.kalman = True

    pipeline = SkiRacingPipeline(
        gate_model_path=args.gate_model,
        discipline=args.discipline,
        gate_init_mode=args.gate_init_mode,
        gate_consensus_min_support=args.gate_consensus_min_support,
        gate_full_track=args.gate_full_track,
        outlier_filter=args.outlier_filter,
        kalman_smooth=args.kalman,
    )

    results = pipeline.process_video(
        video_path=args.video,
        output_dir=args.output_dir,
        gate_conf=args.gate_conf,
        gate_iou=args.gate_iou,
        skier_conf=args.skier_conf,
        kalman_process_noise=args.kalman_q,
        gate_search_frames=args.gate_search_frames,
        gate_search_stride=args.gate_search_stride,
    )


if __name__ == "__main__":
    main()
