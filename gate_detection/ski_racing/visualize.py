"""
Visualization module for ski racing analysis.
Creates trajectory overlays, bird's-eye views, speed profiles,
and summary figures.
"""
import cv2
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def create_demo_video(video_path, analysis_path, output_path,
                      gate_model_path=None, live_gate_stride=3):
    """
    Create demo video with trajectory and gate overlay.

    Args:
        video_path: Path to original video.
        analysis_path: Path to analysis JSON (from pipeline).
        output_path: Path for output video.
        gate_model_path: Path to gate detector model. When provided, gates are
            re-detected live on each frame so positions follow the camera exactly.
        live_gate_stride: Run live inference every N frames (default 3).
            Lower = more accurate but slower rendering.
    """
    with open(analysis_path, "r") as f:
        analysis = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build frame -> position lookup (final and raw/debug)
    trajectory = {
        p["frame"]: (int(p["x"]), int(p["y"]))
        for p in analysis.get("trajectory_2d", [])
    }
    raw_source = analysis.get("trajectory_2d_raw") or analysis.get("trajectory_2d_original") or []
    trajectory_raw = {
        p["frame"]: (int(p["x"]), int(p["y"]))
        for p in raw_source
    }
    outlier_frames = set(int(f) for f in analysis.get("outlier_frames", []))

    # Live detector: re-detect gates on each frame so positions follow the camera
    live_detector = None
    if gate_model_path is not None:
        from ski_racing.detection import GateDetector
        live_detector = GateDetector(gate_model_path)
    live_cache = []        # last live-detected gate positions
    live_frame_at = -999   # frame index of last live inference run

    # Static/per-frame fallback (used when live_detector is None)
    frame_gate_lookup = None
    last_gate_positions = None
    if live_detector is None:
        if "frames" in analysis:
            frame_gate_lookup = {}
            for frame_entry in analysis.get("frames", []):
                frame_idx = frame_entry.get("frame")
                gates = []
                for g in frame_entry.get("gates", []):
                    if "center_x" in g and "base_y" in g:
                        gates.append((int(g["center_x"]), int(g["base_y"]), g))
                frame_gate_lookup[frame_idx] = gates
        else:
            gate_positions = [
                (int(g["center_x"]), int(g["base_y"]), g)
                for g in analysis.get("gates", [])
            ]

    gate_positions = []
    frame_num = 0
    trail_points = []
    raw_trail_points = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resolve gate positions for this frame
        if live_detector is not None:
            # Live mode: re-detect every live_gate_stride frames, cache between
            if frame_num - live_frame_at >= live_gate_stride:
                dets = live_detector.detect_in_frame(frame, conf=0.20, iou=0.45)
                live_cache = [(int(d["center_x"]), int(d["base_y"]), d) for d in dets]
                live_frame_at = frame_num
            gate_positions = live_cache
        elif frame_gate_lookup is not None:
            # Per-frame data: only show gates that were actually detected
            # (skip is_interpolated=True — those are stale positions from when
            # the gate wasn't visible, not real current detections)
            raw = frame_gate_lookup.get(frame_num, None)
            if raw is not None:
                gate_positions = [
                    (gx, gy, gm) for gx, gy, gm in raw
                    if not (isinstance(gm, dict) and gm.get("is_interpolated", False))
                ]
                if gate_positions:
                    last_gate_positions = gate_positions
            elif last_gate_positions is not None:
                gate_positions = last_gate_positions
            else:
                gate_positions = []
        # else: static gate_positions set above the loop, unchanged each frame

        # Draw gates
        for i, (gx, gy, gmeta) in enumerate(gate_positions):
            if isinstance(gmeta, dict) and "class" in gmeta:
                color = (0, 0, 255) if int(gmeta["class"]) % 2 == 0 else (255, 0, 0)
            elif isinstance(gmeta, dict) and "gate_id" in gmeta:
                color = (0, 0, 255) if int(gmeta["gate_id"]) % 2 == 0 else (255, 0, 0)
            else:
                color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)  # Red / Blue
            cv2.circle(frame, (gx, gy), 10, color, -1)
            cv2.circle(frame, (gx, gy), 15, color, 2)

        # Build trail
        if frame_num in trajectory:
            trail_points.append(trajectory[frame_num])
        if frame_num in trajectory_raw:
            raw_trail_points.append(trajectory_raw[frame_num])

        # Draw raw/debug trail first (thin orange)
        for i in range(1, len(raw_trail_points)):
            cv2.line(frame, raw_trail_points[i - 1], raw_trail_points[i], (0, 140, 255), 1)

        # Draw filtered trajectory trail
        for i in range(1, len(trail_points)):
            cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 255, 255), 3)

        # Text overlay
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Gates: {len(gate_positions)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Outliers: {len(outlier_frames)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 255), 2)

        course_total = analysis.get("course_gates_count")
        y_cursor = 150
        if course_total is not None:
            cv2.putText(frame, f"Course Total: {course_total}",
                        (10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_cursor += 40
        else:
            y_cursor = 145

        # Physics status — guard against sentinel string "disabled"
        physics = analysis.get("physics_validation")
        if isinstance(physics, dict):
            status = "PASS" if physics.get("valid") else f"FAIL ({len(physics.get('issues', []))} issues)"
            color = (0, 255, 0) if physics.get("valid") else (0, 0, 255)
            cv2.putText(frame, f"Physics: {status}", (10, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Stabilization indicator
        if analysis.get("stabilized"):
            cv2.putText(frame, "STABILIZED", (width - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    print(f"✓ Demo video saved to {output_path}")


def create_summary_figure(analysis_path, output_path):
    """
    Create 4-panel summary figure:
    1. 2D trajectory
    2. 3D bird's-eye view (or disabled notice)
    3. Speed profile (or disabled notice)
    4. Statistics

    Handles the 2D-first sprint JSON schema where trajectory_3d and
    physics_validation are the sentinel string "disabled" rather than
    data structures.  Also handles legacy JSONs where those keys are
    absent, None, or empty lists.
    """
    with open(analysis_path, "r") as f:
        analysis = json.load(f)

    # Normalise 3D trajectory — covers: absent key, "disabled", None, []
    _raw_3d = analysis.get("trajectory_3d")
    traj_3d = _raw_3d if isinstance(_raw_3d, list) and len(_raw_3d) > 0 else []
    traj_3d_enabled = len(traj_3d) > 0

    # Normalise physics — covers: "disabled", None, absent key
    physics = analysis.get("physics_validation")
    if not isinstance(physics, dict):
        physics = {}
    physics_has_metrics = bool(physics.get("metrics") and physics["metrics"].get("speeds_kmh"))

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: 2D trajectory
    ax1 = plt.subplot(2, 2, 1)
    x_2d = [p["x"] for p in analysis.get("trajectory_2d") or []]
    y_2d = [p["y"] for p in analysis.get("trajectory_2d") or []]
    ax1.plot(x_2d, y_2d, "b-", linewidth=2, alpha=0.7)
    if x_2d:
        ax1.scatter(x_2d[0], y_2d[0], c="green", s=150, label="Start", zorder=5)
        ax1.scatter(x_2d[-1], y_2d[-1], c="red", s=150, label="Finish", zorder=5)
    ax1.invert_yaxis()
    ax1.set_title("2D Trajectory (Pixels)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: 3D bird's-eye view — disabled notice when 3D is not available
    ax2 = plt.subplot(2, 2, 2)
    if traj_3d_enabled:
        x_3d = [p["x"] for p in traj_3d]
        y_3d = [p["y"] for p in traj_3d]
        ax2.plot(x_3d, y_3d, "r-", linewidth=2, alpha=0.7)
        if x_3d:
            ax2.scatter(x_3d[0], y_3d[0], c="green", s=150, label="Start", zorder=5)
            ax2.scatter(x_3d[-1], y_3d[-1], c="red", s=150, label="Finish", zorder=5)
        ax2.set_xlabel("X (meters)")
        ax2.set_ylabel("Y (meters)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")
        ax2.set_title("3D Trajectory (Meters, Bird's Eye)", fontsize=14, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "3D disabled\n(2D-first mode)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=13, color="gray")
        ax2.set_title("3D Trajectory — Disabled", fontsize=14, fontweight="bold")
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Panel 3: Speed profile — requires 3D trajectory
    ax3 = plt.subplot(2, 2, 3)
    if traj_3d_enabled:
        speed_vals = []
        fps = (analysis.get("video_info") or {}).get("fps", 30)
        dt = 1.0 / fps
        for i in range(1, len(traj_3d)):
            dx = traj_3d[i]["x"] - traj_3d[i - 1]["x"]
            dy = traj_3d[i]["y"] - traj_3d[i - 1]["y"]
            dist = (dx**2 + dy**2) ** 0.5
            speed_vals.append(dist / dt * 3.6)
        ax3.plot(speed_vals, "g-", linewidth=2)
        ax3.set_title("Speed Profile", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Speed (km/h)")
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Speed profile unavailable\n(requires 3D trajectory)",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=13, color="gray")
        ax3.set_title("Speed Profile — Unavailable", fontsize=14, fontweight="bold")
        ax3.set_xticks([])
        ax3.set_yticks([])

    # Panel 4: Statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")

    gates_count = len(analysis.get("gates") or [])
    traj_count = len(analysis.get("trajectory_2d") or [])
    video_name = Path(analysis.get("video", "unknown")).name

    if physics_has_metrics and traj_3d_enabled:
        m = physics["metrics"]
        status = "PASS" if physics.get("valid") else "FAIL"
        stats_text = (
            f"ANALYSIS SUMMARY\n"
            f"{'=' * 30}\n\n"
            f"Gates Detected:     {gates_count}\n"
            f"Trajectory Points:  {traj_count}\n"
            f"Total Distance:     {m['total_distance_m']:.1f} m\n"
            f"Duration:           {m['duration_s']:.1f} s\n\n"
            f"Avg Speed:          {m['speeds_kmh']['mean']:.1f} km/h\n"
            f"Max Speed:          {m['speeds_kmh']['max']:.1f} km/h\n"
            f"Max G-force:        {m['g_forces']['max']:.2f} G\n"
            f"Min Turn Radius:    {m['turn_radii_m']['min']:.1f} m\n\n"
            f"Physics Check:      {status}\n"
        )
        if analysis.get("stabilized"):
            stats_text += "Mode:               STABILIZED\n"
        stats_text += f"Video: {video_name}"
    else:
        stats_text = (
            f"ANALYSIS SUMMARY\n"
            f"{'=' * 30}\n\n"
            f"Gates Detected:     {gates_count}\n"
            f"Trajectory Points:  {traj_count}\n"
            f"3D / Physics:       disabled (2D-first sprint)\n"
        )
        if analysis.get("stabilized"):
            stats_text += "Mode:               STABILIZED\n"
        stats_text += f"Video: {video_name}"

    ax4.text(0.1, 0.5, stats_text, fontsize=11, family="monospace",
             verticalalignment="center", transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Summary saved to {output_path}")


def plot_trajectory_comparison(analysis_paths, output_path):
    """
    Compare trajectories from multiple runs (bird's-eye view overlay).

    Args:
        analysis_paths: List of analysis JSON file paths.
        output_path: Path for output image.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, len(analysis_paths)))

    for i, path in enumerate(analysis_paths):
        with open(path, "r") as f:
            analysis = json.load(f)

        _raw = analysis.get("trajectory_3d")
        t3d = _raw if isinstance(_raw, list) and len(_raw) > 0 else []
        x = [p["x"] for p in t3d]
        y = [p["y"] for p in t3d]
        label = Path(analysis.get("video", "unknown")).stem

        ax.plot(x, y, color=colors[i], linewidth=2, alpha=0.8, label=label)

    ax.set_title("Trajectory Comparison (Bird's Eye View)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Comparison plot saved to {output_path}")
