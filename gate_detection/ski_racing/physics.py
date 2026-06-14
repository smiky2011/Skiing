"""
Physics Validation Engine.

This is the project's secret weapon. Instead of just reporting ML metrics
(mAP, F1), we validate that the AI's output is physically plausible.

Key insight: A 95% accurate gate detector can still produce trajectories
that imply 15G turns - which is physically impossible. The physics engine
catches these errors and flags them for correction.

This module validates:
- Speed: Are estimated speeds within realistic racing ranges?
- G-forces: Are lateral forces humanly survivable?
- Turn radii: Are turns physically possible on skis?
- Trajectory smoothness: Is the path continuous (no teleportation)?
"""
import numpy as np
from typing import List, Dict, Optional, Tuple


class PhysicsValidator:
    """
    Validate that extracted trajectories are physically plausible
    for alpine ski racing.

    Realistic ranges (from published biomechanics data):
    - Slalom speed: 30-60 km/h
    - GS speed: 50-90 km/h
    - Downhill speed: 80-150 km/h
    - Peak G-forces: 2-4G (typical), up to 5G briefly
    - Min turn radius: ~8m (slalom), ~25m (GS)
    """

    # Physical limits for different disciplines
    LIMITS = {
        "slalom": {
            "max_speed_kmh": 70,
            "min_speed_kmh": 15,
            "max_g_force": 5.0,
            "min_turn_radius_m": 6.0,
            "max_acceleration_ms2": 15.0,  # Including gravity component
        },
        "giant_slalom": {
            "max_speed_kmh": 100,
            "min_speed_kmh": 30,
            "max_g_force": 4.5,
            "min_turn_radius_m": 20.0,
            "max_acceleration_ms2": 12.0,
        },
        "downhill": {
            "max_speed_kmh": 160,
            "min_speed_kmh": 60,
            "max_g_force": 4.0,
            "min_turn_radius_m": 50.0,
            "max_acceleration_ms2": 10.0,
        },
    }

    def __init__(self, discipline="slalom", fps=30.0):
        """
        Args:
            discipline: "slalom", "giant_slalom", or "downhill".
            fps: Video frame rate (needed for speed calculations).
        """
        self.discipline = discipline
        self.fps = fps
        self.limits = self.LIMITS[discipline]
        self.dt = 1.0 / fps  # Time between frames

    def validate_trajectory(self, trajectory_3d):
        """
        Run all physics checks on a 3D trajectory.

        Args:
            trajectory_3d: List of {"frame": int, "x": float, "y": float} in meters.

        Returns:
            Dictionary with validation results and issues found.
        """
        if len(trajectory_3d) < 3:
            return {"valid": False, "issues": ["Trajectory too short for validation"]}

        speeds, dts = self.calculate_speeds(trajectory_3d)
        accelerations = self.calculate_accelerations(speeds, dts)
        turn_radii = self.calculate_turn_radii(trajectory_3d)
        g_forces = self.calculate_g_forces(speeds, turn_radii)
        smoothness = self.calculate_smoothness(trajectory_3d)

        issues = []

        # Check speeds
        if len(speeds) > 0:
            max_speed = max(speeds)
            min_speed = min(speeds) if min(speeds) > 0 else 0
            avg_speed = np.mean(speeds)

            if max_speed > self.limits["max_speed_kmh"]:
                issues.append(
                    f"Unrealistic max speed: {max_speed:.1f} km/h "
                    f"(limit: {self.limits['max_speed_kmh']} km/h for {self.discipline})"
                )
            if avg_speed < self.limits["min_speed_kmh"]:
                issues.append(
                    f"Unrealistically slow avg speed: {avg_speed:.1f} km/h "
                    f"(expected >{self.limits['min_speed_kmh']} km/h)"
                )

        # Check G-forces
        if len(g_forces) > 0:
            max_g = max(g_forces)
            if max_g > self.limits["max_g_force"]:
                issues.append(
                    f"Unrealistic G-force: {max_g:.1f}G "
                    f"(limit: {self.limits['max_g_force']}G) — "
                    f"check homography or gate spacing"
                )

        # Check turn radii
        valid_radii = [r for r in turn_radii if r > 0 and r < 500]
        if len(valid_radii) > 0:
            min_radius = min(valid_radii)
            if min_radius < self.limits["min_turn_radius_m"]:
                issues.append(
                    f"Unrealistic min turn radius: {min_radius:.1f}m "
                    f"(limit: {self.limits['min_turn_radius_m']}m)"
                )

        # Check trajectory smoothness
        if smoothness["max_jump_m"] > 5.0:
            issues.append(
                f"Trajectory discontinuity: {smoothness['max_jump_m']:.1f}m jump "
                f"between consecutive frames"
            )

        # Check accelerations
        if len(accelerations) > 0:
            max_accel = max(abs(a) for a in accelerations)
            if max_accel > self.limits["max_acceleration_ms2"]:
                issues.append(
                    f"Unrealistic acceleration: {max_accel:.1f} m/s^2 "
                    f"(limit: {self.limits['max_acceleration_ms2']})"
                )

        speeds_arr = np.asarray(speeds, dtype=float) if speeds else np.asarray([], dtype=float)
        speeds_nz = speeds_arr[speeds_arr > 1e-3]
        if len(speeds_nz) < max(10, int(0.5 * len(speeds_arr))):
            speeds_nz = speeds_arr

        def _pct(arr, q):
            return float(np.percentile(arr, q)) if len(arr) else 0.0

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "metrics": {
                "speeds_kmh": {
                    "min": float(min(speeds)) if speeds else 0,
                    "max": float(max(speeds)) if speeds else 0,
                    "mean": float(np.mean(speeds)) if speeds else 0,
                    "std": float(np.std(speeds)) if speeds else 0,
                    "median": _pct(speeds_nz, 50),
                    "p90": _pct(speeds_nz, 90),
                    "p95": _pct(speeds_nz, 95),
                    "n": int(len(speeds_arr)),
                    "n_nonzero": int(len(speeds_nz)),
                },
                "g_forces": {
                    "max": float(max(g_forces)) if g_forces else 0,
                    "mean": float(np.mean(g_forces)) if g_forces else 0,
                },
                "turn_radii_m": {
                    "min": float(min(valid_radii)) if valid_radii else 0,
                    "mean": float(np.mean(valid_radii)) if valid_radii else 0,
                },
                "smoothness": smoothness,
                "total_distance_m": self._total_distance(trajectory_3d),
                "duration_s": self._duration_seconds(trajectory_3d),
            },
            "discipline": self.discipline,
            "fps": self.fps,
        }

    def calculate_speeds(self, trajectory_3d):
        """
        Calculate instantaneous speeds in km/h from 3D trajectory.
        """
        speeds = []
        dts = []
        for i in range(1, len(trajectory_3d)):
            p1 = trajectory_3d[i - 1]
            p2 = trajectory_3d[i]
            dx = p2["x"] - p1["x"]
            dy = p2["y"] - p1["y"]
            distance = (dx**2 + dy**2) ** 0.5
            f1 = p1.get("frame", i - 1)
            f2 = p2.get("frame", i)
            frame_delta = max(1, int(f2 - f1))
            dt = frame_delta / self.fps
            speed_ms = distance / dt
            speed_kmh = speed_ms * 3.6
            speeds.append(speed_kmh)
            dts.append(dt)
        return speeds, dts

    def calculate_accelerations(self, speeds_kmh, dts=None):
        """
        Calculate accelerations in m/s^2 from speed profile.
        """
        accelerations = []
        for i in range(1, len(speeds_kmh)):
            dv = (speeds_kmh[i] - speeds_kmh[i - 1]) / 3.6  # Convert to m/s
            if dts is None:
                dt = self.dt
            else:
                # Use average dt between adjacent speed samples
                dt = (dts[i - 1] + dts[i]) / 2 if i < len(dts) else dts[i - 1]
            if dt <= 0:
                continue
            accel = dv / dt
            accelerations.append(accel)
        return accelerations

    def calculate_turn_radii(self, trajectory_3d):
        """
        Calculate turn radius at each point using three consecutive points.
        Uses the circumscribed circle formula.
        """
        radii = []
        for i in range(1, len(trajectory_3d) - 1):
            p1 = (trajectory_3d[i - 1]["x"], trajectory_3d[i - 1]["y"])
            p2 = (trajectory_3d[i]["x"], trajectory_3d[i]["y"])
            p3 = (trajectory_3d[i + 1]["x"], trajectory_3d[i + 1]["y"])

            radius = self._circumradius(p1, p2, p3)
            radii.append(radius)

        return radii

    def calculate_g_forces(self, speeds_kmh, turn_radii):
        """
        Calculate lateral G-forces from speed and turn radius.
        G = v^2 / (r * g) where g = 9.81 m/s^2
        """
        g_forces = []
        n = min(len(speeds_kmh), len(turn_radii))

        for i in range(n):
            speed_ms = speeds_kmh[i] / 3.6
            radius = turn_radii[i]

            if radius > 0.1:  # Avoid division by near-zero
                g_force = (speed_ms**2) / (radius * 9.81)
                g_forces.append(g_force)
            else:
                g_forces.append(0.0)

        return g_forces

    def calculate_smoothness(self, trajectory_3d):
        """
        Assess trajectory smoothness.
        Large jumps between frames indicate tracking errors.
        """
        jumps = []
        for i in range(1, len(trajectory_3d)):
            p1 = trajectory_3d[i - 1]
            p2 = trajectory_3d[i]
            dx = p2["x"] - p1["x"]
            dy = p2["y"] - p1["y"]
            jump = (dx**2 + dy**2) ** 0.5
            jumps.append(jump)

        return {
            "max_jump_m": float(max(jumps)) if jumps else 0,
            "mean_jump_m": float(np.mean(jumps)) if jumps else 0,
            "std_jump_m": float(np.std(jumps)) if jumps else 0,
        }

    def _circumradius(self, p1, p2, p3):
        """
        Calculate radius of circumscribed circle through three points.
        Returns float('inf') for collinear points.
        """
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        # Side lengths
        a = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        b = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
        c = ((ax - cx) ** 2 + (ay - cy) ** 2) ** 0.5

        # Area via cross product: area = 0.5 * |AB x AC|
        area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))

        if area < 1e-10:
            return float("inf")  # Points are collinear

        # Circumradius: R = abc / (4 * area)
        radius = (a * b * c) / (4 * area)
        return radius

    def _total_distance(self, trajectory_3d):
        """Calculate total distance covered in meters."""
        total = 0
        for i in range(1, len(trajectory_3d)):
            p1 = trajectory_3d[i - 1]
            p2 = trajectory_3d[i]
            dx = p2["x"] - p1["x"]
            dy = p2["y"] - p1["y"]
            total += (dx**2 + dy**2) ** 0.5
        return float(total)

    def _duration_seconds(self, trajectory_3d):
        """Estimate duration from frame indices if available."""
        if not trajectory_3d:
            return 0.0
        f0 = trajectory_3d[0].get("frame")
        f1 = trajectory_3d[-1].get("frame")
        if f0 is not None and f1 is not None and f1 >= f0:
            return (f1 - f0) / self.fps
        return len(trajectory_3d) / self.fps

    def print_report(self, validation_result):
        """Print a human-readable validation report."""
        r = validation_result
        m = r["metrics"]

        print("\n" + "=" * 60)
        print("  PHYSICS VALIDATION REPORT")
        print("=" * 60)
        print(f"  Discipline:    {r['discipline']}")
        print(f"  Video FPS:     {r['fps']}")
        print(f"  Duration:      {m['duration_s']:.1f}s")
        print(f"  Distance:      {m['total_distance_m']:.1f}m")
        print("-" * 60)

        print(f"\n  Speed (km/h):  {m['speeds_kmh']['mean']:.1f} avg, "
              f"{m['speeds_kmh']['median']:.1f} med, "
              f"{m['speeds_kmh']['p90']:.1f} p90, "
              f"{m['speeds_kmh']['max']:.1f} max, "
              f"{m['speeds_kmh']['min']:.1f} min")

        print(f"  G-forces:      {m['g_forces']['mean']:.2f} avg, "
              f"{m['g_forces']['max']:.2f} max")

        print(f"  Turn radius:   {m['turn_radii_m']['min']:.1f}m min, "
              f"{m['turn_radii_m']['mean']:.1f}m avg")

        print(f"  Smoothness:    {m['smoothness']['max_jump_m']:.2f}m max jump, "
              f"{m['smoothness']['mean_jump_m']:.3f}m avg")

        print("-" * 60)
        if r["valid"]:
            print("  ✅ PASSED — Trajectory is physically plausible")
        else:
            print(f"  ❌ FAILED — {len(r['issues'])} issue(s) found:")
            for issue in r["issues"]:
                print(f"     • {issue}")
        print("=" * 60 + "\n")
