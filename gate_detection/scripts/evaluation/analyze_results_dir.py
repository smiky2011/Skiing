#!/usr/bin/env python3
"""
Analyze a process_video results directory and write a Markdown report.

Expected inputs in --results-dir:
  - *_analysis.json files (per-video outputs from SkiRacingPipeline)
  - optional run_summary.json (written by scripts/process_video.py)

Output:
  - <results-dir>/analysis_report.md
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _fmt(value: Any, ndigits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def _trajectory_jump_stats(traj: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not traj or len(traj) < 2:
        return {
            "max_jump": None,
            "jumps_gt_10m": 0,
            "jumps_gt_50m": 0,
            "top_jumps": [],
        }

    max_jump = 0.0
    jumps_10 = 0
    jumps_50 = 0
    top: List[Tuple[float, int, int]] = []
    for i in range(1, len(traj)):
        a = traj[i - 1]
        b = traj[i]
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        ax = _safe_float(a.get("x"))
        ay = _safe_float(a.get("y"))
        bx = _safe_float(b.get("x"))
        by = _safe_float(b.get("y"))
        if ax is None or ay is None or bx is None or by is None:
            continue
        d = float(math.hypot(bx - ax, by - ay))
        f0 = _safe_int(a.get("frame")) or (i - 1)
        f1 = _safe_int(b.get("frame")) or i
        top.append((d, int(f0), int(f1)))
        if d > max_jump:
            max_jump = d
        if d > 10.0:
            jumps_10 += 1
        if d > 50.0:
            jumps_50 += 1

    top.sort(key=lambda t: t[0], reverse=True)
    return {
        "max_jump": float(max_jump) if top else None,
        "jumps_gt_10m": int(jumps_10),
        "jumps_gt_50m": int(jumps_50),
        "top_jumps": top[:5],
    }


def _auto_calib_summary(ac: Any) -> str:
    if not isinstance(ac, dict):
        return ""
    applied = bool(ac.get("applied"))
    unable = bool(ac.get("unable_to_calibrate"))
    corr = _safe_float(ac.get("correction"))
    reason = ac.get("reason")
    if unable:
        return f"unable ({_fmt(corr, 2)}x, {reason})" if corr is not None else f"unable ({reason})"
    if applied:
        return f"applied ({_fmt(corr, 2)}x)" if corr is not None else "applied"
    return f"not_applied ({_fmt(corr, 2)}x)" if corr is not None else "not_applied"


def _short_commit(commit: Any) -> str:
    if not commit:
        return ""
    s = str(commit).strip()
    return s[:8]


def _truncate(text: Any, max_len: int = 120) -> str:
    if not text:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a test_videos results directory")
    parser.add_argument("--results-dir", required=True, help="Directory containing *_analysis.json outputs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    run_summary_path = results_dir / "run_summary.json"
    run_entries: Optional[List[Dict[str, Any]]] = None
    if run_summary_path.exists():
        try:
            data = _load_json(run_summary_path)
            if isinstance(data, list):
                run_entries = [e for e in data if isinstance(e, dict)]
        except Exception:
            run_entries = None

    analysis_paths = sorted(results_dir.glob("*_analysis.json"))
    analyses: Dict[str, Dict[str, Any]] = {}
    for p in analysis_paths:
        try:
            obj = _load_json(p)
        except Exception:
            continue
        video_name = Path(obj.get("video", "")).name
        if not video_name:
            video_name = p.name.replace("_analysis.json", "")
        analyses[video_name] = obj

    rows: List[Dict[str, Any]] = []

    if run_entries is not None:
        # Use run_summary.json ordering as primary.
        seen = set()
        for entry in run_entries:
            video = str(entry.get("video") or "")
            seen.add(video)
            status = str(entry.get("status") or "")
            elapsed_s = _safe_float(entry.get("elapsed_s"))
            err = entry.get("error")
            obj = analyses.get(video)
            rows.append({
                "video": video,
                "status": status,
                "elapsed_s": elapsed_s,
                "run_error": err,
                "analysis": obj,
            })
        # Also include any *_analysis.json not referenced in run_summary.json.
        for video, obj in sorted(analyses.items(), key=lambda t: t[0]):
            if video in seen:
                continue
            rows.append({
                "video": video,
                "status": "ok",
                "elapsed_s": None,
                "run_error": None,
                "analysis": obj,
            })
    else:
        # No run summary; analyze whatever is present.
        for video, obj in sorted(analyses.items(), key=lambda t: t[0]):
            rows.append({
                "video": video,
                "status": "ok",
                "elapsed_s": None,
                "run_error": None,
                "analysis": obj,
            })

    # Build report rows
    report_rows: List[Dict[str, Any]] = []
    for row in rows:
        video = row["video"]
        status = row["status"]
        analysis_obj = row.get("analysis")
        obj = analysis_obj if isinstance(analysis_obj, dict) else {}
        info = obj.get("video_info") or {}
        td = obj.get("tracking_diagnostics") or {}
        pv = obj.get("physics_validation") or {}
        speeds = (pv.get("metrics") or {}).get("speeds_kmh") or {}
        gforces = (pv.get("metrics") or {}).get("g_forces") or {}
        traj = obj.get("trajectory_3d_raw") or obj.get("trajectory_3d") or []
        jumps = _trajectory_jump_stats(traj if isinstance(traj, list) else None)

        gates = obj.get("gates") or []
        gates_n = len(gates) if isinstance(gates, list) else None

        failure_reason = td.get("failure_reason") or row.get("run_error") or ""
        failure_reason = _truncate(failure_reason, 140)

        flags = []
        if analysis_obj is None:
            if status != "ok":
                flags.append("no_analysis_json")
        elif not obj.get("git_commit"):
            flags.append("missing_git_metadata")
        if (
            str(td.get("selected_method") or "") == "temporal"
            and "ModuleNotFoundError" in str(td.get("failure_reason") or "")
            and "lap" in str(td.get("failure_reason") or "")
        ):
            flags.append("old_temporal_fallback(lap_missing)")

        report_rows.append({
            "video": video,
            "status": status,
            "timestamp": obj.get("timestamp"),
            "commit": _short_commit(obj.get("git_commit")),
            "dirty": obj.get("git_dirty"),
            "fps": info.get("fps"),
            "frames": info.get("total_frames"),
            "discipline": obj.get("discipline"),
            "gates": gates_n,
            "track_method": td.get("selected_method"),
            "cov": td.get("bytetrack_coverage"),
            "physics": pv.get("valid"),
            "p90": speeds.get("p90"),
            "vmax": speeds.get("max"),
            "gmax": gforces.get("max"),
            "max_jump": jumps.get("max_jump"),
            "j10": jumps.get("jumps_gt_10m"),
            "j50": jumps.get("jumps_gt_50m"),
            "auto_calib": _auto_calib_summary(obj.get("auto_calibration")),
            "failure": failure_reason,
            "flags": ", ".join(flags),
            "top_jumps": jumps.get("top_jumps") or [],
        })

    # Summary counts
    total = len(report_rows)
    ok = sum(1 for r in report_rows if r.get("status") == "ok")
    err = sum(1 for r in report_rows if r.get("status") != "ok")
    physics_pass = sum(1 for r in report_rows if r.get("physics") is True)
    gates_lt_2 = sum(1 for r in report_rows if isinstance(r.get("gates"), int) and r["gates"] < 2)
    missing_analysis = sum(1 for r in report_rows if "no_analysis_json" in str(r.get("flags") or ""))
    mixed_version = sum(
        1
        for r in report_rows
        if any(
            f in str(r.get("flags") or "")
            for f in ("missing_git_metadata", "old_temporal_fallback(lap_missing)")
        )
    )

    lines: List[str] = []
    lines.append(f"# Results analysis: `{results_dir}`")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Videos: {total} (ok={ok}, error={err})")
    lines.append(f"- Physics: pass={physics_pass}, fail={total - physics_pass}")
    lines.append(f"- Gates < 2: {gates_lt_2}")
    lines.append(f"- Missing analysis JSONs: {missing_analysis}")
    lines.append(f"- Mixed-version flags: {mixed_version}")
    lines.append("")

    # Table
    headers = [
        "video",
        "status",
        "timestamp",
        "commit",
        "gates",
        "track",
        "cov",
        "physics",
        "p90_kmh",
        "vmax_kmh",
        "max_jump_m",
        "j>10m",
        "j>50m",
        "auto_calib",
        "failure",
        "flags",
    ]
    lines.append("## Table")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in report_rows:
        lines.append("| " + " | ".join([
            _truncate(r.get("video"), 60),
            _truncate(r.get("status"), 10),
            _truncate(r.get("timestamp"), 19),
            _truncate(r.get("commit"), 10),
            _fmt(r.get("gates")),
            _truncate(r.get("track_method"), 18),
            _fmt(_safe_float(r.get("cov")), 3),
            _fmt(r.get("physics")),
            _fmt(_safe_float(r.get("p90")), 2),
            _fmt(_safe_float(r.get("vmax")), 2),
            _fmt(_safe_float(r.get("max_jump")), 2),
            _fmt(r.get("j10")),
            _fmt(r.get("j50")),
            _truncate(r.get("auto_calib"), 24),
            _truncate(r.get("failure"), 60),
            _truncate(r.get("flags"), 32),
        ]) + " |")

    # Worst discontinuities
    lines.append("")
    lines.append("## Worst discontinuities (top 5 jumps per video)")
    for r in report_rows:
        top_jumps = r.get("top_jumps") or []
        if not top_jumps:
            continue
        lines.append("")
        lines.append(f"### {r.get('video')}")
        for d, f0, f1 in top_jumps:
            lines.append(f"- {d:.2f}m jump: frame {f0} → {f1}")

    out_path = results_dir / "analysis_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
