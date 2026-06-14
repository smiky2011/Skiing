#!/usr/bin/env python3
"""
Single-command evaluation pipeline for ski racing:
  Stage 1: Holdout gate detection metrics
  Stage 2: 3-video regression suite
  Stage 3: Markdown summary + PASS/FAIL verdict
"""
import argparse
import inspect
import json
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import parse_thresholds, run_holdout_evaluation  # noqa: E402
from ski_racing.pipeline import SkiRacingPipeline  # noqa: E402


REGRESSION_VIDEO_IDS = ("2907", "2909", "2911")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# direction:
#   higher -> lower than baseline is worse
#   lower  -> higher than baseline is worse
METRIC_SPECS = {
    "gates_detected": {"label": "Gates detected", "direction": "higher"},
    "trajectory_coverage": {"label": "Trajectory coverage", "direction": "higher"},
    "confirmed_gate_count": {"label": "Confirmed gate count", "direction": "higher"},
    "ghost_gate_count_raw": {"label": "Ghost gate count (raw)", "direction": "lower"},
    "gate_interp_rate": {"label": "Gate interpolation rate", "direction": "lower"},
    "track_id_switches": {"label": "Track ID switches", "direction": "lower"},
    "p90_speed_kmh": {"label": "P90 speed (km/h)", "direction": "lower"},
    "max_speed_kmh": {"label": "Max speed (km/h)", "direction": "lower"},
    "max_g_force": {"label": "Max G-force", "direction": "lower"},
    "max_jump_m": {"label": "Max jump (m)", "direction": "lower"},
    "auto_cal_correction": {"label": "Auto-cal correction", "direction": "lower"},
    "physics_issue_count": {"label": "Physics issue count", "direction": "lower"},
    "course_gates_count": {"label": "Course gates count", "direction": "higher"},
}


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_scalar(raw_value):
    value = raw_value.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path):
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        data[key.strip()] = parse_scalar(raw_value)
    return data


def load_regression_config(config_path):
    config_path = resolve_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Regression config not found: {config_path}")

    config = load_simple_yaml(config_path)
    defaults = {
        "gate_conf": 0.35,
        "gate_iou": 0.55,
        "stabilize": True,  # deprecated compatibility key (remove after 2026-04-30)
        # camera_mode and projection removed — 2D-first pipeline no longer uses them.
        # Old YAML files that still contain these keys will have them loaded into the
        # config dict (no parse error) but they will not be forwarded to the pipeline.
        "discipline": "slalom",
        "gate_init_mode": "single_best",
        "gate_consensus_min_support": 3,
        "skier_conf": 0.25,
        "gate_search_frames": 150,
        "gate_search_stride": 5,
        "gate_track_frames": 120,
        "gate_track_stride": 3,
        "gate_track_min_obs": 3,
        "frame_stride": 1,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    return config_path, config


def get_git_commit():
    try:
        proc = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def discover_regression_videos(regression_dir):
    regression_dir = resolve_path(regression_dir)
    if not regression_dir.exists():
        raise FileNotFoundError(f"Regression video directory not found: {regression_dir}")

    candidates = sorted(
        path
        for path in regression_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    videos = {}
    for video_id in REGRESSION_VIDEO_IDS:
        matches = [path for path in candidates if video_id in path.stem]
        if not matches:
            raise FileNotFoundError(
                f"Missing regression video containing '{video_id}' in {regression_dir}"
            )
        videos[video_id] = matches[0]

    return regression_dir, videos


def safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def extract_stage2_metrics(video_id, video_path, analysis_path, results):
    # Guard against sentinel string "disabled" (2D-first sprint) as well as
    # None / absent key.  `or {}` would wrongly keep the truthy string "disabled".
    physics = results.get("physics_validation")
    if not isinstance(physics, dict):
        # "disabled" sentinel or missing — intentional during 2D-only sprint;
        # speed/G-force/jump metrics are expected to be zero.
        physics = {}
    metrics = physics.get("metrics") or {}
    speeds = metrics.get("speeds_kmh") or {}
    g_forces = metrics.get("g_forces") or {}
    smoothness = metrics.get("smoothness") or {}
    auto_calibration = results.get("auto_calibration") or {}
    gate_tracking_quality = results.get("gate_tracking_quality") or {}

    total_frames = int((results.get("video_info") or {}).get("total_frames") or 0)
    tracked_frames = len(results.get("trajectory_2d") or [])

    coverage = safe_float(tracked_frames / total_frames, 0.0) if total_frames else 0.0
    correction = auto_calibration.get("correction")
    if correction is None:
        correction = 1.0

    return {
        "video_id": video_id,
        "video_file": video_path.name,
        "analysis_json": str(analysis_path),
        "gates_detected": int(len(results.get("gates") or [])),
        "confirmed_gate_count": int(gate_tracking_quality.get("confirmed_gate_count", 0)),
        "ghost_gate_count_raw": int(gate_tracking_quality.get("ghost_gate_count_raw", 0)),
        "gate_interp_rate": safe_float(gate_tracking_quality.get("interp_rate_overall"), 0.0),
        "provisional_dropped_count": int(gate_tracking_quality.get("provisional_dropped_count", 0)),
        "trajectory_coverage": float(coverage),
        "tracked_frames": tracked_frames,
        "total_frames": total_frames,
        "track_id_switches": int(results.get("track_id_switches") or 0),
        "p90_speed_kmh": safe_float(speeds.get("p90"), 0.0),
        "max_speed_kmh": safe_float(speeds.get("max"), 0.0),
        "max_g_force": safe_float(g_forces.get("max"), 0.0),
        "max_jump_m": safe_float(smoothness.get("max_jump_m"), 0.0),
        "auto_cal_correction": safe_float(correction, 1.0),
        "auto_cal_applied": bool(auto_calibration.get("applied", False)),
        "physics_issue_count": int(len(physics.get("issues") or [])),
        "course_gates_count": int(results.get("course_gates_count") or 0),
    }


def aggregate_stage2(per_video):
    return {
        "videos": int(len(per_video)),
        "gates_detected": safe_mean([row["gates_detected"] for row in per_video]),
        "confirmed_gate_count": safe_mean([row["confirmed_gate_count"] for row in per_video]),
        "ghost_gate_count_raw": safe_mean([row["ghost_gate_count_raw"] for row in per_video]),
        "gate_interp_rate": safe_mean([row["gate_interp_rate"] for row in per_video]),
        "provisional_dropped_count": safe_mean([row["provisional_dropped_count"] for row in per_video]),
        "trajectory_coverage": safe_mean([row["trajectory_coverage"] for row in per_video]),
        "track_id_switches": safe_mean([row["track_id_switches"] for row in per_video]),
        "p90_speed_kmh": safe_mean([row["p90_speed_kmh"] for row in per_video]),
        "max_speed_kmh": safe_mean([row["max_speed_kmh"] for row in per_video]),
        "max_g_force": safe_mean([row["max_g_force"] for row in per_video]),
        "max_jump_m": safe_mean([row["max_jump_m"] for row in per_video]),
        "auto_cal_correction": safe_mean([row["auto_cal_correction"] for row in per_video]),
        "physics_issue_count": safe_mean([row["physics_issue_count"] for row in per_video]),
        "course_gates_count": safe_mean([row["course_gates_count"] for row in per_video]),
    }


def nested_get(payload, path):
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def extract_baseline_summary(baseline_payload):
    baseline_f1 = None
    f1_paths = (
        ("stage1", "summary", "f1"),
        ("stage1", "primary_threshold", "metrics", "f1"),
        ("summary", "recommended", "f1"),
        ("metrics", "f1"),
    )
    for path in f1_paths:
        candidate = nested_get(baseline_payload, path)
        if candidate is not None:
            baseline_f1 = safe_float(candidate)
            break

    stage2_source = None
    for path in (("stage2", "aggregate"), ("aggregate", "new"), ("aggregate",)):
        candidate = nested_get(baseline_payload, path)
        if isinstance(candidate, dict):
            stage2_source = candidate
            break

    baseline_stage2 = {}
    if isinstance(stage2_source, dict):
        alias_map = {
            "gates_detected": ("gates_detected", "avg_gates_detected"),
            "confirmed_gate_count": ("confirmed_gate_count", "avg_confirmed_gate_count"),
            "ghost_gate_count_raw": ("ghost_gate_count_raw", "avg_ghost_gate_count_raw"),
            "gate_interp_rate": ("gate_interp_rate", "avg_gate_interp_rate"),
            "provisional_dropped_count": (
                "provisional_dropped_count",
                "avg_provisional_dropped_count",
            ),
            "trajectory_coverage": ("trajectory_coverage", "avg_traj_coverage"),
            "p90_speed_kmh": ("p90_speed_kmh", "avg_speed_p90_kmh"),
            "max_speed_kmh": ("max_speed_kmh", "avg_speed_max_kmh"),
            "max_g_force": ("max_g_force", "avg_gforce_max"),
            "max_jump_m": ("max_jump_m", "avg_max_jump_m"),
            "auto_cal_correction": ("auto_cal_correction", "avg_auto_calibration_correction"),
            "physics_issue_count": ("physics_issue_count", "avg_physics_issue_count"),
            "course_gates_count": ("course_gates_count",),
        }
        for metric_name, aliases in alias_map.items():
            for alias in aliases:
                if alias in stage2_source:
                    baseline_stage2[metric_name] = safe_float(stage2_source[alias])
                    break

    return {"f1": baseline_f1, "stage2": baseline_stage2}


def compare_against_baseline(stage1_result, stage2_result, baseline_payload):
    comparison = {
        "has_baseline": baseline_payload is not None,
        "status": "PASS",
        "reasons": [],
        "f1": None,
        "stage2": [],
    }

    if baseline_payload is None:
        comparison["status"] = "PASS"
        comparison["reasons"] = ["No baseline provided; skipped regression delta checks."]
        return comparison

    baseline = extract_baseline_summary(baseline_payload)
    current_f1 = safe_float((stage1_result.get("summary") or {}).get("f1"), 0.0)
    baseline_f1 = baseline.get("f1")

    if baseline_f1 is None:
        comparison["reasons"].append("Baseline F1 missing; unable to apply F1 gate.")
    else:
        delta = current_f1 - baseline_f1
        delta_pct = (delta / abs(baseline_f1) * 100.0) if baseline_f1 else None
        comparison["f1"] = {
            "current": current_f1,
            "baseline": baseline_f1,
            "delta": delta,
            "delta_pct": delta_pct,
        }
        if current_f1 < baseline_f1:
            comparison["reasons"].append(
                f"F1 decreased ({current_f1:.4f} < baseline {baseline_f1:.4f})."
            )

    current_aggregate = stage2_result.get("aggregate") or {}
    baseline_stage2 = baseline.get("stage2") or {}

    for metric_name, spec in METRIC_SPECS.items():
        current_value = safe_float(current_aggregate.get(metric_name), 0.0)
        baseline_value = baseline_stage2.get(metric_name)

        if baseline_value is None:
            continue

        baseline_value = safe_float(baseline_value)
        delta = current_value - baseline_value
        delta_pct = (delta / abs(baseline_value) * 100.0) if baseline_value else None

        if spec["direction"] == "higher":
            if baseline_value == 0:
                degradation_ratio = float("inf") if current_value < 0 else 0.0
            else:
                degradation_ratio = (baseline_value - current_value) / abs(baseline_value)
        else:
            if baseline_value == 0:
                degradation_ratio = float("inf") if current_value > 0 else 0.0
            else:
                degradation_ratio = (current_value - baseline_value) / abs(baseline_value)

        degraded = degradation_ratio > 0.20
        comparison["stage2"].append(
            {
                "metric": metric_name,
                "label": spec["label"],
                "direction": spec["direction"],
                "current": current_value,
                "baseline": baseline_value,
                "delta": delta,
                "delta_pct": delta_pct,
                "degradation_ratio": degradation_ratio,
                "degraded": degraded,
            }
        )

        if degraded:
            comparison["reasons"].append(
                f"{spec['label']} degraded by more than 20% "
                f"(baseline {baseline_value:.4f}, current {current_value:.4f})."
            )

    comparison["status"] = "FAIL" if comparison["reasons"] and baseline_payload is not None else "PASS"
    return comparison


def format_delta_percent(value):
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def render_summary(
    report_dir,
    model_path,
    git_commit,
    stage1_result,
    stage2_result,
    comparison,
    baseline_path,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = comparison.get("status", "FAIL")

    lines = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- Git commit: `{git_commit or 'unknown'}`")
    if baseline_path:
        lines.append(f"- Baseline: `{baseline_path}`")
    lines.append(f"- Verdict: **{verdict}**")
    lines.append("")

    lines.append("## Stage 1 - Holdout Detection Metrics")
    lines.append("")
    lines.append("| Confidence | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for threshold in stage1_result.get("thresholds", []):
        key = f"{float(threshold):.2f}"
        row = (stage1_result.get("per_threshold") or {}).get(key, {})
        lines.append(
            "| "
            f"{threshold:.2f} | {safe_float(row.get('precision')):.4f} | "
            f"{safe_float(row.get('recall')):.4f} | {safe_float(row.get('f1')):.4f} | "
            f"{int(row.get('tp', 0))} | {int(row.get('fp', 0))} | {int(row.get('fn', 0))} |"
        )
    lines.append("")

    lines.append("## Stage 2 - Regression Suite")
    lines.append("")
    lines.append(
        "| Video | Gates | Coverage | P90 speed (km/h) | Max speed (km/h) | "
        "Max G-force | Max jump (m) | Auto-cal correction | Physics issues |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in stage2_result.get("per_video", []):
        lines.append(
            f"| `{row['video_id']}` | {row['gates_detected']} | {row['trajectory_coverage']:.3f} | "
            f"{row['p90_speed_kmh']:.2f} | {row['max_speed_kmh']:.2f} | {row['max_g_force']:.2f} | "
            f"{row['max_jump_m']:.2f} | {row['auto_cal_correction']:.2f} | {row['physics_issue_count']} |"
        )

    agg = stage2_result.get("aggregate") or {}
    lines.append(
        f"| **Mean** | **{agg.get('gates_detected', 0):.3f}** | "
        f"**{agg.get('trajectory_coverage', 0):.3f}** | **{agg.get('p90_speed_kmh', 0):.2f}** | "
        f"**{agg.get('max_speed_kmh', 0):.2f}** | **{agg.get('max_g_force', 0):.2f}** | "
        f"**{agg.get('max_jump_m', 0):.2f}** | **{agg.get('auto_cal_correction', 0):.2f}** | "
        f"**{agg.get('physics_issue_count', 0):.3f}** |"
    )
    lines.append("")

    if comparison.get("has_baseline"):
        lines.append("## Delta vs Baseline")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Delta | Delta % |")
        lines.append("|---|---:|---:|---:|---:|")

        f1_row = comparison.get("f1")
        if f1_row:
            lines.append(
                f"| F1 | {f1_row['baseline']:.4f} | {f1_row['current']:.4f} | "
                f"{f1_row['delta']:+.4f} | {format_delta_percent(f1_row['delta_pct'])} |"
            )

        for row in comparison.get("stage2", []):
            lines.append(
                f"| {row['label']} | {row['baseline']:.4f} | {row['current']:.4f} | "
                f"{row['delta']:+.4f} | {format_delta_percent(row['delta_pct'])} |"
            )
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")

    reasons = comparison.get("reasons") or []
    if reasons:
        lines.append("")
        lines.append("Reasons:")
        for reason in reasons:
            lines.append(f"- {reason}")

    summary_path = report_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def make_output_dir(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = output_root / f"eval_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"eval_{stamp}_{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def parse_args():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline (holdout + regression + summary)")
    parser.add_argument("--model", required=True, help="Path to gate detector model (.pt)")
    parser.add_argument("--baseline", help="Path to previous eval_result.json for delta checks")

    parser.add_argument(
        "--data",
        default="data/datasets/final_combined_1class_20260226_curated/test",
        help="Holdout split path (or data.yaml)",
    )
    parser.add_argument(
        "--config",
        default="configs/regression_defaults.yaml",
        help="Frozen regression config",
    )
    parser.add_argument(
        "--regression-dir",
        default="tracks/E_evaluation_ci/regression_videos",
        help="Directory containing frozen regression videos",
    )
    parser.add_argument(
        "--output-root",
        default="docs/reports",
        help="Directory where eval_YYYYMMDD_HHMM reports are created",
    )

    parser.add_argument(
        "--thresholds",
        default="0.25,0.35,0.45,0.55",
        help="Comma-separated confidence thresholds for holdout stage",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.35,
        help="Primary threshold for Stage 1 summary",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="IoU threshold used for TP matching in Stage 1",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.55,
        help="NMS IoU threshold used for Stage 1 inference",
    )
    parser.add_argument(
        "--ensemble-models",
        default=None,
        help="Comma-separated additional model paths for ensemble evaluation",
    )
    parser.add_argument(
        "--ensemble-nms-iou",
        type=float,
        default=0.50,
        help="NMS IoU threshold for merging ensemble detections",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (should match training size)",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.80,
        help="Minimum F1 threshold for PASS verdict (default 0.80)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_path = resolve_path(args.data)
    output_root = resolve_path(args.output_root)
    report_dir = make_output_dir(output_root)

    config_path, regression_config = load_regression_config(args.config)
    regression_dir, regression_videos = discover_regression_videos(args.regression_dir)

    baseline_payload = None
    baseline_path = None
    if args.baseline:
        baseline_path = resolve_path(args.baseline)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    print(f"[INFO] Report directory: {report_dir}")

    # Parse ensemble models
    ensemble_model_paths = None
    if args.ensemble_models:
        ensemble_model_paths = [
            str(resolve_path(p.strip()))
            for p in args.ensemble_models.split(",")
            if p.strip()
        ]

    # Stage 1
    print("[INFO] Stage 1/3 - Holdout detection metrics")
    stage1_path = report_dir / "stage1_holdout.json"
    stage1_result = run_holdout_evaluation(
        model_path=str(model_path),
        data_path=str(data_path),
        output_path=stage1_path,
        thresholds=parse_thresholds(args.thresholds),
        match_iou=float(args.match_iou),
        nms_iou=float(args.nms_iou),
        default_threshold=float(args.default_threshold),
        ensemble_model_paths=ensemble_model_paths,
        ensemble_nms_iou=float(args.ensemble_nms_iou),
        imgsz=int(args.imgsz),
    )

    # F1 gate check
    stage1_f1 = safe_float((stage1_result.get("summary") or {}).get("f1"), 0.0)
    min_f1 = float(args.min_f1)
    if stage1_f1 < min_f1:
        print(f"[WARN] Stage 1 F1 = {stage1_f1:.4f} < {min_f1:.2f} minimum threshold")
    else:
        print(f"[INFO] Stage 1 F1 = {stage1_f1:.4f} >= {min_f1:.2f} threshold PASS")

    # Stage 2
    print("[INFO] Stage 2/3 - 3-video regression")
    stage2_file = report_dir / "stage2_regression.json"
    stage2_dir = report_dir / "stage2_regression"
    stage2_tmp_dir = report_dir / "_stage2_tmp"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage2_tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Signature preflight ──────────────────────────────────────────────────
    # Warn (not hard-fail) if this runner is about to pass config-derived keys
    # that no longer exist in the pipeline signatures.
    _init_params = set(inspect.signature(SkiRacingPipeline.__init__).parameters) - {"self"}
    _run_params = set(inspect.signature(SkiRacingPipeline.process_video).parameters) - {"self"}

    init_kwargs = {"gate_model_path": str(model_path)}
    _stab = regression_config.get("stabilize", False)  # legacy fallback
    config_init_kwargs = {
        "discipline": str(regression_config["discipline"]),
        "gate_init_mode": str(regression_config.get("gate_init_mode", "single_best")),
        "gate_consensus_min_support": int(regression_config.get("gate_consensus_min_support", 3)),
        "gate_full_track": bool(regression_config.get("gate_full_track", _stab)),
        "outlier_filter": bool(regression_config.get("outlier_filter", _stab)),
        "kalman_smooth": bool(regression_config.get("kalman_smooth", _stab)),
    }
    for key, value in config_init_kwargs.items():
        if key not in _init_params:
            warnings.warn(
                f"[run_eval] Config key '{key}' is not a valid pipeline parameter and will be ignored.",
                stacklevel=2,
            )
            continue
        init_kwargs[key] = value
    # ────────────────────────────────────────────────────────────────────────

    pipeline = SkiRacingPipeline(**init_kwargs)

    stage2_per_video = []
    for video_id in REGRESSION_VIDEO_IDS:
        video_path = regression_videos[video_id]
        print(f"[INFO]   Processing regression video {video_id}: {video_path.name}")
        run_kwargs = {
            "video_path": str(video_path),
            "output_dir": str(stage2_tmp_dir),
        }
        config_run_kwargs = {
            "gate_conf": safe_float(regression_config["gate_conf"]),
            "gate_iou": safe_float(regression_config["gate_iou"]),
            "skier_conf": safe_float(regression_config.get("skier_conf"), 0.25),
            "gate_search_frames": int(regression_config.get("gate_search_frames", 150)),
            "gate_search_stride": int(regression_config.get("gate_search_stride", 5)),
            "gate_track_frames": int(regression_config.get("gate_track_frames", 120)),
            "gate_track_stride": int(regression_config.get("gate_track_stride", 3)),
            "gate_track_min_obs": int(regression_config.get("gate_track_min_obs", 3)),
            "frame_stride": int(regression_config.get("frame_stride", 1)),
        }
        for key, value in config_run_kwargs.items():
            if key not in _run_params:
                warnings.warn(
                    f"[run_eval] Config key '{key}' is not a valid pipeline parameter and will be ignored.",
                    stacklevel=2,
                )
                continue
            run_kwargs[key] = value

        results = pipeline.process_video(**run_kwargs)

        canonical_analysis_path = stage2_dir / f"{video_id}_analysis.json"
        canonical_analysis_path.write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )

        metrics = extract_stage2_metrics(
            video_id=video_id,
            video_path=video_path,
            analysis_path=canonical_analysis_path.relative_to(report_dir),
            results=results,
        )
        stage2_per_video.append(metrics)

    shutil.rmtree(stage2_tmp_dir, ignore_errors=True)

    stage2_result = {
        "config": str(config_path),
        "regression_dir": str(regression_dir),
        "settings": regression_config,
        "videos": [
            {"video_id": video_id, "path": str(regression_videos[video_id])}
            for video_id in REGRESSION_VIDEO_IDS
        ],
        "per_video": stage2_per_video,
        "aggregate": aggregate_stage2(stage2_per_video),
    }
    stage2_file.write_text(json.dumps(stage2_result, indent=2), encoding="utf-8")

    # Stage 3
    print("[INFO] Stage 3/3 - Summary + verdict")
    comparison = compare_against_baseline(stage1_result, stage2_result, baseline_payload)

    # Apply F1 minimum gate regardless of baseline
    if stage1_f1 < min_f1:
        comparison["status"] = "FAIL"
        comparison["reasons"].append(
            f"F1 ({stage1_f1:.4f}) below minimum threshold ({min_f1:.2f})."
        )

    git_commit = get_git_commit()
    summary_path = render_summary(
        report_dir=report_dir,
        model_path=model_path,
        git_commit=git_commit,
        stage1_result=stage1_result,
        stage2_result=stage2_result,
        comparison=comparison,
        baseline_path=baseline_path,
    )

    eval_result_path = report_dir / "eval_result.json"
    eval_result_payload = {
        "timestamp": datetime.now().isoformat(),
        "model": str(model_path),
        "git_commit": git_commit,
        "output_dir": str(report_dir),
        "artifacts": {
            "stage1_holdout": str(stage1_path),
            "stage2_regression": str(stage2_file),
            "summary": str(summary_path),
        },
        "baseline": str(baseline_path) if baseline_path else None,
        "verdict": {
            "status": comparison.get("status", "FAIL"),
            "reasons": comparison.get("reasons") or [],
        },
        "stage1": stage1_result,
        "stage2": stage2_result,
        "comparison": comparison,
    }
    eval_result_path.write_text(json.dumps(eval_result_payload, indent=2), encoding="utf-8")

    print(f"[INFO] Summary: {summary_path}")
    print(f"[INFO] Eval result: {eval_result_path}")
    print(f"VERDICT: {comparison.get('status', 'FAIL')}")
    for reason in comparison.get("reasons") or []:
        print(f" - {reason}")


if __name__ == "__main__":
    main()
