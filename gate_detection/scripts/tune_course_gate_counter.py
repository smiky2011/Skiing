#!/usr/bin/env python3
"""
Automated parameter tuning for CourseGateCounter.

3-phase search that runs until all three target videos hit >=90% accuracy
or --max-iterations is reached. Writes winning params to
configs/course_gate_defaults.yaml.
"""
import argparse
import csv
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ski_racing.pipeline import SkiRacingPipeline

TARGETS = {
    "1592_raw": 32,
    "mmexport1704641869528": 28,
    "mmexport1706076255933": 24,
}
ACCURACY_THRESHOLD = 0.90
VIDEO_DIR = PROJECT_ROOT / "tests" / "unseen_videos_20260228"

PARAM_GRID = {
    "course_gate_conf": [0.15, 0.20, 0.25],
    "min_hits": [2, 3, 4],
    "track_missing_max": [6, 8, 12],
    "fragment_merge_gap_max": [30, 45, 60],
    "match_thresh_ratio": [0.05, 0.06, 0.08],
}

FIXED_PARAMS = {
    "course_gate_stride": 2,
    "dedup_dx_ratio": 0.03,
    "dedup_dy_ratio": 0.05,
    "dedup_overlap_thresh": 0.50,
    "fragment_merge_dist_ratio": 0.10,
}

DEFAULT_PARAMS = {
    "course_gate_conf": 0.20,
    "min_hits": 3,
    "track_missing_max": 8,
    "fragment_merge_gap_max": 45,
    "match_thresh_ratio": 0.06,
    **FIXED_PARAMS,
}


def find_video(stem):
    for ext in (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".mkv"):
        candidate = VIDEO_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    for f in VIDEO_DIR.iterdir():
        if f.is_file() and stem in f.stem:
            return f
    return None


def _write_yaml(path, params, stamp):
    lines = [
        f"# Written by tune_course_gate_counter.py on {stamp}",
        f"course_gate_conf: {params['course_gate_conf']}",
        f"course_gate_stride: {params.get('course_gate_stride', 2)}",
        f"course_gate_min_hits: {params['min_hits']}",
        f"track_missing_max: {params['track_missing_max']}",
        f"match_thresh_ratio: {params['match_thresh_ratio']}",
        f"fragment_merge_gap_max: {params['fragment_merge_gap_max']}",
        f"dedup_dx_ratio: 0.03",
        f"dedup_dy_ratio: 0.05",
        f"dedup_overlap_thresh: 0.50",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_params(pipeline, params, video_map):
    """Run pipeline with given params on all videos, return per-video results."""
    per_video = {}
    for stem, target in TARGETS.items():
        video_path = video_map.get(stem)
        if video_path is None:
            per_video[stem] = {"target": target, "detected": 0, "accuracy": 0.0}
            continue
        results = pipeline.process_video(
            video_path=str(video_path),
            output_dir="/tmp/tune_course_gate",
            course_gate_count=True,
            course_gate_conf=params["course_gate_conf"],
            course_gate_stride=params.get("course_gate_stride", 2),
            course_gate_min_hits=params["min_hits"],
            course_gate_track_missing_max=params["track_missing_max"],
            course_gate_fragment_merge_gap_max=params["fragment_merge_gap_max"],
            course_gate_match_thresh_ratio=params["match_thresh_ratio"],
        )
        detected = int(results.get("course_gates_count", 0))
        accuracy = min(detected, target) / target if target > 0 else 0.0
        per_video[stem] = {"target": target, "detected": detected, "accuracy": accuracy}
    return per_video


def all_pass(per_video):
    return all(v["accuracy"] >= ACCURACY_THRESHOLD for v in per_video.values())


def min_accuracy(per_video):
    if not per_video:
        return 0.0
    return min(v["accuracy"] for v in per_video.values())


def format_result(iteration, max_iter, params, per_video):
    parts = [f"[iter {iteration:>3d}/{max_iter}]"]
    parts.append(f"conf={params['course_gate_conf']:.2f}")
    parts.append(f"min_hits={params['min_hits']}")
    parts.append(f"tmm={params['track_missing_max']}")
    parts.append(f"fmg={params['fragment_merge_gap_max']}")
    parts.append(f"mtr={params['match_thresh_ratio']:.2f}")
    parts.append("|")
    for stem in ("1592_raw", "mmexport1704641869528", "mmexport1706076255933"):
        v = per_video.get(stem, {"detected": 0, "target": 0, "accuracy": 0})
        short = stem[:6] if len(stem) > 6 else stem
        parts.append(f"{short}:{v['detected']}/{v['target']}({v['accuracy']:.1%})")
    status = "ALL PASS" if all_pass(per_video) else "FAIL"
    parts.append(f"| {status}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Tune CourseGateCounter parameters")
    parser.add_argument("--model", required=True, help="Path to gate detector model")
    parser.add_argument("--max-iterations", type=int, default=50)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    pipeline = SkiRacingPipeline(
        gate_model_path=str(model_path),
        gate_full_track=True,
        gate_init_mode="consensus",
    )

    # Discover videos
    video_map = {}
    for stem in TARGETS:
        vp = find_video(stem)
        if vp is None:
            print(f"[WARN] Video not found for {stem}")
        else:
            video_map[stem] = vp

    if not video_map:
        print("[ERROR] No videos found. Cannot tune.")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = PROJECT_ROOT / "tests" / f"course_gate_tuning_{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "sweep_results.csv"
    yaml_path = PROJECT_ROOT / "configs" / "course_gate_defaults.yaml"

    all_results = []
    iteration = 0
    max_iter = args.max_iterations

    # Phase 1: Try defaults
    iteration += 1
    per_video = evaluate_params(pipeline, DEFAULT_PARAMS, video_map)
    print(format_result(iteration, max_iter, DEFAULT_PARAMS, per_video))
    all_results.append({"params": dict(DEFAULT_PARAMS), "per_video": per_video, "min_acc": min_accuracy(per_video)})
    if all_pass(per_video):
        print("\nPhase 1 (defaults): ALL PASS!")
        _write_yaml(yaml_path, DEFAULT_PARAMS, stamp)
        _write_csv(csv_path, all_results)
        sys.exit(0)

    # Phase 2: Coarse grid sweep
    grid_keys = list(PARAM_GRID.keys())
    grid_values = [PARAM_GRID[k] for k in grid_keys]
    combos = list(itertools.product(*grid_values))

    for combo in combos:
        if iteration >= max_iter:
            break
        iteration += 1
        params = dict(FIXED_PARAMS)
        for k, v in zip(grid_keys, combo):
            params[k] = v
        # Skip if same as default (already tried)
        if all(params.get(k) == DEFAULT_PARAMS.get(k) for k in grid_keys):
            continue
        per_video = evaluate_params(pipeline, params, video_map)
        print(format_result(iteration, max_iter, params, per_video))
        all_results.append({"params": dict(params), "per_video": per_video, "min_acc": min_accuracy(per_video)})
        if all_pass(per_video):
            print(f"\nPhase 2 (grid): ALL PASS at iteration {iteration}!")
            _write_yaml(yaml_path, params, stamp)
            _write_csv(csv_path, all_results)
            sys.exit(0)

    # Phase 3: Fine-tune around best candidate
    best_row = max(all_results, key=lambda r: r["min_acc"])
    best_params = best_row["params"]
    print(f"\nPhase 3: Fine-tuning around best candidate (min_acc={best_row['min_acc']:.1%})")

    # Generate neighborhood by varying each param ±1 step
    neighbors = []
    for key in grid_keys:
        vals = PARAM_GRID[key]
        current = best_params.get(key)
        if current in vals:
            idx = vals.index(current)
            for offset in (-1, 1):
                ni = idx + offset
                if 0 <= ni < len(vals) and vals[ni] != current:
                    neighbor = dict(best_params)
                    neighbor[key] = vals[ni]
                    neighbors.append(neighbor)

    for params in neighbors:
        if iteration >= max_iter:
            break
        iteration += 1
        per_video = evaluate_params(pipeline, params, video_map)
        print(format_result(iteration, max_iter, params, per_video))
        all_results.append({"params": dict(params), "per_video": per_video, "min_acc": min_accuracy(per_video)})
        if all_pass(per_video):
            print(f"\nPhase 3 (fine-tune): ALL PASS at iteration {iteration}!")
            _write_yaml(yaml_path, params, stamp)
            _write_csv(csv_path, all_results)
            sys.exit(0)

    # Exhausted
    best_row = max(all_results, key=lambda r: r["min_acc"])
    print(f"\nExhausted {iteration} iterations. Best min_accuracy = {best_row['min_acc']:.1%}")
    _write_yaml(yaml_path, best_row["params"], stamp)
    _write_csv(csv_path, all_results)
    sys.exit(1)


def _write_csv(csv_path, all_results):
    if not all_results:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["iteration", "course_gate_conf", "min_hits", "track_missing_max",
                   "fragment_merge_gap_max", "match_thresh_ratio", "min_accuracy"]
        for stem in TARGETS:
            header.extend([f"{stem}_detected", f"{stem}_accuracy"])
        writer.writerow(header)
        for i, row in enumerate(all_results, 1):
            line = [
                i,
                row["params"].get("course_gate_conf"),
                row["params"].get("min_hits"),
                row["params"].get("track_missing_max"),
                row["params"].get("fragment_merge_gap_max"),
                row["params"].get("match_thresh_ratio"),
                f"{row['min_acc']:.4f}",
            ]
            for stem in TARGETS:
                v = row["per_video"].get(stem, {})
                line.append(v.get("detected", 0))
                line.append(f"{v.get('accuracy', 0):.4f}")
            writer.writerow(line)


if __name__ == "__main__":
    main()
