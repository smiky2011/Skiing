#!/usr/bin/env python3
"""
Acceptance evaluation for course gate counting accuracy.

Runs the pipeline on unseen test videos and checks that the detected
course gate count meets the >=90% accuracy threshold for each video.
"""
import argparse
import sys
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


def find_video(stem):
    """Find video file matching stem in VIDEO_DIR."""
    for ext in (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".mkv"):
        candidate = VIDEO_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Try partial match
    for f in VIDEO_DIR.iterdir():
        if f.is_file() and stem in f.stem:
            return f
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate course gate counting accuracy")
    parser.add_argument("--model", required=True, help="Path to gate detector model")
    parser.add_argument("--course-gate-conf", type=float, default=None)
    parser.add_argument("--course-gate-stride", type=int, default=None)
    parser.add_argument("--course-gate-min-hits", type=int, default=3)
    parser.add_argument("--course-gate-track-missing-max", type=int, default=8)
    parser.add_argument("--course-gate-fragment-merge-gap-max", type=int, default=45)
    parser.add_argument("--course-gate-match-thresh-ratio", type=float, default=0.06)
    parser.add_argument("--n-runs", type=int, default=1)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    pipeline = SkiRacingPipeline(
        gate_model_path=str(model_path),
        gate_full_track=True,
        gate_init_mode="consensus",
    )

    all_pass = True
    print(f"\n{'Video':<30s} | {'Target':>6s} | {'Detected':>8s} | {'Accuracy':>8s} | {'Status'}")
    print("-" * 80)

    for stem, target in TARGETS.items():
        video_path = find_video(stem)
        if video_path is None:
            print(f"{stem:<30s} | {target:>6d} | {'N/A':>8s} | {'N/A':>8s} | SKIP (not found)")
            all_pass = False
            continue

        counts = []
        for run_i in range(args.n_runs):
            results = pipeline.process_video(
                video_path=str(video_path),
                output_dir="/tmp/eval_course_gate",
                course_gate_count=True,
                course_gate_conf=args.course_gate_conf,
                course_gate_stride=args.course_gate_stride,
                course_gate_min_hits=args.course_gate_min_hits,
                course_gate_track_missing_max=args.course_gate_track_missing_max,
                course_gate_fragment_merge_gap_max=args.course_gate_fragment_merge_gap_max,
                course_gate_match_thresh_ratio=args.course_gate_match_thresh_ratio,
            )
            counts.append(int(results.get("course_gates_count", 0)))

        detected = counts[0] if len(counts) == 1 else counts
        accuracy = min(counts[0], target) / target if target > 0 else 0.0
        status = "PASS" if accuracy >= ACCURACY_THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False

        if args.n_runs == 1:
            print(f"{stem:<30s} | {target:>6d} | {counts[0]:>8d} | {accuracy:>7.1%} | {status}")
        else:
            import statistics
            mean_count = statistics.mean(counts)
            std_count = statistics.stdev(counts) if len(counts) > 1 else 0.0
            print(f"{stem:<30s} | {target:>6d} | {mean_count:>7.1f}±{std_count:.1f} | {accuracy:>7.1%} | {status}")

    print()
    if all_pass:
        print("ALL PASS")
        sys.exit(0)
    else:
        print("SOME FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
