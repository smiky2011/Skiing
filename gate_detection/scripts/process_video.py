"""
Process a ski racing video: gate detection + 2D skier tracking.

Note: 3D coordinate transformation and physics validation have been removed
until gate detection quality is reliable enough to support them.

Usage:
    python scripts/process_video.py data/test_videos/race1.mp4 --gate-model models/gate_detector_best.pt
    python scripts/process_video.py data/test_videos/ --gate-model models/gate_detector_best.pt --discipline giant_slalom
"""
import sys
import json
import time
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ski_racing.pipeline import SkiRacingPipeline
from ski_racing.visualize import create_summary_figure, create_demo_video
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process ski racing video(s)")
    parser.add_argument("input", help="Video file or directory of videos")
    parser.add_argument("--gate-model", required=True, help="Path to gate detector model")
    parser.add_argument("--discipline", default=None,
                        choices=["slalom", "giant_slalom"],
                        help="Discipline. If omitted, auto-detects from gates")
    parser.add_argument("--gate-conf", type=float, default=0.25,
                        help="Gate detection confidence threshold (default 0.25)")
    parser.add_argument("--gate-iou", type=float, default=0.45,
                        help="Gate detection NMS IoU threshold (default 0.45)")
    parser.add_argument("--skier-conf", type=float, default=0.25,
                        help="Skier detection confidence threshold")
    parser.add_argument("--gate-search-frames", type=int, default=300,
                        help="Search first N frames for best gate detection (default 300)")
    parser.add_argument("--gate-search-stride", type=int, default=3,
                        help="Stride for gate search frames (default 3)")
    parser.add_argument("--gate-init-mode", default="single_best",
                        choices=["single_best", "consensus"],
                        help="Gate initialization mode: single best frame or cross-frame consensus.")
    parser.add_argument("--gate-consensus-min-support", type=int, default=3,
                        help="Minimum sampled-frame support for consensus gate seeds (default 3).")
    parser.add_argument("--gate-track-frames", type=int, default=120,
                        help="Frames to use for temporal gate stabilization")
    parser.add_argument("--gate-track-stride", type=int, default=3,
                        help="Stride for gate tracking frames")
    parser.add_argument("--gate-track-min-obs", type=int, default=3,
                        help="Minimum observations per gate to keep")
    parser.add_argument("--output-dir", default="artifacts/outputs", help="Output directory")
    parser.add_argument("--demo-video", action="store_true", help="Also create demo video with overlay")
    parser.add_argument("--live-gate-stride", type=int, default=3,
                        help="Run live gate detection every N frames in demo video (default 3). "
                             "Lower = more accurate but slower rendering.")
    parser.add_argument("--summary", action="store_true", help="Also create summary figure")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Process every Nth frame for tracking")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames processed from start")
    parser.add_argument("--max-jump", type=float, default=None,
                        help="Max pixel jump for temporal tracking")
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
    parser.add_argument("--no-course-gate-count", dest="course_gate_count",
                        action="store_false", default=True,
                        help="Disable course-wide gate counting pass.")
    parser.add_argument("--course-gate-conf", type=float, default=None,
                        help="Confidence threshold for course gate counter (default: auto)")
    parser.add_argument("--course-gate-stride", type=int, default=None,
                        help="Frame stride for course gate counter (default: auto)")
    parser.add_argument("--course-gate-min-hits", type=int, default=3,
                        help="Minimum track hits for course gate counter (default 3)")
    parser.add_argument("--course-gate-track-missing-max", type=int, default=8,
                        help="Max missing samples before dropping track (default 8)")
    parser.add_argument("--course-gate-fragment-merge-gap-max", type=int, default=45,
                        help="Max frame gap for fragment merging (default 45)")
    parser.add_argument("--course-gate-match-thresh-ratio", type=float, default=0.06,
                        help="Match threshold as fraction of frame width (default 0.06)")

    # ── Deprecated flags (one-release compat shim) ──────────────────────────
    # These parameters were removed when the pipeline moved to 2D-first mode.
    # They are accepted silently so existing shell scripts don't break with
    # "unrecognized argument" errors, but they have no effect.
    # TODO(remove-after-next-release): delete these deprecated arguments.
    parser.add_argument("--gate-spacing", dest="_deprecated_gate_spacing",
                        type=float, default=None,
                        help="[deprecated — ignored in 2D-first pipeline, will be removed next release]")
    parser.add_argument("--camera-mode", dest="_deprecated_camera_mode",
                        default=None,
                        help="[deprecated — ignored in 2D-first pipeline, will be removed next release]")
    parser.add_argument("--camera-pitch-deg", "--camera-pitch",
                        dest="_deprecated_camera_pitch_deg",
                        type=float, default=None,
                        help="[deprecated — ignored in 2D-first pipeline, will be removed next release]")
    parser.add_argument("--no-physics", dest="_deprecated_no_physics",
                        action="store_true",
                        help="[deprecated — ignored in 2D-first pipeline, will be removed next release]")
    parser.add_argument("--projection", dest="_deprecated_projection",
                        default=None,
                        help="[deprecated — ignored in 2D-first pipeline, will be removed next release]")
    # ────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    # Load course gate config defaults (only override args still at None/default)
    _cg_config_path = project_root / "configs" / "course_gate_defaults.yaml"
    if _cg_config_path.exists():
        _cg_config = {}
        for _line in _cg_config_path.read_text(encoding="utf-8").splitlines():
            _line = _line.split("#", 1)[0].strip()
            if not _line or ":" not in _line:
                continue
            _k, _v = _line.split(":", 1)
            _k = _k.strip()
            _v = _v.strip()
            try:
                _cg_config[_k] = float(_v) if "." in _v else int(_v)
            except ValueError:
                _cg_config[_k] = _v
        if args.course_gate_conf is None and "course_gate_conf" in _cg_config:
            args.course_gate_conf = float(_cg_config["course_gate_conf"])
        if args.course_gate_stride is None and "course_gate_stride" in _cg_config:
            args.course_gate_stride = int(_cg_config["course_gate_stride"])
        if "course_gate_min_hits" in _cg_config and args.course_gate_min_hits == 3:
            args.course_gate_min_hits = int(_cg_config["course_gate_min_hits"])
        if "track_missing_max" in _cg_config and args.course_gate_track_missing_max == 8:
            args.course_gate_track_missing_max = int(_cg_config["track_missing_max"])
        if "match_thresh_ratio" in _cg_config and args.course_gate_match_thresh_ratio == 0.06:
            args.course_gate_match_thresh_ratio = float(_cg_config["match_thresh_ratio"])
        if "fragment_merge_gap_max" in _cg_config and args.course_gate_fragment_merge_gap_max == 45:
            args.course_gate_fragment_merge_gap_max = int(_cg_config["fragment_merge_gap_max"])

    if args.stabilize:
        print(
            "WARNING --stabilize is deprecated; use --gate-full-track, --outlier-filter, "
            "--kalman instead. Removal target: 2026-04-30."
        )
        args.gate_full_track = True
        args.outlier_filter = True
        args.kalman = True

    # Emit DeprecationWarnings for any deprecated flags the user actually passed.
    _deprecated = {
        "--gate-spacing": args._deprecated_gate_spacing,
        "--camera-mode": args._deprecated_camera_mode,
        "--camera-pitch-deg": args._deprecated_camera_pitch_deg,
        "--no-physics": args._deprecated_no_physics or None,
        "--projection": args._deprecated_projection,
    }
    for flag, value in _deprecated.items():
        if value is not None:
            warnings.warn(
                f"{flag} is deprecated and has no effect in the 2D-first pipeline. "
                "It will be removed in the next release.",
                DeprecationWarning,
                stacklevel=2,
            )

    pipeline = None
    pipeline_init_error = None
    try:
        pipeline = SkiRacingPipeline(
            gate_model_path=args.gate_model,
            discipline=args.discipline,
            gate_init_mode=args.gate_init_mode,
            gate_consensus_min_support=args.gate_consensus_min_support,
            gate_full_track=args.gate_full_track,
            outlier_filter=args.outlier_filter,
            kalman_smooth=args.kalman,
        )
    except Exception as exc:
        pipeline_init_error = exc

    input_path = Path(args.input)
    if input_path.is_dir():
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
        videos = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in video_exts
        )
    else:
        videos = [input_path]

    print(f"Found {len(videos)} video(s) to process\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = output_dir / "run_summary.json"
    existing_summary = []
    if run_summary_path.exists():
        try:
            data = json.loads(run_summary_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                existing_summary = [e for e in data if isinstance(e, dict)]
        except Exception:
            existing_summary = []

    summary_by_video = {
        str(e.get("video")): dict(e)
        for e in existing_summary
        if isinstance(e, dict) and e.get("video")
    }
    processed_in_run = []
    run_summary = []

    for video in videos:
        t0 = time.time()
        if pipeline is None:
            run_summary.append({
                "video": video.name,
                "status": "error",
                "elapsed_s": float(time.time() - t0),
                "error": f"Pipeline init failed: {pipeline_init_error}",
            })
            processed_in_run.append(video.name)
            print(f"✗ Error processing {video.name}: pipeline init failed: {pipeline_init_error}\n")
            continue
        try:
            results = pipeline.process_video(
                video_path=str(video),
                output_dir=args.output_dir,
                gate_conf=args.gate_conf,
                gate_iou=args.gate_iou,
                skier_conf=args.skier_conf,
                gate_search_frames=args.gate_search_frames,
                gate_search_stride=args.gate_search_stride,
                gate_track_frames=args.gate_track_frames,
                gate_track_stride=args.gate_track_stride,
                gate_track_min_obs=args.gate_track_min_obs,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                max_jump=args.max_jump,
                kalman_process_noise=args.kalman_q,
                course_gate_count=args.course_gate_count,
                course_gate_conf=args.course_gate_conf,
                course_gate_stride=args.course_gate_stride,
                course_gate_min_hits=args.course_gate_min_hits,
                course_gate_track_missing_max=args.course_gate_track_missing_max,
                course_gate_fragment_merge_gap_max=args.course_gate_fragment_merge_gap_max,
                course_gate_match_thresh_ratio=args.course_gate_match_thresh_ratio,
            )

            analysis_path = Path(args.output_dir) / f"{video.stem}_analysis.json"

            if args.summary:
                summary_path = Path(args.output_dir) / f"{video.stem}_summary.png"
                create_summary_figure(str(analysis_path), str(summary_path))

            if args.demo_video:
                demo_path = Path(args.output_dir) / f"{video.stem}_demo.mp4"
                create_demo_video(str(video), str(analysis_path), str(demo_path),
                                  gate_model_path=args.gate_model,
                                  live_gate_stride=args.live_gate_stride)

            run_summary.append({
                "video": video.name,
                "status": "ok",
                "elapsed_s": float(time.time() - t0),
            })
            processed_in_run.append(video.name)
            print(f"✓ Successfully processed {video.name}\n")
        except Exception as e:
            run_summary.append({
                "video": video.name,
                "status": "error",
                "elapsed_s": float(time.time() - t0),
                "error": str(e),
            })
            processed_in_run.append(video.name)
            print(f"✗ Error processing {video.name}: {e}\n")

    for entry in run_summary:
        summary_by_video[entry["video"]] = entry
    # Preserve any existing entries not touched in this run.
    combined = []
    for name in processed_in_run:
        if name in summary_by_video:
            combined.append(summary_by_video[name])
    for name in sorted(k for k in summary_by_video.keys() if k not in set(processed_in_run)):
        combined.append(summary_by_video[name])

    with open(run_summary_path, "w") as f:
        json.dump(combined, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
