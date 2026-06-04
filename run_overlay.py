from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ski_pose_overlay import ProcessingSettings, process_video


def parse_rect(value: str | None, name: str) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"{name} must be x1,y1,x2,y2")
    return tuple(parts)  # type: ignore[return-value]


def parse_box(value: str | None) -> tuple[float, float, float, float] | None:
    return parse_rect(value, "initial box")


def parse_point(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("target point must be x,y")
    return tuple(parts)  # type: ignore[return-value]


def parse_region(value: str | None) -> tuple[float, float, float, float] | None:
    return parse_rect(value, "target region")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a confidence-aware skeleton overlay for one alpine ski video."
    )
    parser.add_argument("video", help="Input MP4/MOV file")
    parser.add_argument("--output-dir", default="outputs", help="Directory for MP4 and JSON outputs")
    parser.add_argument(
        "--preset",
        default="standard",
        choices=["standard", "far", "occlusion"],
        help="Convenience defaults for difficult footage",
    )
    parser.add_argument("--backend", default="auto", choices=["auto", "yolo", "mediapipe"])
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO pose weights path/name")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker YAML")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=None, help="Model confidence threshold")
    parser.add_argument("--device", default="", help="Optional torch device, for example cpu or mps")
    parser.add_argument("--start-frame", type=int, default=0, help="Start processing at this source frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames")
    parser.add_argument("--initial-box", type=parse_box, default=None, help="Optional x1,y1,x2,y2 target skier box")
    parser.add_argument("--target-point", type=parse_point, default=None, help="Optional x,y target skier hint")
    parser.add_argument("--target-frame", type=int, default=0, help="Frame where target-point applies")
    parser.add_argument("--target-region", type=parse_region, default=None, help="Optional x1,y1,x2,y2 target acquisition region")
    parser.add_argument(
        "--target-strategy",
        default=None,
        choices=["first", "moving"],
        help="first locks immediately; moving waits for a downhill-moving tracklet",
    )
    parser.add_argument("--target-lock-delay-frames", type=int, default=None, help="Minimum frames to observe before moving target lock")
    parser.add_argument("--provisional-target", action="store_true", help="Draw a moving target candidate before final target lock")
    parser.add_argument("--early-provisional-target", action="store_true", help="Draw a relaxed nearest-corridor provisional target before final moving lock")
    parser.add_argument("--target-min-frames", type=int, default=None, help="Moving target strategy minimum seen frames")
    parser.add_argument("--target-min-displacement", type=float, default=None, help="Moving target strategy minimum path displacement")
    parser.add_argument("--target-min-downhill", type=float, default=None, help="Moving target strategy minimum downward image motion")
    parser.add_argument("--target-min-downhill-ratio", type=float, default=None, help="Minimum downhill/displacement ratio for moving target lock")
    parser.add_argument("--target-min-downhill-rate", type=float, default=None, help="Minimum downhill pixels per observed frame for moving target lock")
    parser.add_argument("--target-max-centerline-distance-ratio", type=float, default=None, help="Maximum distance from target-region centerline as a frame-width ratio")
    parser.add_argument("--target-min-track-confidence", type=float, default=None, help="Minimum max bbox confidence for moving target lock")
    parser.add_argument("--target-min-region-progress", type=float, default=None, help="Minimum vertical progress within target region for moving target lock")
    parser.add_argument("--refine-pose", action="store_true", help="Run a second pose pass on the selected skier crop")
    parser.add_argument("--crop-imgsz", type=int, default=None, help="YOLO image size for crop refinement")
    parser.add_argument("--crop-margin", type=float, default=0.50, help="Crop expansion around selected skier")
    parser.add_argument("--min-draw-conf", type=float, default=None, help="Minimum joint confidence to draw")
    parser.add_argument("--observe-conf", type=float, default=None, help="Minimum joint confidence to smooth/observe")
    parser.add_argument("--confident-conf", type=float, default=None, help="Joint confidence for green rendering")
    parser.add_argument("--max-infer-gap", type=int, default=None, help="Short gap length to render inferred joints")
    parser.add_argument("--no-box", action="store_true", help="Hide skier bounding box")
    parser.add_argument("--no-labels", action="store_true", help="Hide status text")
    parser.add_argument("--no-smooth", action="store_true", help="Disable temporal smoothing")
    parser.add_argument("--no-roi-recovery", action="store_true", help="Disable predicted-crop recovery")
    parser.add_argument("--no-motion-inference", action="store_true", help="Disable optical-flow inferred joints")
    parser.add_argument("--recall-region-crop", action="store_true", help="Run an extra upsampled target-region detection pass")
    parser.add_argument("--recall-crop-upscale", type=float, default=None, help="Scale factor for recall region crop inference")
    parser.add_argument("--recall-crop-imgsz", type=int, default=None, help="YOLO image size for recall crop inference")
    parser.add_argument("--recall-crop-conf", type=float, default=None, help="Confidence threshold for recall crop inference")
    args = parser.parse_args()

    preset_defaults = {
        "standard": {
            "imgsz": 1280,
            "conf": 0.25,
            "min_draw_conf": 0.20,
            "observe_conf": 0.15,
            "confident_conf": 0.50,
            "max_infer_gap": 3,
            "crop_imgsz": 960,
            "refine_pose": False,
            "roi_recovery": True,
            "motion_inference": True,
            "recall_region_crop": False,
            "recall_crop_upscale": 2.0,
            "recall_crop_imgsz": 1920,
            "recall_crop_conf": None,
            "target_strategy": "first",
            "target_lock_delay_frames": 0,
            "provisional_target": False,
            "early_provisional_target": False,
            "target_min_frames": 5,
            "target_min_displacement_px": 65.0,
            "target_min_downhill_px": 20.0,
            "target_min_downhill_ratio": 0.0,
            "target_min_downhill_rate_px": 0.0,
            "target_max_centerline_distance_ratio": 1.0,
            "target_min_track_confidence": 0.0,
            "target_min_region_progress": 0.0,
        },
        "far": {
            "imgsz": 1920,
            "conf": 0.05,
            "min_draw_conf": 0.10,
            "observe_conf": 0.08,
            "confident_conf": 0.35,
            "max_infer_gap": 5,
            "crop_imgsz": 960,
            "refine_pose": True,
            "roi_recovery": True,
            "motion_inference": True,
            "recall_region_crop": False,
            "recall_crop_upscale": 2.0,
            "recall_crop_imgsz": 1920,
            "recall_crop_conf": None,
            "target_strategy": "moving",
            "target_lock_delay_frames": 0,
            "provisional_target": False,
            "early_provisional_target": False,
            "target_min_frames": 5,
            "target_min_displacement_px": 45.0,
            "target_min_downhill_px": 10.0,
            "target_min_downhill_ratio": 0.0,
            "target_min_downhill_rate_px": 0.0,
            "target_max_centerline_distance_ratio": 1.0,
            "target_min_track_confidence": 0.0,
            "target_min_region_progress": 0.0,
        },
        "occlusion": {
            "imgsz": 1600,
            "conf": 0.15,
            "min_draw_conf": 0.15,
            "observe_conf": 0.10,
            "confident_conf": 0.40,
            "max_infer_gap": 6,
            "crop_imgsz": 960,
            "refine_pose": True,
            "roi_recovery": True,
            "motion_inference": True,
            "recall_region_crop": False,
            "recall_crop_upscale": 2.0,
            "recall_crop_imgsz": 1920,
            "recall_crop_conf": None,
            "target_strategy": "first",
            "target_lock_delay_frames": 0,
            "provisional_target": False,
            "early_provisional_target": False,
            "target_min_frames": 5,
            "target_min_displacement_px": 65.0,
            "target_min_downhill_px": 20.0,
            "target_min_downhill_ratio": 0.0,
            "target_min_downhill_rate_px": 0.0,
            "target_max_centerline_distance_ratio": 1.0,
            "target_min_track_confidence": 0.0,
            "target_min_region_progress": 0.0,
        },
    }[args.preset]

    settings = ProcessingSettings(
        backend=args.backend,
        model=args.model,
        tracker=args.tracker,
        imgsz=args.imgsz if args.imgsz is not None else preset_defaults["imgsz"],
        conf=args.conf if args.conf is not None else preset_defaults["conf"],
        device=args.device,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        draw_box=not args.no_box,
        draw_labels=not args.no_labels,
        smooth=not args.no_smooth,
        min_draw_conf=args.min_draw_conf if args.min_draw_conf is not None else preset_defaults["min_draw_conf"],
        observe_conf=args.observe_conf if args.observe_conf is not None else preset_defaults["observe_conf"],
        confident_conf=args.confident_conf if args.confident_conf is not None else preset_defaults["confident_conf"],
        max_infer_gap=args.max_infer_gap if args.max_infer_gap is not None else preset_defaults["max_infer_gap"],
        initial_box=args.initial_box,
        target_point=args.target_point,
        target_frame=args.target_frame,
        target_region=args.target_region,
        target_strategy=args.target_strategy if args.target_strategy is not None else preset_defaults["target_strategy"],
        target_lock_delay_frames=(
            args.target_lock_delay_frames
            if args.target_lock_delay_frames is not None
            else preset_defaults["target_lock_delay_frames"]
        ),
        provisional_target=(
            args.provisional_target
            or args.early_provisional_target
            or bool(preset_defaults["provisional_target"])
        ),
        early_provisional_target=(
            args.early_provisional_target or bool(preset_defaults["early_provisional_target"])
        ),
        target_min_frames=args.target_min_frames if args.target_min_frames is not None else preset_defaults["target_min_frames"],
        target_min_displacement_px=(
            args.target_min_displacement
            if args.target_min_displacement is not None
            else preset_defaults["target_min_displacement_px"]
        ),
        target_min_downhill_px=(
            args.target_min_downhill
            if args.target_min_downhill is not None
            else preset_defaults["target_min_downhill_px"]
        ),
        target_min_downhill_ratio=(
            args.target_min_downhill_ratio
            if args.target_min_downhill_ratio is not None
            else preset_defaults["target_min_downhill_ratio"]
        ),
        target_min_downhill_rate_px=(
            args.target_min_downhill_rate
            if args.target_min_downhill_rate is not None
            else preset_defaults["target_min_downhill_rate_px"]
        ),
        target_max_centerline_distance_ratio=(
            args.target_max_centerline_distance_ratio
            if args.target_max_centerline_distance_ratio is not None
            else preset_defaults["target_max_centerline_distance_ratio"]
        ),
        target_min_track_confidence=(
            args.target_min_track_confidence
            if args.target_min_track_confidence is not None
            else preset_defaults["target_min_track_confidence"]
        ),
        target_min_region_progress=(
            args.target_min_region_progress
            if args.target_min_region_progress is not None
            else preset_defaults["target_min_region_progress"]
        ),
        refine_pose=args.refine_pose or bool(preset_defaults["refine_pose"]),
        crop_imgsz=args.crop_imgsz if args.crop_imgsz is not None else preset_defaults["crop_imgsz"],
        crop_margin=args.crop_margin,
        roi_recovery=bool(preset_defaults["roi_recovery"]) and not args.no_roi_recovery,
        motion_inference=bool(preset_defaults["motion_inference"]) and not args.no_motion_inference,
        recall_region_crop=args.recall_region_crop or bool(preset_defaults["recall_region_crop"]),
        recall_crop_upscale=(
            args.recall_crop_upscale
            if args.recall_crop_upscale is not None
            else preset_defaults["recall_crop_upscale"]
        ),
        recall_crop_imgsz=(
            args.recall_crop_imgsz
            if args.recall_crop_imgsz is not None
            else preset_defaults["recall_crop_imgsz"]
        ),
        recall_crop_conf=(
            args.recall_crop_conf
            if args.recall_crop_conf is not None
            else preset_defaults["recall_crop_conf"]
        ),
    )

    def progress(frame: int, total: int) -> None:
        if frame == 1 or frame % 25 == 0 or frame == total:
            print(f"processed {frame}/{total} frames", flush=True)

    result = process_video(
        Path(args.video),
        output_dir=Path(args.output_dir),
        settings=settings,
        progress_callback=progress,
    )
    print(json.dumps({"settings": asdict(settings), "result": asdict(result)}, indent=2))


if __name__ == "__main__":
    main()
