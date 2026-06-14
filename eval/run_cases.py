from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ski_pose_overlay import ProcessingSettings, process_video


PRESETS: dict[str, dict[str, Any]] = {
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
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a list at key 'cases'")
    return cases


def settings_for_case(
    case: dict[str, Any], full_video: bool = False, model_override: str | None = None
) -> ProcessingSettings:
    preset_name = str(case.get("preset", "standard"))
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}' in case {case.get('id')}")
    preset = PRESETS[preset_name]

    return ProcessingSettings(
        backend=str(case.get("backend", "auto")),
        model=str(model_override or case.get("model", "ski_pose_v1.pt")),
        tracker=str(case.get("tracker", "bytetrack.yaml")),
        imgsz=int(case.get("imgsz", preset["imgsz"])),
        conf=float(case.get("conf", preset["conf"])),
        device=str(case.get("device", "")),
        start_frame=0 if full_video else int(case.get("start_frame", 0)),
        max_frames=None
        if full_video
        else int(case["max_frames"])
        if case.get("max_frames") is not None
        else None,
        draw_box=bool(case.get("draw_box", True)),
        draw_labels=bool(case.get("draw_labels", True)),
        smooth=bool(case.get("smooth", True)),
        min_draw_conf=float(case.get("min_draw_conf", preset["min_draw_conf"])),
        observe_conf=float(case.get("observe_conf", preset["observe_conf"])),
        confident_conf=float(case.get("confident_conf", preset["confident_conf"])),
        max_infer_gap=int(case.get("max_infer_gap", preset["max_infer_gap"])),
        initial_box=tuple(case["initial_box"]) if case.get("initial_box") else None,
        target_point=tuple(case["target_point"]) if case.get("target_point") else None,
        target_frame=int(case.get("target_frame", 0)),
        target_region=tuple(case["target_region"]) if case.get("target_region") else None,
        target_strategy=str(case.get("target_strategy", preset["target_strategy"])),
        target_lock_delay_frames=int(
            case.get("target_lock_delay_frames", preset["target_lock_delay_frames"])
        ),
        provisional_target=bool(case.get("provisional_target", preset["provisional_target"])),
        early_provisional_target=bool(
            case.get("early_provisional_target", preset["early_provisional_target"])
        ),
        target_min_frames=int(case.get("target_min_frames", preset["target_min_frames"])),
        target_min_displacement_px=float(
            case.get("target_min_displacement_px", preset["target_min_displacement_px"])
        ),
        target_min_downhill_px=float(
            case.get("target_min_downhill_px", preset["target_min_downhill_px"])
        ),
        target_min_downhill_ratio=float(
            case.get("target_min_downhill_ratio", preset["target_min_downhill_ratio"])
        ),
        target_min_downhill_rate_px=float(
            case.get("target_min_downhill_rate_px", preset["target_min_downhill_rate_px"])
        ),
        target_max_centerline_distance_ratio=float(
            case.get(
                "target_max_centerline_distance_ratio",
                preset["target_max_centerline_distance_ratio"],
            )
        ),
        target_min_track_confidence=float(
            case.get("target_min_track_confidence", preset["target_min_track_confidence"])
        ),
        target_min_region_progress=float(
            case.get("target_min_region_progress", preset["target_min_region_progress"])
        ),
        refine_pose=bool(case.get("refine_pose", preset["refine_pose"])),
        crop_imgsz=int(case.get("crop_imgsz", preset["crop_imgsz"])),
        crop_margin=float(case.get("crop_margin", 0.50)),
        too_small_height_ratio=float(case.get("too_small_height_ratio", 0.08)),
        far_skier_height_ratio=float(case.get("far_skier_height_ratio", 0.14)),
        roi_recovery=bool(case.get("roi_recovery", preset["roi_recovery"])),
        motion_inference=bool(case.get("motion_inference", preset["motion_inference"])),
        recall_region_crop=bool(case.get("recall_region_crop", preset["recall_region_crop"])),
        recall_crop_upscale=float(case.get("recall_crop_upscale", preset["recall_crop_upscale"])),
        recall_crop_imgsz=int(case.get("recall_crop_imgsz", preset["recall_crop_imgsz"])),
        recall_crop_conf=(
            float(case["recall_crop_conf"])
            if case.get("recall_crop_conf") is not None
            else preset["recall_crop_conf"]
        ),
    )


def safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 5) if denominator else 0.0


def run_case(
    case: dict[str, Any],
    output_root: Path,
    full_video: bool = False,
    model_override: str | None = None,
) -> dict[str, Any]:
    case_id = str(case["id"])
    video = Path(str(case["video"])).expanduser()
    output_dir = output_root / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = settings_for_case(case, full_video=full_video, model_override=model_override)

    print(f"\n== {case_id} ==")
    print(f"video: {video}")
    print(f"output: {output_dir}")

    def progress(frame: int, total: int) -> None:
        if frame == 1 or frame % 25 == 0 or frame == total:
            print(f"  processed {frame}/{total}", flush=True)

    result = process_video(
        video,
        output_dir=output_dir,
        settings=settings,
        progress_callback=progress,
    )
    diagnostics = {}
    report_path = Path(result.report_json)
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as file:
            diagnostics = json.load(file).get("diagnostics", {})

    summary = {
        "id": case_id,
        "video": str(video),
        "full_video": full_video,
        "purpose": case.get("purpose", ""),
        "review_focus": case.get("review_focus", []),
        "output_dir": str(output_dir),
        "output_video": result.output_video,
        "keypoints_json": result.keypoints_json,
        "report_json": result.report_json,
        "backend": result.backend,
        "frames_processed": result.frames_processed,
        "frames_with_pose": result.frames_with_pose,
        "tracking_lost_frames": result.tracking_lost_frames,
        "low_confidence_frames": result.low_confidence_frames,
        "pose_coverage": safe_ratio(result.frames_with_pose, result.frames_processed),
        "tracking_lost_rate": safe_ratio(result.tracking_lost_frames, result.frames_processed),
        "low_confidence_rate": safe_ratio(result.low_confidence_frames, result.frames_processed),
        "elapsed_sec": round(result.elapsed_sec, 2),
        "diagnostics": diagnostics,
        "settings": asdict(settings),
    }
    with (output_dir / "case_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def write_markdown_summary(summaries: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        "| Case | Frames | Pose Coverage | Tracking Lost | Low Confidence | Failure Counts | Overlay |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in summaries:
        failure_counts = item.get("diagnostics", {}).get("failure_counts", {})
        failure_text = ", ".join(f"{key}:{value}" for key, value in sorted(failure_counts.items()))
        if not failure_text:
            failure_text = "-"
        lines.append(
            "| {id} | {frames_processed} | {pose_coverage:.1%} | "
            "{tracking_lost_rate:.1%} | {low_confidence_rate:.1%} | {failure_text} | `{output_video}` |".format(
                failure_text=failure_text, **item
            )
        )
    lines.append("")
    lines.append("Review each overlay visually before using these numbers to choose a fix.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ski pose detection eval cases.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["eval/cases.json"],
        help="Path to cases manifest, or one or more case ids from the default manifest",
    )
    parser.add_argument("--output-root", default="outputs/eval_baseline", help="Output directory")
    parser.add_argument("--case-id", action="append", help="Run only the given case id; may be repeated")
    parser.add_argument("--full-video", action="store_true", help="Ignore per-case max_frames and process full videos")
    parser.add_argument("--model", default=None, help="Override the pose model for all cases (e.g. yolo11n-pose.pt for a stock baseline)")
    args = parser.parse_args()

    case_args = list(args.cases)
    manifest_path = Path(case_args[0])
    if len(case_args) == 1 and manifest_path.exists():
        cases = load_cases(manifest_path)
        case_ids = []
    elif len(case_args) > 1 and manifest_path.exists():
        cases = load_cases(manifest_path)
        case_ids = case_args[1:]
    else:
        cases = load_cases(Path("eval/cases.json"))
        case_ids = case_args

    wanted = set([*(args.case_id or []), *case_ids])
    if wanted:
        cases = [case for case in cases if case.get("id") in wanted]
        missing = wanted - {str(case.get("id")) for case in cases}
        if missing:
            raise ValueError(f"Unknown case id(s): {', '.join(sorted(missing))}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_case(case, output_root, full_video=args.full_video, model_override=args.model)
        for case in cases
    ]
    summary_json = output_root / "summary.json"
    summary_md = output_root / "summary.md"
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump({"cases": summaries}, file, ensure_ascii=False, indent=2)
    write_markdown_summary(summaries, summary_md)

    print(f"\nWrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
