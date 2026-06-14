#!/usr/bin/env python3
"""
Mine small/far gate frames from videos and write pseudo-labels in YOLO format.

Selection rule per frame:
- At least `min_gates` detections.
- Median predicted gate height < `max_median_height_px`.
- Keep all predicted gates for selected frames (not just small boxes).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def list_videos(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Video path not found: {path}")
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def ensure_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_dir} (use --overwrite)")
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (
        clamp01(cx / max(w, 1)),
        clamp01(cy / max(h, 1)),
        clamp01(bw / max(w, 1)),
        clamp01(bh / max(h, 1)),
    )


def mine_frames(
    model: YOLO,
    videos: List[Path],
    output_dir: Path,
    max_frames: int,
    min_gates: int,
    min_conf: float,
    nms_iou: float,
    max_median_height_px: float,
    vid_stride: int,
    min_frame_gap: int,
) -> Dict[str, object]:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    selected_rows: List[Dict[str, object]] = []
    total_saved = 0

    for video in videos:
        if total_saved >= max_frames:
            break
        last_saved_frame = -10_000_000
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue

        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if total_saved >= max_frames:
                break
            if frame_idx % max(1, vid_stride) != 0:
                continue
            if frame_idx - last_saved_frame < min_frame_gap:
                continue

            results = model.predict(
                source=frame,
                conf=min_conf,
                iou=nms_iou,
                save=False,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            h, w = frame.shape[:2]

            boxes = []
            heights = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                    conf = float(box.conf[0])
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    if bw < 2 or bh < 2:
                        continue
                    boxes.append((x1, y1, x2, y2, conf))
                    heights.append(bh)

            if len(boxes) < min_gates:
                continue

            heights_sorted = sorted(heights)
            mid = len(heights_sorted) // 2
            if len(heights_sorted) % 2 == 0:
                median_h = (heights_sorted[mid - 1] + heights_sorted[mid]) / 2.0
            else:
                median_h = heights_sorted[mid]

            if median_h >= max_median_height_px:
                continue

            stem = f"smallgate_{video.stem.replace(' ', '_')}_{frame_idx:05d}"
            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"

            cv2.imwrite(str(image_path), frame)
            lines: List[str] = []
            for (x1, y1, x2, y2, _conf) in boxes:
                cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_path.write_text("\n".join(lines), encoding="utf-8")

            total_saved += 1
            last_saved_frame = frame_idx
            selected_rows.append(
                {
                    "video": str(video),
                    "frame": frame_idx,
                    "saved_image": str(image_path),
                    "saved_label": str(label_path),
                    "detections": len(boxes),
                    "median_height_px": median_h,
                }
            )

        cap.release()

    summary = {
        "videos": [str(v) for v in videos],
        "max_frames": max_frames,
        "selected_frames": total_saved,
        "min_gates": min_gates,
        "min_conf": min_conf,
        "nms_iou": nms_iou,
        "max_median_height_px": max_median_height_px,
        "vid_stride": vid_stride,
        "min_frame_gap": min_frame_gap,
        "items": selected_rows,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine small-gate pseudo-labeled frames from videos")
    parser.add_argument("--model", required=True, help="Gate detector model (.pt)")
    parser.add_argument("--videos", required=True, help="Video file or directory containing videos")
    parser.add_argument("--output", required=True, help="Output flat YOLO dataset directory")
    parser.add_argument("--max-frames", type=int, default=40, help="Maximum frames to save")
    parser.add_argument("--min-gates", type=int, default=2, help="Minimum detections per selected frame")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.55, help="NMS IoU")
    parser.add_argument(
        "--max-median-height-px",
        type=float,
        default=120.0,
        help="Only keep frames where median predicted gate height is below this threshold.",
    )
    parser.add_argument("--vid-stride", type=int, default=3, help="Video stride passed to inference")
    parser.add_argument("--min-frame-gap", type=int, default=45, help="Minimum frame index gap per video")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model).resolve()
    videos_arg = Path(args.videos).resolve()
    output_dir = Path(args.output).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    videos = list_videos(videos_arg)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {videos_arg}")

    ensure_output(output_dir, overwrite=bool(args.overwrite))
    model = YOLO(str(model_path))
    summary = mine_frames(
        model=model,
        videos=videos,
        output_dir=output_dir,
        max_frames=int(args.max_frames),
        min_gates=int(args.min_gates),
        min_conf=float(args.min_conf),
        nms_iou=float(args.nms_iou),
        max_median_height_px=float(args.max_median_height_px),
        vid_stride=int(args.vid_stride),
        min_frame_gap=int(args.min_frame_gap),
    )

    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved mined dataset: {output_dir}")


if __name__ == "__main__":
    main()
