#!/usr/bin/env python3
"""
Repair calibration hard-case labels using error-case overlays and target GT counts.

Workflow per frame:
1) Start from existing label boxes.
2) Extract candidate GT boxes from overlay colors (green/yellow).
3) Merge candidates with existing boxes by IoU.
4) Match target gt_count from hard_cases_top20.csv by adding model predictions or trimming.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(line: str, w: int, h: int) -> Optional[List[float]]:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cx = float(parts[1]) * w
        cy = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h
    except ValueError:
        return None
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return [x1, y1, x2, y2]


def xyxy_to_yolo(box: List[float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    cx_n = max(0.0, min(1.0, cx / max(w, 1)))
    cy_n = max(0.0, min(1.0, cy / max(h, 1)))
    bw_n = max(0.0, min(1.0, bw / max(w, 1)))
    bh_n = max(0.0, min(1.0, bh / max(h, 1)))
    return f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}"


def read_existing_boxes(label_path: Path, w: int, h: int) -> List[List[float]]:
    if not label_path.exists():
        return []
    boxes: List[List[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        box = yolo_to_xyxy(line, w, h)
        if box is not None:
            boxes.append(box)
    return boxes


def dedupe_boxes(boxes: List[List[float]], iou_thr: float = 0.55) -> List[List[float]]:
    out: List[List[float]] = []
    for box in boxes:
        if any(box_iou(box, b) >= iou_thr for b in out):
            continue
        out.append(box)
    return out


def extract_overlay_candidates(overlay_img: np.ndarray) -> List[List[float]]:
    hsv = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2HSV)
    # Green GT + yellow/orange FN box colors.
    green = cv2.inRange(hsv, (30, 60, 50), (95, 255, 255))
    yellow = cv2.inRange(hsv, (10, 80, 70), (30, 255, 255))
    mask = cv2.bitwise_or(green, yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[List[float]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 65:
            continue
        if h < 18 or w < 3:
            continue
        if (h / max(w, 1)) < 1.0:
            continue
        # Discard tiny legend text near top border.
        if y < 20 and h < 35:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])

    boxes.sort(key=lambda b: (b[3] - b[1]) * (b[2] - b[0]), reverse=True)
    return dedupe_boxes(boxes, iou_thr=0.45)


def prediction_candidates(model: YOLO, image_path: Path) -> List[Tuple[List[float], float]]:
    pred = model.predict(source=str(image_path), conf=0.05, iou=0.55, save=False, verbose=False)
    if not pred:
        return []
    out: List[Tuple[List[float], float]] = []
    boxes = pred[0].boxes
    if boxes is None:
        return out
    for box in boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
        conf = float(box.conf[0])
        out.append(([float(x) for x in xyxy], conf))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


@dataclass
class RepairRow:
    hard_image: str
    calibration_label: str
    target_gt_count: int
    before_count: int
    overlay_candidate_count: int
    prediction_candidate_count: int
    after_count: int
    status: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair calibration hard-case labels")
    parser.add_argument("--dataset", required=True, help="Dataset root containing calibration/images and calibration/labels")
    parser.add_argument("--split-manifest", required=True, help="JSON manifest from calibration split")
    parser.add_argument("--hard-cases-csv", required=True, help="hard_cases_top20.csv")
    parser.add_argument("--overlay-dir", required=True, help="Directory with overlay images")
    parser.add_argument("--model", required=True, help="Model used to fill missing candidates")
    parser.add_argument("--output-report", required=True, help="JSON report path")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    split_manifest = Path(args.split_manifest).resolve()
    hard_cases_csv = Path(args.hard_cases_csv).resolve()
    overlay_dir = Path(args.overlay_dir).resolve()
    output_report = Path(args.output_report).resolve()

    split_data = json.loads(split_manifest.read_text(encoding="utf-8"))
    moved = split_data.get("moved", [])

    target_counts: Dict[str, int] = {}
    with hard_cases_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_counts[row["image"]] = int(row["gt_count"])

    model = YOLO(str(Path(args.model).resolve()))
    rows: List[RepairRow] = []

    for item in moved:
        hard_image = item["hard_image"]
        cal_label_name = item["cal_label"]
        cal_image_name = item["cal_image"]
        target = target_counts.get(hard_image)
        if target is None:
            continue

        image_path = dataset / "calibration" / "images" / cal_image_name
        label_path = dataset / "calibration" / "labels" / cal_label_name
        overlay_path = overlay_dir / hard_image

        frame = cv2.imread(str(image_path))
        if frame is None:
            rows.append(
                RepairRow(
                    hard_image=hard_image,
                    calibration_label=cal_label_name,
                    target_gt_count=target,
                    before_count=0,
                    overlay_candidate_count=0,
                    prediction_candidate_count=0,
                    after_count=0,
                    status="missing_calibration_image",
                )
            )
            continue
        h, w = frame.shape[:2]

        existing = read_existing_boxes(label_path, w, h)
        before = len(existing)
        merged = existing.copy()

        overlay_candidates: List[List[float]] = []
        if overlay_path.exists():
            overlay_img = cv2.imread(str(overlay_path))
            if overlay_img is not None:
                overlay_candidates = extract_overlay_candidates(overlay_img)
                for box in overlay_candidates:
                    if any(box_iou(box, cur) >= 0.45 for cur in merged):
                        continue
                    merged.append(box)

        merged = dedupe_boxes(merged, iou_thr=0.5)

        pred_candidates = prediction_candidates(model, image_path)
        for box, _conf in pred_candidates:
            if len(merged) >= target:
                break
            if any(box_iou(box, cur) >= 0.40 for cur in merged):
                continue
            merged.append(box)

        merged = dedupe_boxes(merged, iou_thr=0.5)
        if len(merged) > target:
            merged.sort(key=lambda b: (b[3] - b[1]) * (b[2] - b[0]), reverse=True)
            merged = merged[:target]

        merged = dedupe_boxes(merged, iou_thr=0.5)
        after = len(merged)
        status = "ok" if after == target else "count_mismatch"

        label_lines = [xyxy_to_yolo(box, w, h) for box in merged]
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        rows.append(
            RepairRow(
                hard_image=hard_image,
                calibration_label=cal_label_name,
                target_gt_count=target,
                before_count=before,
                overlay_candidate_count=len(overlay_candidates),
                prediction_candidate_count=len(pred_candidates),
                after_count=after,
                status=status,
            )
        )

    report = {
        "dataset": str(dataset),
        "split_manifest": str(split_manifest),
        "hard_cases_csv": str(hard_cases_csv),
        "overlay_dir": str(overlay_dir),
        "rows": [row.__dict__ for row in rows],
        "summary": {
            "total": len(rows),
            "ok": sum(1 for r in rows if r.status == "ok"),
            "count_mismatch": sum(1 for r in rows if r.status == "count_mismatch"),
            "missing_calibration_image": sum(1 for r in rows if r.status == "missing_calibration_image"),
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote repair report: {output_report}")


if __name__ == "__main__":
    main()

