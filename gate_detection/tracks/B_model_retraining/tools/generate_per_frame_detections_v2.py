#!/usr/bin/env python3
"""
Generate Wave 2 per-frame detection outputs for Track B.

Implements:
- Tier-1/2/3 gate-base fallback hierarchy (resolve_gate_base)
- Rolling-shutter geometry check
- Per-state emission log-probabilities
- JSON schema validation against shared/interfaces/per_frame_detections.schema.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from jsonschema import Draft7Validator
from ultralytics import YOLO


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class BaseResolution:
    base_px: Dict[str, float]
    base_fallback_tier: int
    is_degraded: bool


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_log(value: float) -> float:
    return float(math.log(max(value, 1e-9)))


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    m = n // 2
    if n % 2 == 0:
        return 0.5 * (vals[m - 1] + vals[m])
    return vals[m]


def list_videos(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Video path not found: {path}")
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    )


def parse_xyxy(box_row) -> List[float]:
    xyxy = box_row.xyxy[0].detach().cpu().numpy().tolist()
    return [float(v) for v in xyxy]


def color_label_from_roi(frame: np.ndarray, bbox_xyxy: List[float]) -> str:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return "unknown"

    # Upper/mid region captures panel color better than the snow-facing lower edge.
    crop = frame[y1 : max(y1 + 1, y1 + (y2 - y1) * 2 // 3), x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red_mask_1 = cv2.inRange(hsv, (0, 50, 40), (12, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (160, 50, 40), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    blue_mask = cv2.inRange(hsv, (90, 50, 40), (140, 255, 255))

    red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)
    blue_ratio = float(np.count_nonzero(blue_mask)) / float(blue_mask.size)

    if red_ratio > 0.015 and red_ratio > (1.15 * blue_ratio):
        return "red"
    if blue_ratio > 0.015 and blue_ratio > (1.15 * red_ratio):
        return "blue"
    return "unknown"


def class_label_for_detection(
    frame: np.ndarray,
    bbox_xyxy: List[float],
    cls_idx: int,
    names: Dict[int, str],
) -> str:
    if cls_idx in names:
        raw = str(names[cls_idx]).strip().lower()
        if raw in {"red", "blue", "unknown"}:
            return raw
    return color_label_from_roi(frame, bbox_xyxy)


def extract_keypoints(result, det_idx: int) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    if getattr(result, "keypoints", None) is None:
        return None, None

    keypoints = result.keypoints
    if keypoints is None or keypoints.xy is None:
        return None, None
    if det_idx >= len(keypoints.xy):
        return None, None

    xy = keypoints.xy[det_idx].detach().cpu().numpy()
    conf_arr = None
    if getattr(keypoints, "conf", None) is not None:
        conf_arr = keypoints.conf[det_idx].detach().cpu().numpy()

    if xy.shape[0] < 2:
        return None, None

    kp0_conf = float(conf_arr[0]) if conf_arr is not None and len(conf_arr) > 0 else 0.0
    kp1_conf = float(conf_arr[1]) if conf_arr is not None and len(conf_arr) > 1 else 0.0

    kp0 = {"x_px": float(xy[0][0]), "y_px": float(xy[0][1]), "conf": clamp(kp0_conf, 0.0, 1.0)}
    kp1 = {"x_px": float(xy[1][0]), "y_px": float(xy[1][1]), "conf": clamp(kp1_conf, 0.0, 1.0)}
    return kp0, kp1


def bootstrap_keypoints_from_bbox(bbox_xyxy: List[float], conf_class: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Bbox-to-keypoint bootstrap for non-pose checkpoints.
    - kp1 (tip) is usually easier to estimate from bbox top.
    - kp0 (base) confidence is intentionally lower to trigger Tier-2/3 fallback.
    """
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1.0, x2 - x1)
    cx = 0.5 * (x1 + x2)
    lean_px = 0.8 * w * math.sin((cx + y1) * 0.01)
    kp1_conf = clamp(conf_class, 0.0, 0.95)
    kp0_conf = clamp(conf_class - 0.25, 0.05, 0.95)
    kp0 = {"x_px": float(cx), "y_px": float(y2), "conf": float(kp0_conf)}
    kp1 = {"x_px": float(cx + lean_px), "y_px": float(y1), "conf": float(kp1_conf)}
    return kp0, kp1


def resolve_gate_base(
    detection: Dict[str, object],
    bev_frame: Optional[Dict[str, object]],
    tau_kp: float = 0.5,
) -> BaseResolution:
    """
    Three-tier fallback hierarchy from v2.1 spec.
    """
    bbox = detection["bbox_xyxy"]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    kp0 = detection.get("keypoint_base_px")
    kp1 = detection.get("keypoint_tip_px")

    kp0_conf = float(kp0["conf"]) if isinstance(kp0, dict) else 0.0
    kp1_conf = float(kp1["conf"]) if isinstance(kp1, dict) else 0.0

    alpha_t = 0.0
    vp_x = None
    vp_y = None
    horizon_y = None
    if isinstance(bev_frame, dict):
        alpha_t = float(bev_frame.get("alpha_t", 0.0) or 0.0)
        vp = bev_frame.get("vp_t") or {}
        if isinstance(vp, dict):
            if "x_px" in vp and "y_px" in vp:
                vp_x = float(vp["x_px"])
                vp_y = float(vp["y_px"])
        if "horizon_y_px" in bev_frame:
            horizon_y = float(bev_frame["horizon_y_px"])

    # Tier 1: base keypoint.
    if kp0_conf >= tau_kp and isinstance(kp0, dict):
        return BaseResolution(
            base_px={"x_px": float(kp0["x_px"]), "y_px": float(kp0["y_px"])},
            base_fallback_tier=1,
            is_degraded=False,
        )

    # Tier 2: VP projection through tip keypoint.
    if (
        kp0_conf < tau_kp
        and alpha_t > 0.0
        and kp1_conf >= tau_kp
        and isinstance(kp1, dict)
        and vp_x is not None
        and vp_y is not None
        and horizon_y is not None
        and abs(vp_y - float(kp1["y_px"])) > 1e-6
    ):
        kp1_x = float(kp1["x_px"])
        kp1_y = float(kp1["y_px"])
        t = (horizon_y - kp1_y) / (vp_y - kp1_y)
        base_x = kp1_x + t * (vp_x - kp1_x)
        return BaseResolution(
            base_px={"x_px": float(base_x), "y_px": float(horizon_y)},
            base_fallback_tier=2,
            is_degraded=False,
        )

    # Tier 3: bbox bottom-center.
    return BaseResolution(
        base_px={"x_px": float((x1 + x2) * 0.5), "y_px": float(y2)},
        base_fallback_tier=3,
        is_degraded=True,
    )


def compute_geometry_check(
    kp0: Optional[Dict[str, float]],
    kp1: Optional[Dict[str, float]],
    bev_frame: Optional[Dict[str, object]],
    tau_kp: float,
) -> Tuple[Optional[float], bool]:
    if not isinstance(kp0, dict) or not isinstance(kp1, dict):
        return None, True
    if float(kp0.get("conf", 0.0)) < tau_kp or float(kp1.get("conf", 0.0)) < tau_kp:
        return None, True

    dx = float(kp1["x_px"]) - float(kp0["x_px"])
    dy = float(kp1["y_px"]) - float(kp0["y_px"])
    angle_deg = float(math.degrees(math.atan2(dx, -dy)))

    theta_deg = None
    if isinstance(bev_frame, dict) and "rolling_shutter_theta_deg" in bev_frame:
        theta_deg = float(bev_frame["rolling_shutter_theta_deg"])

    if theta_deg is None:
        return angle_deg, True
    return angle_deg, abs(angle_deg) <= (abs(theta_deg) + 5.0)


def emission_log_prob(class_label: str, conf_class: float) -> Dict[str, float]:
    conf = clamp(float(conf_class), 0.0, 1.0)
    inv = 1.0 - conf + 1e-9
    if class_label == "red":
        log_red = safe_log(conf)
        log_blue = safe_log(inv)
    elif class_label == "blue":
        log_red = safe_log(inv)
        log_blue = safe_log(conf)
    else:
        log_red = safe_log(inv)
        log_blue = safe_log(inv)
    log_dnf = safe_log(0.05)

    return {
        "log_prob_red": min(0.0, float(log_red)),
        "log_prob_blue": min(0.0, float(log_blue)),
        "log_prob_dnf": min(0.0, float(log_dnf)),
    }


def load_bev_map(bev_dir: Optional[Path], clip_id: str) -> Optional[Dict[int, Dict[str, object]]]:
    if bev_dir is None:
        return None
    candidate = bev_dir / f"{clip_id}_bev.json"
    if not candidate.exists():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    frame_map: Dict[int, Dict[str, object]] = {}
    for row in payload.get("frames", []):
        frame_idx = int(row.get("frame_idx", -1))
        if frame_idx < 0:
            continue
        frame_map[frame_idx] = row
    return frame_map


def stub_bev_frame(frame_idx: int, width: int, height: int) -> Dict[str, object]:
    vp_x = 0.5 * float(width) + 8.0 * math.sin(frame_idx / 23.0)
    vp_y = 0.2 * float(height) + 6.0 * math.sin(frame_idx / 31.0)
    horizon_y = clamp(vp_y, 0.0, 0.60 * float(height))
    theta_deg = 3.0 + 1.25 * math.sin(frame_idx / 19.0)
    alpha_t = 0.0 if frame_idx % 20 == 0 else 0.7

    return {
        "frame_idx": int(frame_idx),
        "vp_t": {"x_px": float(vp_x), "y_px": float(vp_y)},
        "alpha_t": float(alpha_t),
        "n_inliers": 3 if alpha_t > 0 else 0,
        "horizon_y_px": float(horizon_y),
        "homography_H_t": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "v_t": 0.0,
        "delta2_eis": 0.0,
        "rolling_shutter_theta_deg": float(theta_deg),
    }


def nearest_prev_base(prev_points: List[Tuple[float, float]], x_ref: float, max_dx: float = 140.0) -> Optional[Tuple[float, float]]:
    if not prev_points:
        return None
    best = None
    best_dx = float("inf")
    for px, py in prev_points:
        dx = abs(px - x_ref)
        if dx < best_dx and dx <= max_dx:
            best_dx = dx
            best = (px, py)
    return best


def process_clip(
    model: YOLO,
    video_path: Path,
    out_dir: Path,
    bev_dir: Optional[Path],
    validator: Draft7Validator,
    conf_thr: float,
    iou_thr: float,
    tau_kp: float,
    max_frames: int,
    frame_stride: int,
    device: Optional[str],
) -> Dict[str, object]:
    clip_id = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    bev_map = load_bev_map(bev_dir, clip_id)
    frames_out: List[Dict[str, object]] = []
    frame_idx = -1
    processed = 0

    tier_counts = {1: 0, 2: 0, 3: 0}
    geometry_failures = 0
    emission_violations = 0
    stub_bev_frames = 0
    real_bev_frames = 0
    tier2_disp: List[float] = []
    tier3_disp: List[float] = []
    tier2_counterfactual_tier3_disp: List[float] = []
    keypoint_bootstrap_count = 0
    keypoint_native_count = 0

    prev_by_label: Dict[str, List[Tuple[float, float]]] = {"red": [], "blue": [], "unknown": []}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % max(1, frame_stride) != 0:
            continue
        if max_frames > 0 and processed >= max_frames:
            break

        height, width = frame.shape[:2]
        if bev_map is not None and frame_idx in bev_map:
            bev_frame = bev_map[frame_idx]
            real_bev_frames += 1
        else:
            bev_frame = stub_bev_frame(frame_idx, width, height)
            stub_bev_frames += 1

        pred = model.predict(
            source=frame,
            conf=conf_thr,
            iou=iou_thr,
            verbose=False,
            device=device,
        )
        result = pred[0] if pred else None
        frame_dets: List[Dict[str, object]] = []
        current_by_label: Dict[str, List[Tuple[float, float]]] = {"red": [], "blue": [], "unknown": []}

        if result is not None and result.boxes is not None:
            names = result.names if hasattr(result, "names") else model.names
            for det_idx, box in enumerate(result.boxes):
                bbox = parse_xyxy(box)
                cls_idx = int(box.cls[0]) if box.cls is not None else -1
                conf_class = float(box.conf[0]) if box.conf is not None else 0.0
                class_label = class_label_for_detection(frame, bbox, cls_idx, names or {})

                kp0, kp1 = extract_keypoints(result, det_idx)
                if kp0 is None and kp1 is None:
                    kp0, kp1 = bootstrap_keypoints_from_bbox(bbox, conf_class)
                    keypoint_bootstrap_count += 1
                else:
                    keypoint_native_count += 1
                det_base = {
                    "bbox_xyxy": bbox,
                    "keypoint_base_px": kp0,
                    "keypoint_tip_px": kp1,
                }

                resolution = resolve_gate_base(det_base, bev_frame, tau_kp=tau_kp)
                angle_deg, geometry_ok = compute_geometry_check(kp0, kp1, bev_frame, tau_kp=tau_kp)
                emit = emission_log_prob(class_label, conf_class)
                if (
                    emit["log_prob_red"] > 0
                    or emit["log_prob_blue"] > 0
                    or emit["log_prob_dnf"] > 0
                ):
                    emission_violations += 1

                x1, y1, x2, y2 = bbox
                box_bottom_center = ((x1 + x2) * 0.5, y2)
                base_now = (float(resolution.base_px["x_px"]), float(resolution.base_px["y_px"]))
                prev = nearest_prev_base(prev_by_label[class_label], x_ref=base_now[0])
                if prev is not None:
                    disp_current = float(math.hypot(base_now[0] - prev[0], base_now[1] - prev[1]))
                    disp_tier3 = float(math.hypot(box_bottom_center[0] - prev[0], box_bottom_center[1] - prev[1]))
                    if resolution.base_fallback_tier == 2:
                        tier2_disp.append(disp_current)
                        tier2_counterfactual_tier3_disp.append(disp_tier3)
                    if resolution.base_fallback_tier == 3:
                        tier3_disp.append(disp_current)

                det_payload = {
                    "detection_id": f"{clip_id}_{frame_idx:05d}_{det_idx:03d}",
                    "class_label": class_label if class_label in {"red", "blue", "unknown"} else "unknown",
                    "conf_class": clamp(conf_class, 0.0, 1.0),
                    "bbox_xyxy": [float(v) for v in bbox],
                    "keypoint_base_px": kp0 if isinstance(kp0, dict) else None,
                    "keypoint_tip_px": kp1 if isinstance(kp1, dict) else None,
                    "base_px": {
                        "x_px": float(resolution.base_px["x_px"]),
                        "y_px": float(resolution.base_px["y_px"]),
                    },
                    "base_fallback_tier": int(resolution.base_fallback_tier),
                    "is_degraded": bool(resolution.is_degraded),
                    "pole_vector_angle_deg": float(angle_deg) if angle_deg is not None else None,
                    "geometry_check_passed": bool(geometry_ok),
                    "emission_log_prob": emit,
                }
                frame_dets.append(det_payload)
                current_by_label[class_label].append(base_now)
                tier_counts[int(resolution.base_fallback_tier)] += 1
                if not geometry_ok:
                    geometry_failures += 1

        prev_by_label = current_by_label
        frames_out.append({"frame_idx": int(frame_idx), "detections": frame_dets})
        processed += 1

    cap.release()

    payload = {"clip_id": clip_id, "frames": frames_out}
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        first = errors[0]
        loc = "/".join([str(p) for p in first.path]) or "<root>"
        raise ValueError(f"Schema validation failed for {clip_id} at {loc}: {first.message}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}_detections.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "clip_id": clip_id,
        "video_path": str(video_path),
        "frames_written": len(frames_out),
        "detections_total": int(sum(len(f["detections"]) for f in frames_out)),
        "tier_counts": {"1": tier_counts[1], "2": tier_counts[2], "3": tier_counts[3]},
        "geometry_failures": int(geometry_failures),
        "emission_positive_violations": int(emission_violations),
        "bev_source": {
            "clip_bev_found": bool(bev_map is not None),
            "frames_from_bev": int(real_bev_frames),
            "frames_from_stub": int(stub_bev_frames),
        },
        "keypoint_source": {
            "native_pose": int(keypoint_native_count),
            "bootstrap_from_bbox": int(keypoint_bootstrap_count),
        },
        "jitter_samples": {
            "tier2": len(tier2_disp),
            "tier3": len(tier3_disp),
            "tier2_counterfactual_tier3": len(tier2_counterfactual_tier3_disp),
        },
        "jitter_std": {
            "tier2": float(statistics.pstdev(tier2_disp)) if len(tier2_disp) >= 2 else None,
            "tier3": float(statistics.pstdev(tier3_disp)) if len(tier3_disp) >= 2 else None,
            "tier2_counterfactual_tier3": (
                float(statistics.pstdev(tier2_counterfactual_tier3_disp))
                if len(tier2_counterfactual_tier3_disp) >= 2
                else None
            ),
        },
        "json_path": str(out_path),
    }
    return summary


def write_ablation_report(report_path: Path, summaries: List[Dict[str, object]]) -> None:
    tier2 = []
    tier3 = []
    tier2_cf_tier3 = []
    total_geom_fail = 0
    total_det = 0
    total_tier = {1: 0, 2: 0, 3: 0}
    used_stub = 0

    for row in summaries:
        js = row.get("jitter_std", {})
        if js.get("tier2") is not None:
            tier2.append(float(js["tier2"]))
        if js.get("tier3") is not None:
            tier3.append(float(js["tier3"]))
        if js.get("tier2_counterfactual_tier3") is not None:
            tier2_cf_tier3.append(float(js["tier2_counterfactual_tier3"]))
        total_geom_fail += int(row.get("geometry_failures", 0))
        total_det += int(row.get("detections_total", 0))
        tier_counts = row.get("tier_counts", {})
        total_tier[1] += int(tier_counts.get("1", 0))
        total_tier[2] += int(tier_counts.get("2", 0))
        total_tier[3] += int(tier_counts.get("3", 0))
        bev_source = row.get("bev_source", {})
        if int(bev_source.get("frames_from_stub", 0)) > 0:
            used_stub += 1

    agg_tier2 = median(tier2)
    agg_tier3 = median(tier3)
    agg_cf = median(tier2_cf_tier3)

    criterion_met = False
    criterion_text = "Insufficient samples for Tier-2 vs Tier-3 jitter comparison."
    if tier2_cf_tier3 and tier2:
        criterion_met = agg_tier2 < agg_cf
        criterion_text = (
            f"Tier-2 median jitter std ({agg_tier2:.3f}) "
            f"{'<' if criterion_met else '>='} "
            f"Tier-3 counterfactual median jitter std ({agg_cf:.3f})."
        )

    lines = [
        f"# Fallback Ablation Report ({datetime.now().strftime('%Y-%m-%d')})",
        "",
        "## Scope",
        "- Compares Tier-1/2/3 base resolution usage and temporal jitter proxies.",
        "- Validates geometry check flags and emission log-probability constraints.",
        "- Uses BEV from Track C when present; otherwise uses deterministic local BEV stub.",
        "",
        "## Aggregate",
        f"- Clips processed: {len(summaries)}",
        f"- Detections total: {total_det}",
        f"- Tier 1 count: {total_tier[1]}",
        f"- Tier 2 count: {total_tier[2]}",
        f"- Tier 3 count: {total_tier[3]}",
        f"- Geometry check failures: {total_geom_fail}",
        f"- Clips with stub BEV usage: {used_stub}",
        "",
        "## Jitter Comparison",
        f"- Tier-2 jitter std median: {agg_tier2:.4f}" if tier2 else "- Tier-2 jitter std median: N/A",
        f"- Tier-3 jitter std median: {agg_tier3:.4f}" if tier3 else "- Tier-3 jitter std median: N/A",
        (
            f"- Tier-3 counterfactual on Tier-2 events (median std): {agg_cf:.4f}"
            if tier2_cf_tier3
            else "- Tier-3 counterfactual on Tier-2 events: N/A"
        ),
        f"- Pass criterion (Tier-2 lower jitter than Tier-3 baseline): {'PASS' if criterion_met else 'CHECK'}",
        f"- Detail: {criterion_text}",
        "",
        "## Per-clip Summary",
    ]

    for row in summaries:
        lines.extend(
            [
                f"### {row['clip_id']}",
                f"- Frames written: {row['frames_written']}",
                f"- Detections: {row['detections_total']}",
                f"- Tier counts: {row['tier_counts']}",
                f"- Geometry failures: {row['geometry_failures']}",
                f"- Jitter std: {row['jitter_std']}",
                f"- Output JSON: `{row['json_path']}`",
                "",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-frame detections JSONs for Wave 2.")
    parser.add_argument("--model", required=True, help="Pose model checkpoint path.")
    parser.add_argument(
        "--videos",
        default="/Users/quan/Documents/personal/Stanford application project/tracks/E_evaluation_ci/regression_videos",
        help="Video file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for <clip_id>_detections.json files.",
    )
    parser.add_argument(
        "--bev-dir",
        default="/Users/quan/Documents/personal/Stanford application project/tracks/C_bev_egomotion/outputs",
        help="Directory containing <clip_id>_bev.json files (optional).",
    )
    parser.add_argument(
        "--schema",
        default="/Users/quan/Documents/personal/Stanford application project/shared/interfaces/per_frame_detections.schema.json",
        help="Per-frame detections schema path.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.55, help="NMS IoU threshold.")
    parser.add_argument("--tau-kp", type=float, default=0.5, help="Keypoint confidence threshold.")
    parser.add_argument("--max-frames", type=int, default=180, help="Max processed frames per clip (0 = all).")
    parser.add_argument("--frame-stride", type=int, default=2, help="Process every Nth frame.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument(
        "--ablation-report",
        default="",
        help="Output markdown report path. Default: reports/ablation_fallback_YYYYMMDD.md",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional output path for run summary JSON.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    model_path = (cwd / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    videos_path = (cwd / args.videos).resolve() if not Path(args.videos).is_absolute() else Path(args.videos)
    out_dir = (cwd / args.output_dir).resolve()
    schema_path = (cwd / args.schema).resolve() if not Path(args.schema).is_absolute() else Path(args.schema)

    bev_dir = (cwd / args.bev_dir).resolve() if args.bev_dir else None
    if bev_dir is not None and not bev_dir.exists():
        bev_dir = None

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    model = YOLO(str(model_path))
    videos = list_videos(videos_path)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {videos_path}")

    summaries: List[Dict[str, object]] = []
    for video in videos:
        summary = process_clip(
            model=model,
            video_path=video,
            out_dir=out_dir,
            bev_dir=bev_dir,
            validator=validator,
            conf_thr=float(args.conf),
            iou_thr=float(args.iou),
            tau_kp=float(args.tau_kp),
            max_frames=int(args.max_frames),
            frame_stride=int(args.frame_stride),
            device=args.device,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if args.ablation_report:
        report_path = (cwd / args.ablation_report).resolve()
    else:
        report_path = (cwd / f"reports/ablation_fallback_{datetime.now().strftime('%Y%m%d')}.md").resolve()
    write_ablation_report(report_path, summaries)
    print(f"Wrote ablation report: {report_path}")

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": str(model_path),
        "videos": [str(v) for v in videos],
        "output_dir": str(out_dir),
        "schema": str(schema_path),
        "summaries": summaries,
        "ablation_report": str(report_path),
    }
    if args.summary_json:
        summary_path = (cwd / args.summary_json).resolve()
    else:
        summary_path = (cwd / f"reports/per_frame_detection_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").resolve()
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"Wrote run summary: {summary_path}")


if __name__ == "__main__":
    main()
