#!/usr/bin/env python3
"""
Metric harness for gate tracking outputs.

Inputs:
- Ground truth annotations (JSON per-frame format or YOLO labels split)
- Predictions (per_frame_detections schema JSON or compatible per-frame JSON)

Outputs:
- JSON report with IDF1, HOTA, ID switches, fragmentation, topological ordering error,
  missed/false rates, and temporal jitter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Observation:
    frame_idx: int
    obj_id: str
    x: float
    y: float
    bbox: Optional[Tuple[float, float, float, float]]
    class_label: str = "unknown"


def repo_root_from_script() -> Path:
    # tracks/A_eval_harness/scripts/run_metrics.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3]


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_flat_yaml(path: Path) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not value:
            parsed[key] = ""
            continue
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            parsed[key] = value[1:-1]
            continue
        lowered = value.lower()
        if lowered == "true":
            parsed[key] = True
            continue
        if lowered == "false":
            parsed[key] = False
            continue
        try:
            if "." in value:
                parsed[key] = float(value)
            else:
                parsed[key] = int(value)
            continue
        except ValueError:
            parsed[key] = value
    return parsed


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return parse_flat_yaml(path)


def resolve_dataset_split(data_path: Path) -> Tuple[Path, Path]:
    """
    Resolve YOLO-like split into images/labels directories.
    """
    if data_path.is_file() and data_path.suffix.lower() in {".yaml", ".yml"}:
        payload = load_yaml(data_path)
        test_ref = payload.get("test")
        if not test_ref:
            raise ValueError(f"No 'test' entry found in dataset YAML: {data_path}")
        images_dir = (data_path.parent / str(test_ref)).resolve()
        labels_dir = images_dir.parent / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir
        raise FileNotFoundError(f"Expected YOLO split dirs not found: {images_dir} and {labels_dir}")

    if data_path.is_dir():
        if (data_path / "images").is_dir() and (data_path / "labels").is_dir():
            return (data_path / "images").resolve(), (data_path / "labels").resolve()
        if (data_path / "test" / "images").is_dir() and (data_path / "test" / "labels").is_dir():
            return (data_path / "test" / "images").resolve(), (data_path / "test" / "labels").resolve()
        if data_path.name == "images":
            labels_dir = data_path.parent / "labels"
            if labels_dir.is_dir():
                return data_path.resolve(), labels_dir.resolve()

    raise FileNotFoundError(
        f"Unable to resolve annotation split from '{data_path}'. "
        "Expected data.yaml, split dir (images/labels), or dataset root."
    )


def yolo_xywhn_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    x_center_px = x_center * image_width
    y_center_px = y_center * image_height
    width_px = width * image_width
    height_px = height * image_height
    x1 = x_center_px - width_px / 2.0
    y1 = y_center_px - height_px / 2.0
    x2 = x_center_px + width_px / 2.0
    y2 = y_center_px + height_px / 2.0
    return x1, y1, x2, y2


def bbox_center_bottom(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def load_yolo_ground_truth(gt_path: Path) -> Tuple[str, Dict[int, List[Observation]]]:
    images_dir, labels_dir = resolve_dataset_split(gt_path)
    image_paths = list_images(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    frames: Dict[int, List[Observation]] = {}
    for frame_idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        image_height, image_width = frame.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        observations: List[Observation] = []
        if label_path.exists():
            for det_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, cx, cy, w, h = parts
                bbox = yolo_xywhn_to_xyxy(
                    float(cx),
                    float(cy),
                    float(w),
                    float(h),
                    image_width,
                    image_height,
                )
                x, y = bbox_center_bottom(bbox)
                observations.append(
                    Observation(
                        frame_idx=frame_idx,
                        obj_id=f"gt_{frame_idx}_{det_idx}",
                        x=x,
                        y=y,
                        bbox=bbox,
                        class_label=f"class_{class_id}",
                    )
                )
        frames[frame_idx] = observations

    clip_id = images_dir.parent.name
    return clip_id, frames


def extract_point_and_bbox(entry: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
    base = entry.get("base_px")
    if isinstance(base, dict) and "x_px" in base and "y_px" in base:
        point = (float(base["x_px"]), float(base["y_px"]))
    elif "center_x" in entry and "base_y" in entry:
        point = (float(entry["center_x"]), float(entry["base_y"]))
    elif "x" in entry and "y" in entry:
        point = (float(entry["x"]), float(entry["y"]))
    else:
        point = None

    raw_bbox = entry.get("bbox_xyxy") or entry.get("bbox")
    bbox: Optional[Tuple[float, float, float, float]] = None
    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        bbox = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
        if point is None:
            point = bbox_center_bottom(bbox)

    return point, bbox


def load_frames_json(payload: Dict[str, Any], mode: str) -> Tuple[str, Dict[int, List[Observation]]]:
    """
    Parse per-frame JSON.
    Supports per_frame_detections schema and lenient generic variants.
    """
    clip_id = str(payload.get("clip_id") or payload.get("video_id") or payload.get("video") or "unknown_clip")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("JSON input must contain a top-level 'frames' list")

    frames: Dict[int, List[Observation]] = defaultdict(list)
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            continue
        frame_idx = int(raw_frame.get("frame_idx", raw_frame.get("frame", 0)))
        raw_objects = raw_frame.get("detections")
        if raw_objects is None:
            raw_objects = raw_frame.get("objects")
        if raw_objects is None:
            raw_objects = raw_frame.get("gates")
        if not isinstance(raw_objects, list):
            continue

        for det_idx, det in enumerate(raw_objects):
            if not isinstance(det, dict):
                continue
            point, bbox = extract_point_and_bbox(det)
            if point is None:
                continue
            obj_id: str
            track_id = det.get("track_id")
            if track_id is not None:
                obj_id = str(track_id)
            elif det.get("detection_id") is not None:
                obj_id = str(det["detection_id"])
            elif det.get("id") is not None:
                obj_id = str(det["id"])
            elif det.get("gate_id") is not None:
                obj_id = str(det["gate_id"])
            else:
                prefix = "pred" if mode == "prediction" else "gt"
                obj_id = f"{prefix}_{frame_idx}_{det_idx}"
            class_label = str(det.get("class_label", det.get("class_name", "unknown")))
            frames[frame_idx].append(
                Observation(
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    x=float(point[0]),
                    y=float(point[1]),
                    bbox=bbox,
                    class_label=class_label,
                )
            )

    return clip_id, dict(frames)


def load_ground_truth(path: Path) -> Tuple[str, Dict[int, List[Observation]], str]:
    if path.is_dir():
        clip_id, frames = load_yolo_ground_truth(path)
        return clip_id, frames, "yolo_split"

    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        clip_id, frames = load_yolo_ground_truth(path)
        return clip_id, frames, "yolo_yaml"

    if path.is_file() and path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        clip_id, frames = load_frames_json(payload, mode="ground_truth")
        return clip_id, frames, "json_frames"

    raise ValueError(f"Unsupported ground-truth input: {path}")


def load_predictions(path: Path) -> Tuple[str, Dict[int, List[Observation]], str]:
    if path.is_file() and path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        clip_id, frames = load_frames_json(payload, mode="prediction")
        return clip_id, frames, "json_frames"
    raise ValueError(f"Unsupported prediction input: {path}")


def greedy_match_points(
    gt_obs: Sequence[Observation],
    pred_obs: Sequence[Observation],
    max_distance_px: float,
) -> List[Tuple[int, int, float]]:
    if not gt_obs or not pred_obs:
        return []

    candidates: List[Tuple[float, int, int]] = []
    for g_idx, gt in enumerate(gt_obs):
        for p_idx, pred in enumerate(pred_obs):
            dist = math.hypot(gt.x - pred.x, gt.y - pred.y)
            if dist <= max_distance_px:
                candidates.append((dist, g_idx, p_idx))
    candidates.sort(key=lambda item: item[0])

    used_gt = set()
    used_pred = set()
    matches: List[Tuple[int, int, float]] = []
    for dist, g_idx, p_idx in candidates:
        if g_idx in used_gt or p_idx in used_pred:
            continue
        used_gt.add(g_idx)
        used_pred.add(p_idx)
        matches.append((g_idx, p_idx, dist))
    return matches


def build_track_sequences(frames: Dict[int, List[Observation]]) -> Dict[str, List[Tuple[int, float, float]]]:
    tracks: DefaultDict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for frame_idx in sorted(frames.keys()):
        for obs in frames[frame_idx]:
            tracks[obs.obj_id].append((frame_idx, obs.x, obs.y))
    return dict(tracks)


def compute_jitter_std(positions: Sequence[Tuple[int, float, float]]) -> float:
    if len(positions) < 3:
        return 0.0
    ordered = sorted(positions, key=lambda item: item[0])
    step_norms = []
    for idx in range(1, len(ordered)):
        _, x0, y0 = ordered[idx - 1]
        _, x1, y1 = ordered[idx]
        step_norms.append(math.hypot(x1 - x0, y1 - y0))
    if len(step_norms) < 2:
        return 0.0
    return float(np.std(step_norms))


def evaluate_metrics(
    gt_frames: Dict[int, List[Observation]],
    pred_frames: Dict[int, List[Observation]],
    max_distance_px: float,
    static_motion_threshold_px: float,
) -> Dict[str, Any]:
    all_frame_indices = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

    gt_total = 0
    pred_total = 0
    matched_total = 0
    missed_total = 0
    false_total = 0

    # For ID and topology metrics.
    pair_counts: Counter = Counter()
    gt_assignments: DefaultDict[str, List[Tuple[int, Optional[str]]]] = defaultdict(list)
    topo_total_pairs = 0
    topo_inversions = 0

    for frame_idx in all_frame_indices:
        gt_obs = gt_frames.get(frame_idx, [])
        pred_obs = pred_frames.get(frame_idx, [])
        gt_total += len(gt_obs)
        pred_total += len(pred_obs)

        matches = greedy_match_points(gt_obs=gt_obs, pred_obs=pred_obs, max_distance_px=max_distance_px)
        matched_total += len(matches)
        missed_total += len(gt_obs) - len(matches)
        false_total += len(pred_obs) - len(matches)

        matched_gt_idxs = {g_idx for g_idx, _, _ in matches}
        for g_idx, gt in enumerate(gt_obs):
            if g_idx not in matched_gt_idxs:
                gt_assignments[gt.obj_id].append((frame_idx, None))

        matched_pairs = []
        for g_idx, p_idx, _ in matches:
            gt = gt_obs[g_idx]
            pred = pred_obs[p_idx]
            pair_counts[(gt.obj_id, pred.obj_id)] += 1
            gt_assignments[gt.obj_id].append((frame_idx, pred.obj_id))
            matched_pairs.append((gt, pred))

        # Topological ordering error: compare pairwise relative y-order of matched gates.
        for i in range(len(matched_pairs)):
            gt_i, pred_i = matched_pairs[i]
            for j in range(i + 1, len(matched_pairs)):
                gt_j, pred_j = matched_pairs[j]
                gt_sign = gt_i.y - gt_j.y
                pred_sign = pred_i.y - pred_j.y
                if abs(gt_sign) < 1e-9 or abs(pred_sign) < 1e-9:
                    continue
                topo_total_pairs += 1
                if gt_sign * pred_sign < 0:
                    topo_inversions += 1

    # ID switches + fragmentation from per-GT assignment timelines.
    id_switches = 0
    track_fragmentation = 0
    for _, timeline in gt_assignments.items():
        ordered = sorted(timeline, key=lambda item: item[0])
        prev_pred: Optional[str] = None
        prev_matched = False
        segments = 0
        for _, pred_id in ordered:
            if pred_id is None:
                prev_matched = False
                continue
            if not prev_matched:
                segments += 1
            elif prev_pred is not None and pred_id != prev_pred:
                id_switches += 1
            prev_pred = pred_id
            prev_matched = True
        track_fragmentation += max(0, segments - 1)

    # IDF1 approximation via greedy maximum-overlap GT<->Pred identity pairing.
    assigned_gt = set()
    assigned_pred = set()
    idtp = 0
    for (gt_id, pred_id), overlap in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True):
        if gt_id in assigned_gt or pred_id in assigned_pred:
            continue
        assigned_gt.add(gt_id)
        assigned_pred.add(pred_id)
        idtp += overlap
    idfn = gt_total - idtp
    idfp = pred_total - idtp
    idf1 = safe_div(2 * idtp, (2 * idtp) + idfp + idfn)

    # HOTA proxy (DetA * AssA geometric mean).
    det_a = safe_div(matched_total, matched_total + 0.5 * (false_total + missed_total))
    ass_a = max(0.0, 1.0 - safe_div(id_switches, max(1, matched_total)))
    hota = math.sqrt(max(0.0, det_a * ass_a))

    missed_gate_rate = safe_div(missed_total, gt_total)
    false_gate_rate = safe_div(false_total, pred_total)
    topological_ordering_error = safe_div(topo_inversions, topo_total_pairs)

    # Temporal jitter on static gates:
    gt_tracks = build_track_sequences(gt_frames)
    pred_tracks = build_track_sequences(pred_frames)
    per_track_jitter: List[Dict[str, Any]] = []
    for gt_id, positions in gt_tracks.items():
        if len(positions) < 3:
            continue
        motion_mean = np.mean(
            [math.hypot(positions[i][1] - positions[i - 1][1], positions[i][2] - positions[i - 1][2]) for i in range(1, len(positions))]
        )
        if motion_mean > static_motion_threshold_px:
            continue

        matched_preds = [
            (pred_id, count)
            for (candidate_gt, pred_id), count in pair_counts.items()
            if candidate_gt == gt_id
        ]
        if not matched_preds:
            continue
        best_pred_id = sorted(matched_preds, key=lambda item: item[1], reverse=True)[0][0]
        pred_positions = pred_tracks.get(best_pred_id, [])
        jitter_std = compute_jitter_std(pred_positions)
        per_track_jitter.append(
            {
                "gt_track_id": gt_id,
                "pred_track_id": best_pred_id,
                "jitter_std": float(jitter_std),
                "track_len": len(pred_positions),
            }
        )

    temporal_jitter_global = float(np.mean([item["jitter_std"] for item in per_track_jitter])) if per_track_jitter else 0.0

    return {
        "IDF1": float(idf1),
        "HOTA": float(hota),
        "id_switches": int(id_switches),
        "track_fragmentation": int(track_fragmentation),
        "topological_ordering_error": float(topological_ordering_error),
        "missed_gate_rate": float(missed_gate_rate),
        "false_gate_rate": float(false_gate_rate),
        "temporal_jitter": {
            "per_track": per_track_jitter,
            "global_mean": temporal_jitter_global,
        },
        "counts": {
            "frames": len(all_frame_indices),
            "gt_total": int(gt_total),
            "pred_total": int(pred_total),
            "matched_total": int(matched_total),
            "missed_total": int(missed_total),
            "false_total": int(false_total),
            "topology_pairs": int(topo_total_pairs),
            "topology_inversions": int(topo_inversions),
        },
        "hota_components": {
            "DetA": float(det_a),
            "AssA": float(ass_a),
        },
    }


def build_dummy_predictions_from_gt(
    gt_frames: Dict[int, List[Observation]],
    miss_rate: float = 0.20,
    false_rate: float = 0.15,
    seed: int = 17,
) -> Dict[int, List[Observation]]:
    """
    Dummy tracker: uses bbox centroids / base points and emits a new ID per detection.
    """
    random.seed(seed)
    np.random.seed(seed)

    pred_frames: Dict[int, List[Observation]] = {}
    false_counter = 0
    for frame_idx in sorted(gt_frames.keys()):
        preds: List[Observation] = []
        gt_obs = gt_frames.get(frame_idx, [])
        for det_idx, gt in enumerate(gt_obs):
            if random.random() < miss_rate:
                continue
            jitter_x = np.random.normal(0, 2.0)
            jitter_y = np.random.normal(0, 2.0)
            pred_x = gt.x + float(jitter_x)
            pred_y = gt.y + float(jitter_y)
            preds.append(
                Observation(
                    frame_idx=frame_idx,
                    obj_id=f"dummy_{frame_idx}_{det_idx}",
                    x=pred_x,
                    y=pred_y,
                    bbox=None,
                    class_label=gt.class_label,
                )
            )

        if random.random() < false_rate:
            false_counter += 1
            base_x = 20.0 + (frame_idx % 50) * 5.0
            base_y = 20.0 + (frame_idx % 40) * 4.0
            preds.append(
                Observation(
                    frame_idx=frame_idx,
                    obj_id=f"dummy_false_{false_counter}",
                    x=base_x,
                    y=base_y,
                    bbox=None,
                    class_label="unknown",
                )
            )

        pred_frames[frame_idx] = preds
    return pred_frames


def build_synthetic_ground_truth(num_frames: int = 90) -> Tuple[str, Dict[int, List[Observation]]]:
    """
    Synthetic GT for smoke tests when annotation files are unavailable.
    Includes stable gate IDs so ID-switch behavior is visible.
    """
    clip_id = "synthetic_eval_clip"
    tracks = [
        ("gate_A", 320.0, 250.0),
        ("gate_B", 520.0, 300.0),
        ("gate_C", 740.0, 360.0),
    ]
    frames: Dict[int, List[Observation]] = {}
    for frame_idx in range(num_frames):
        frame_obs: List[Observation] = []
        for gate_id, base_x, base_y in tracks:
            drift = math.sin(frame_idx / 18.0) * 0.8
            x = base_x + drift
            y = base_y + drift * 0.5
            frame_obs.append(
                Observation(
                    frame_idx=frame_idx,
                    obj_id=gate_id,
                    x=x,
                    y=y,
                    bbox=(x - 10.0, y - 40.0, x + 10.0, y),
                    class_label="gate",
                )
            )
        frames[frame_idx] = frame_obs
    return clip_id, frames


def to_json_frames(clip_id: str, frames: Dict[int, List[Observation]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"clip_id": clip_id, "frames": []}
    for frame_idx in sorted(frames.keys()):
        detections = []
        for det_idx, obs in enumerate(frames[frame_idx]):
            det = {
                "detection_id": f"{clip_id}_{frame_idx:05d}_{det_idx:03d}",
                "track_id": obs.obj_id,
                "base_px": {"x_px": obs.x, "y_px": obs.y},
                "class_label": obs.class_label,
            }
            if obs.bbox is not None:
                det["bbox_xyxy"] = [obs.bbox[0], obs.bbox[1], obs.bbox[2], obs.bbox[3]]
            detections.append(det)
        payload["frames"].append({"frame_idx": frame_idx, "detections": detections})
    return payload


def ensure_output_path(path: Optional[Path]) -> Path:
    if path is not None:
        return path.resolve()
    root = repo_root_from_script()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return root / "outputs" / "metrics" / f"metrics_{stamp}.json"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_path = ensure_output_path(args.output)

    if args.smoke_baseline:
        if args.ground_truth:
            gt_clip_id, gt_frames, gt_format = load_ground_truth(args.ground_truth.resolve())
        else:
            gt_clip_id, gt_frames = build_synthetic_ground_truth(num_frames=args.synthetic_frames)
            gt_format = "synthetic"
        pred_frames = build_dummy_predictions_from_gt(
            gt_frames=gt_frames,
            miss_rate=args.dummy_miss_rate,
            false_rate=args.dummy_false_rate,
            seed=args.seed,
        )
        pred_clip_id = f"{gt_clip_id}_dummy"
        pred_format = "dummy_tracker"
    else:
        if not args.ground_truth or not args.predictions:
            raise ValueError("--ground-truth and --predictions are required unless --smoke-baseline is set")
        gt_clip_id, gt_frames, gt_format = load_ground_truth(args.ground_truth.resolve())
        pred_clip_id, pred_frames, pred_format = load_predictions(args.predictions.resolve())

    metrics = evaluate_metrics(
        gt_frames=gt_frames,
        pred_frames=pred_frames,
        max_distance_px=args.max_distance_px,
        static_motion_threshold_px=args.static_motion_threshold_px,
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "ground_truth": {
            "clip_id": gt_clip_id,
            "format": gt_format,
            "path": str(args.ground_truth.resolve()) if args.ground_truth else None,
        },
        "predictions": {
            "clip_id": pred_clip_id,
            "format": pred_format,
            "path": str(args.predictions.resolve()) if args.predictions else None,
        },
        "settings": {
            "max_distance_px": args.max_distance_px,
            "static_motion_threshold_px": args.static_motion_threshold_px,
            "smoke_baseline": bool(args.smoke_baseline),
            "dummy_miss_rate": args.dummy_miss_rate if args.smoke_baseline else None,
            "dummy_false_rate": args.dummy_false_rate if args.smoke_baseline else None,
            "seed": args.seed if args.smoke_baseline else None,
        },
    }
    report.update(metrics)

    write_json(output_path, report)
    print(f"Wrote metrics report: {output_path}")

    if args.dump_dummy_predictions and args.smoke_baseline:
        dummy_payload = to_json_frames(clip_id=pred_clip_id, frames=pred_frames)
        write_json(args.dump_dummy_predictions.resolve(), dummy_payload)
        print(f"Wrote dummy predictions: {args.dump_dummy_predictions.resolve()}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Run tracking/detection metrics for gate tracker outputs.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help=(
            "Ground-truth input path. Supported: JSON per-frame file, YOLO split directory, "
            "or YOLO data.yaml."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Prediction JSON path (per_frame_detections schema or compatible).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output report path. Default: "
            "outputs/metrics/metrics_YYYYMMDD_HHMM.json"
        ),
    )
    parser.add_argument(
        "--max-distance-px",
        type=float,
        default=60.0,
        help="Max point distance threshold for GT<->prediction matching.",
    )
    parser.add_argument(
        "--static-motion-threshold-px",
        type=float,
        default=2.5,
        help="Mean motion threshold for classifying GT tracks as static (for jitter metric).",
    )
    parser.add_argument(
        "--smoke-baseline",
        action="store_true",
        help="Run baseline on a dummy tracker (new ID per detection).",
    )
    parser.add_argument(
        "--synthetic-frames",
        type=int,
        default=90,
        help="Number of frames for synthetic GT when --smoke-baseline and no --ground-truth are provided.",
    )
    parser.add_argument(
        "--dummy-miss-rate",
        type=float,
        default=0.20,
        help="Dummy tracker miss probability per GT detection.",
    )
    parser.add_argument(
        "--dummy-false-rate",
        type=float,
        default=0.15,
        help="Dummy tracker false-positive insertion probability per frame.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for smoke-baseline generation.",
    )
    parser.add_argument(
        "--dump-dummy-predictions",
        type=Path,
        default=None,
        help="Optional path to save generated dummy predictions JSON (smoke mode only).",
    )
    parser.add_argument(
        "--default-annotation-root",
        type=Path,
        default=root / "data" / "annotations",
        help="Documentation-only reference path; not used unless passed explicitly as --ground-truth.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = run(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
