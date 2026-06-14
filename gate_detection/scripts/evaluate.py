"""
Evaluation utilities for gate detection.

Supports two modes:
1) Legacy single-file evaluation (`--predictions` + `--ground-truth`).
2) Holdout split evaluation for CI (`--model` + `--data` + `--output`).
"""
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_THRESHOLDS = (0.25, 0.35, 0.45, 0.55)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_prf(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return safe_div(inter_area, union_area)


def yolo_xywhn_to_xyxy(x_center, y_center, width, height, image_width, image_height):
    x_center_px = x_center * image_width
    y_center_px = y_center * image_height
    width_px = width * image_width
    height_px = height * image_height

    x1 = x_center_px - width_px / 2.0
    y1 = y_center_px - height_px / 2.0
    x2 = x_center_px + width_px / 2.0
    y2 = y_center_px + height_px / 2.0
    return [x1, y1, x2, y2]


def parse_thresholds(value):
    if not value:
        return list(DEFAULT_THRESHOLDS)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    thresholds = sorted({float(part) for part in parts})
    if not thresholds:
        raise ValueError("At least one confidence threshold is required")
    return thresholds


def parse_flat_yaml(path):
    """
    Tiny fallback YAML parser for simple flat key/value files.
    """
    parsed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()

        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            parsed[key.strip()] = value[1:-1]
            continue

        lowered = value.lower()
        if lowered == "true":
            parsed[key.strip()] = True
            continue
        if lowered == "false":
            parsed[key.strip()] = False
            continue

        try:
            if "." in value:
                parsed[key.strip()] = float(value)
            else:
                parsed[key.strip()] = int(value)
            continue
        except ValueError:
            parsed[key.strip()] = value
    return parsed


def load_yaml(path):
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return parse_flat_yaml(path)


def resolve_dataset_split(data_path):
    """
    Resolve the holdout split into images/labels directories.

    Supported values:
      - split directory with `images/` + `labels/`
      - dataset root with `test/images` + `test/labels`
      - YOLO data.yaml with `test: ...`
      - images directory (with sibling labels directory)
    """
    data_path = Path(data_path)

    if data_path.is_file() and data_path.suffix.lower() in {".yaml", ".yml"}:
        payload = load_yaml(data_path)
        test_ref = payload.get("test")
        if not test_ref:
            raise ValueError(f"No 'test' entry found in dataset YAML: {data_path}")
        images_dir = (data_path.parent / str(test_ref)).resolve()
        labels_dir = images_dir.parent / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir
        raise FileNotFoundError(
            f"Could not resolve split from YAML. Expected: {images_dir} and {labels_dir}"
        )

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
        f"Unable to resolve holdout split from '{data_path}'. "
        "Expected a split dir (images/labels), dataset root, or data.yaml."
    )


def list_images(images_dir):
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def read_ground_truth_boxes(label_path, image_width, image_height):
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, width, height = parts
        boxes.append(
            yolo_xywhn_to_xyxy(
                float(cx),
                float(cy),
                float(width),
                float(height),
                image_width,
                image_height,
            )
        )
    return boxes


def collect_predictions(model, image_paths, conf_min, nms_iou, imgsz=960):
    results = {}
    stream = model.predict(
        source=[str(path) for path in image_paths],
        conf=conf_min,
        iou=nms_iou,
        imgsz=imgsz,
        save=False,
        verbose=False,
        stream=True,
    )

    for prediction in stream:
        image_name = Path(prediction.path).name
        detections = []
        if prediction.boxes is not None:
            for box in prediction.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                detections.append(
                    {
                        "bbox": [float(value) for value in bbox],
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    }
                )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        results[image_name] = detections

    return results


def _ensemble_nms(dets_a, dets_b, iou_thresh=0.50):
    """Merge detections from two models and suppress duplicates."""
    merged = list(dets_a) + list(dets_b)
    merged.sort(key=lambda d: d["confidence"], reverse=True)
    keep = []
    for det in merged:
        suppress = False
        for kept in keep:
            if box_iou(det["bbox"], kept["bbox"]) > iou_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(det)
    return keep


def collect_ensemble_predictions(
    model_paths, image_paths, conf_min, nms_iou, ensemble_nms_iou=0.50, imgsz=960
):
    """Run multiple models and merge predictions with NMS."""
    per_model = []
    for model_path in model_paths:
        model = YOLO(str(model_path))
        preds = collect_predictions(model, image_paths, conf_min, nms_iou, imgsz=imgsz)
        per_model.append(preds)

    if len(per_model) == 1:
        return per_model[0]

    merged = {}
    all_names = set()
    for preds in per_model:
        all_names.update(preds.keys())
    for name in all_names:
        combined = []
        for preds in per_model:
            combined = _ensemble_nms(combined, preds.get(name, []), iou_thresh=ensemble_nms_iou)
        merged[name] = combined
    return merged


def greedy_match(predictions, ground_truth_boxes, iou_threshold=0.5):
    matched_gt = set()
    tp = 0
    fp = 0

    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        best_iou = 0.0
        best_idx = -1

        for idx, gt_box in enumerate(ground_truth_boxes):
            if idx in matched_gt:
                continue
            current_iou = box_iou(prediction["bbox"], gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth_boxes) - len(matched_gt)
    return tp, fp, fn


def run_holdout_evaluation(
    model_path,
    data_path,
    output_path=None,
    thresholds=None,
    match_iou=0.50,
    nms_iou=0.55,
    default_threshold=0.35,
    ensemble_model_paths=None,
    ensemble_nms_iou=0.50,
    imgsz=960,
):
    thresholds = sorted(set(thresholds or DEFAULT_THRESHOLDS))
    images_dir, labels_dir = resolve_dataset_split(data_path)
    image_paths = list_images(images_dir)

    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    all_model_paths = [str(model_path)]
    if ensemble_model_paths:
        all_model_paths.extend(str(p) for p in ensemble_model_paths)

    if len(all_model_paths) > 1:
        predictions_by_image = collect_ensemble_predictions(
            model_paths=all_model_paths,
            image_paths=image_paths,
            conf_min=min(thresholds),
            nms_iou=nms_iou,
            ensemble_nms_iou=ensemble_nms_iou,
            imgsz=imgsz,
        )
    else:
        model = YOLO(str(model_path))
        predictions_by_image = collect_predictions(
            model=model,
            image_paths=image_paths,
            conf_min=min(thresholds),
            nms_iou=nms_iou,
            imgsz=imgsz,
        )

    per_threshold = {}
    total_instances = 0

    for threshold in thresholds:
        threshold_key = f"{threshold:.2f}"
        tp = fp = fn = 0
        per_image = []

        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Failed to read image: {image_path}")

            image_height, image_width = frame.shape[:2]
            label_path = labels_dir / f"{image_path.stem}.txt"
            gt_boxes = read_ground_truth_boxes(label_path, image_width, image_height)
            predictions = [
                pred
                for pred in predictions_by_image.get(image_path.name, [])
                if pred["confidence"] >= threshold
            ]

            image_tp, image_fp, image_fn = greedy_match(predictions, gt_boxes, iou_threshold=match_iou)
            tp += image_tp
            fp += image_fp
            fn += image_fn

            if threshold == thresholds[0]:
                total_instances += len(gt_boxes)

            image_precision, image_recall, image_f1 = compute_prf(image_tp, image_fp, image_fn)
            per_image.append(
                {
                    "image": image_path.name,
                    "ground_truth": len(gt_boxes),
                    "predictions": len(predictions),
                    "tp": image_tp,
                    "fp": image_fp,
                    "fn": image_fn,
                    "precision": image_precision,
                    "recall": image_recall,
                    "f1": image_f1,
                }
            )

        precision, recall, f1 = compute_prf(tp, fp, fn)
        per_threshold[threshold_key] = {
            "threshold": float(threshold),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_image": per_image,
        }

    default_key = f"{float(default_threshold):.2f}"
    if default_key not in per_threshold:
        default_key = f"{thresholds[0]:.2f}"

    summary = per_threshold[default_key]
    result = {
        "mode": "holdout_iou",
        "model": str(model_path),
        "ensemble_models": [str(p) for p in all_model_paths] if len(all_model_paths) > 1 else None,
        "ensemble_nms_iou": float(ensemble_nms_iou) if len(all_model_paths) > 1 else None,
        "data": str(data_path),
        "resolved_split": {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
        },
        "images": len(image_paths),
        "instances": int(total_instances),
        "match_iou": float(match_iou),
        "nms_iou": float(nms_iou),
        "thresholds": [float(value) for value in thresholds],
        "default_threshold": float(summary["threshold"]),
        "summary": {
            "threshold": float(summary["threshold"]),
            "precision": float(summary["precision"]),
            "recall": float(summary["recall"]),
            "f1": float(summary["f1"]),
            "tp": int(summary["tp"]),
            "fp": int(summary["fp"]),
            "fn": int(summary["fn"]),
        },
        "per_threshold": per_threshold,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def evaluate_gate_detection(predicted_gates, ground_truth_gates, threshold=50):
    """
    Legacy point-distance gate evaluation for single JSON files.
    """
    true_positives = 0
    false_positives = 0
    matched = set()

    for pred in predicted_gates:
        px = pred.get("center_x", pred.get("x", 0))
        py = pred.get("base_y", pred.get("y", 0))

        min_dist = float("inf")
        closest_idx = -1

        for i, gt in enumerate(ground_truth_gates):
            if i in matched:
                continue
            gx = gt.get("x", gt.get("center_x", 0))
            gy = gt.get("y", gt.get("base_y", 0))
            dist = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        if min_dist < threshold:
            true_positives += 1
            matched.add(closest_idx)
        else:
            false_positives += 1

    false_negatives = len(ground_truth_gates) - true_positives
    precision, recall, f1 = compute_prf(true_positives, false_positives, false_negatives)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_tracking_coverage(trajectory, total_frames):
    tracked_frames = len(trajectory)
    coverage = tracked_frames / total_frames * 100 if total_frames > 0 else 0
    return {
        "tracked_frames": tracked_frames,
        "total_frames": total_frames,
        "coverage_percent": coverage,
    }


def run_legacy_evaluation(predictions_path, ground_truth_path, threshold=50):
    with open(predictions_path, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    with open(ground_truth_path, "r", encoding="utf-8") as handle:
        ground_truth = json.load(handle)

    gate_results = evaluate_gate_detection(
        predictions["gates"],
        ground_truth.get("gates", []),
        threshold=threshold,
    )

    total_frames = predictions.get("video_info", {}).get("total_frames", 0)
    coverage = evaluate_tracking_coverage(predictions.get("trajectory_2d", []), total_frames)
    return gate_results, coverage


def build_parser():
    parser = argparse.ArgumentParser(description="Gate detector evaluation utilities")

    parser.add_argument("--model", help="Path to YOLO model weights (.pt)")
    parser.add_argument("--data", help="Test split dir (or data.yaml) for holdout eval")
    parser.add_argument("--output", help="Output JSON path for holdout eval")
    parser.add_argument(
        "--thresholds",
        default="0.25,0.35,0.45,0.55",
        help="Comma-separated confidence thresholds for holdout eval",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.35,
        help="Primary threshold to summarize in holdout output",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="IoU threshold used for TP matching",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.55,
        help="NMS IoU threshold used during inference",
    )

    # Legacy mode arguments
    parser.add_argument("--predictions", help="Path to legacy analysis JSON")
    parser.add_argument("--ground-truth", help="Path to legacy ground truth JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=50,
        help="Legacy point-distance match threshold (pixels)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.model:
        if not args.data:
            raise ValueError("--data is required when using --model")
        if not args.output:
            raise ValueError("--output is required when using --model")

        results = run_holdout_evaluation(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            thresholds=parse_thresholds(args.thresholds),
            match_iou=args.match_iou,
            nms_iou=args.nms_iou,
            default_threshold=args.default_threshold,
        )

        print("\n=== Holdout Gate Detection Evaluation ===")
        print(f"Model:      {results['model']}")
        print(f"Split:      {results['resolved_split']['images_dir']}")
        print(f"Images:     {results['images']}")
        print(f"Instances:  {results['instances']}")
        print(f"Match IoU:  {results['match_iou']:.2f}")

        print("\nThreshold Breakdown")
        print("  conf   precision   recall   f1      tp   fp   fn")
        for threshold in results["thresholds"]:
            key = f"{threshold:.2f}"
            row = results["per_threshold"][key]
            print(
                f"  {threshold:>4.2f}   "
                f"{row['precision']:.4f}     {row['recall']:.4f}   {row['f1']:.4f}   "
                f"{row['tp']:>4d} {row['fp']:>4d} {row['fn']:>4d}"
            )

        print(f"\nSaved JSON: {args.output}")
        return

    if not args.predictions or not args.ground_truth:
        raise ValueError(
            "Legacy mode requires --predictions and --ground-truth "
            "(or use --model/--data/--output for holdout mode)."
        )

    gate_results, coverage = run_legacy_evaluation(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        threshold=args.threshold,
    )

    print("\n=== Gate Detection Evaluation (Legacy) ===")
    print(f"  Precision: {gate_results['precision']:.3f}")
    print(f"  Recall:    {gate_results['recall']:.3f}")
    print(f"  F1 Score:  {gate_results['f1']:.3f}")
    print(
        f"  TP: {gate_results['true_positives']}, "
        f"FP: {gate_results['false_positives']}, "
        f"FN: {gate_results['false_negatives']}"
    )

    print("\n=== Tracking Coverage ===")
    print(f"  Tracked: {coverage['tracked_frames']}/{coverage['total_frames']} frames")
    print(f"  Coverage: {coverage['coverage_percent']:.1f}%")


if __name__ == "__main__":
    main()
