#!/usr/bin/env python3
"""
Create a YOLOv8-Pose dataset (2 keypoints) from an existing bbox-only YOLO dataset.

This is a bootstrap tool for Wave 2:
- Converts each bbox label line to pose format with 2 keypoints (base, tip).
- Prioritizes top hard-case images in a manifest for manual refinement.
- Optionally marks base keypoint as invisible on selected hard cases to model occlusion.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass(frozen=True)
class LabelSample:
    split: str
    image_path: Path
    label_path: Path
    image_name: str
    label_name: str
    object_count: int
    median_h_norm: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_bbox_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        return None
    return cls_id, cx, cy, w, h


def find_image_for_stem(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_label_lines(label_path: Path) -> List[str]:
    if not label_path.exists():
        return []
    out: List[str] = []
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if parse_bbox_line(line) is None:
            continue
        out.append(line)
    return out


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
    return sorted_vals[mid]


def iter_samples(dataset_root: Path) -> Iterable[LabelSample]:
    for split in ("train", "valid", "test"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            lines = read_label_lines(label_path)
            image_path = find_image_for_stem(images_dir, label_path.stem)
            if image_path is None:
                continue
            heights = [parse_bbox_line(line)[4] for line in lines if parse_bbox_line(line) is not None]
            yield LabelSample(
                split=split,
                image_path=image_path,
                label_path=label_path,
                image_name=image_path.name,
                label_name=label_path.name,
                object_count=len(lines),
                median_h_norm=median(heights),
            )


def load_hard_case_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_priority_image_names(samples: List[LabelSample], hard_case_rows: List[Dict[str, str]], top_k: int) -> List[str]:
    all_names = {sample.image_name for sample in samples}
    hard_case_names: List[str] = []
    for row in hard_case_rows:
        name = row.get("image", "")
        if name in all_names:
            hard_case_names.append(name)

    selected: List[str] = []
    seen = set()
    for name in hard_case_names:
        if name not in seen:
            selected.append(name)
            seen.add(name)
        if len(selected) >= top_k:
            return selected

    # Fill with smallest-gate samples from train/valid if hard cases are < top_k.
    small_candidates = sorted(
        [sample for sample in samples if sample.split in {"train", "valid"} and sample.object_count > 0],
        key=lambda s: (s.median_h_norm, -s.object_count),
    )
    for sample in small_candidates:
        if sample.image_name in seen:
            continue
        selected.append(sample.image_name)
        seen.add(sample.image_name)
        if len(selected) >= top_k:
            break
    return selected


def build_occluded_base_set(hard_case_rows: List[Dict[str, str]], max_images: int) -> set[str]:
    rows = sorted(
        hard_case_rows,
        key=lambda r: int(r.get("fn", "0")),
        reverse=True,
    )
    selected: set[str] = set()
    for row in rows:
        fn = int(row.get("fn", "0"))
        if fn < 2:
            continue
        name = row.get("image")
        if not name:
            continue
        selected.add(name)
        if len(selected) >= max_images:
            break
    return selected


def to_pose_label_line(
    cls_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    base_invisible: bool,
) -> str:
    # Keypoint 0 (base): bottom-center of bbox in normalized coords.
    # Keypoint 1 (tip): top-center of bbox in normalized coords.
    kp0_x = clamp01(cx)
    kp0_y = clamp01(cy + 0.5 * h)
    kp1_x = clamp01(cx)
    kp1_y = clamp01(cy - 0.5 * h)

    if base_invisible:
        return (
            f"{cls_id} {clamp01(cx):.6f} {clamp01(cy):.6f} {clamp01(w):.6f} {clamp01(h):.6f} "
            f"0 0 0 {kp1_x:.6f} {kp1_y:.6f} 2"
        )

    return (
        f"{cls_id} {clamp01(cx):.6f} {clamp01(cy):.6f} {clamp01(w):.6f} {clamp01(h):.6f} "
        f"{kp0_x:.6f} {kp0_y:.6f} 2 {kp1_x:.6f} {kp1_y:.6f} 2"
    )


def write_pose_yaml(output_root: Path) -> None:
    payload = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "nc: 1\n"
        "names: ['gate']\n"
        "kpt_shape: [2, 3]\n"
        "flip_idx: [0, 1]\n"
    )
    (output_root / "data.yaml").write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Wave 2 YOLOv8-Pose dataset from bbox labels.")
    parser.add_argument(
        "--source",
        default="artifacts/final_combined_1class_20260215",
        help="Source dataset root with train/valid/test images+labels.",
    )
    parser.add_argument(
        "--hard-cases-csv",
        default="reports/error_cases_20260214/hard_cases_top20.csv",
        help="Hard-case ranking CSV to prioritize top images.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output pose dataset root. Default: artifacts/pose_1class_YYYYMMDD",
    )
    parser.add_argument(
        "--priority-k",
        type=int,
        default=50,
        help="Number of priority images to list in annotation manifest.",
    )
    parser.add_argument(
        "--occluded-hard-cases",
        type=int,
        default=12,
        help="Number of hard-case images with bootstrapped base invisibility (kp0=0,0,0).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    args = parser.parse_args()

    cwd = Path.cwd()
    source = (cwd / args.source).resolve()
    if args.output:
        output_root = (cwd / args.output).resolve()
    else:
        output_root = (cwd / f"artifacts/pose_1class_{datetime.now().strftime('%Y%m%d')}").resolve()
    hard_csv = (cwd / args.hard_cases_csv).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root} (use --overwrite)")
        shutil.rmtree(output_root)

    for split in ("train", "valid", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    samples = list(iter_samples(source))
    hard_rows = load_hard_case_rows(hard_csv)
    priority_names = pick_priority_image_names(samples, hard_rows, top_k=max(1, args.priority_k))
    priority_set = set(priority_names)
    occluded_set = build_occluded_base_set(hard_rows, max_images=max(0, args.occluded_hard_cases))

    manifest_rows: List[Dict[str, object]] = []
    counts_by_split = {"train": 0, "valid": 0, "test": 0}
    objects_by_split = {"train": 0, "valid": 0, "test": 0}
    base_invisible_count = 0

    for sample in samples:
        out_image = output_root / sample.split / "images" / sample.image_path.name
        out_label = output_root / sample.split / "labels" / sample.label_path.name
        shutil.copy2(sample.image_path, out_image)

        bbox_lines = read_label_lines(sample.label_path)
        pose_lines: List[str] = []
        base_invisible = sample.image_name in occluded_set
        for line in bbox_lines:
            parsed = parse_bbox_line(line)
            if parsed is None:
                continue
            cls_id, cx, cy, w, h = parsed
            pose_lines.append(
                to_pose_label_line(
                    cls_id=cls_id,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    base_invisible=base_invisible,
                )
            )

        out_label.write_text("\n".join(pose_lines), encoding="utf-8")
        counts_by_split[sample.split] += 1
        objects_by_split[sample.split] += len(pose_lines)
        if base_invisible and pose_lines:
            base_invisible_count += 1

        manifest_rows.append(
            {
                "split": sample.split,
                "image_name": sample.image_name,
                "label_name": sample.label_name,
                "objects": len(pose_lines),
                "median_h_norm": round(sample.median_h_norm, 6),
                "priority_for_manual_review": sample.image_name in priority_set,
                "bootstrap_base_invisible": base_invisible,
            }
        )

    write_pose_yaml(output_root)

    manifest_csv = output_root / "annotation_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image_name",
                "label_name",
                "objects",
                "median_h_norm",
                "priority_for_manual_review",
                "bootstrap_base_invisible",
            ],
        )
        writer.writeheader()
        for row in sorted(
            manifest_rows,
            key=lambda r: (
                0 if r["priority_for_manual_review"] else 1,
                r["split"],
                r["image_name"],
            ),
        ):
            writer.writerow(row)

    summary = {
        "source_dataset": str(source),
        "output_dataset": str(output_root),
        "hard_cases_csv": str(hard_csv) if hard_csv.exists() else None,
        "counts_by_split": counts_by_split,
        "objects_by_split": objects_by_split,
        "total_images": sum(counts_by_split.values()),
        "total_objects": sum(objects_by_split.values()),
        "priority_images_top_k": len(priority_names),
        "base_invisible_bootstrap_images": base_invisible_count,
        "priority_image_names": priority_names,
        "notes": [
            "Keypoints are bootstrapped from bbox geometry for initial pose training.",
            "Priority images should be manually refined for true base/tip placement where possible.",
        ],
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Annotation manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
