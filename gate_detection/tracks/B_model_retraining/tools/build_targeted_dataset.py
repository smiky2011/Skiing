#!/usr/bin/env python3
"""
Build a targeted add-on YOLO dataset for data-centric retraining.

Outputs a flat YOLO layout:
  <output>/images
  <output>/labels

Selection strategy:
- Small/far gates: positive labels with median box height < threshold.
- Hard negatives: empty-label images ranked by model false-positive confidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class Candidate:
    kind: str  # "small_gate" | "hard_negative"
    source_root: Path
    split: str
    image_path: Path
    label_path: Path
    sha1: str
    label_lines: List[str]
    label_count: int
    median_height_px: float
    contrast_std: float
    blur_var: float
    vertical_edge_strength: float
    model_max_conf: float = 0.0
    score: float = 0.0


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_image_for_label(label_path: Path, images_dir: Path) -> Optional[Path]:
    for ext in IMG_EXTS:
        img = images_dir / f"{label_path.stem}{ext}"
        if img.exists():
            return img
    return None


def parse_label_lines(label_path: Path) -> List[str]:
    lines: List[str] = []
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
        except ValueError:
            continue
        lines.append(line)
    return lines


def yolo_heights_px(label_lines: Iterable[str], image_h: int) -> List[float]:
    heights = []
    for line in label_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        h_norm = float(parts[4])
        heights.append(h_norm * float(image_h))
    return heights


def compute_image_quality_metrics(image_path: Path) -> Tuple[float, float, float]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Strong x-gradient indicates vertical structures.
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    vertical_edge_strength = float(np.mean(np.abs(grad_x)))
    return contrast, blur, vertical_edge_strength


def iter_split_samples(source_root: Path) -> Iterable[Tuple[str, Path, Path]]:
    for split in ("train", "valid", "val", "test"):
        images_dir = source_root / split / "images"
        labels_dir = source_root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        split_name = "valid" if split == "val" else split
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image_for_label(label_path, images_dir)
            if image_path is None:
                continue
            yield split_name, image_path, label_path


def collect_base_sha1s(base_dataset: Path) -> set[str]:
    hashes: set[str] = set()
    for split in ("train", "valid", "test"):
        images_dir = base_dataset / split / "images"
        if not images_dir.exists():
            continue
        for image_path in images_dir.iterdir():
            if image_path.suffix.lower() not in IMG_EXTS:
                continue
            try:
                hashes.add(sha1_file(image_path))
            except OSError:
                continue
    return hashes


def dedupe_best_by_sha(candidates: List[Candidate]) -> List[Candidate]:
    best: Dict[str, Candidate] = {}
    for row in candidates:
        prev = best.get(row.sha1)
        if prev is None or row.score > prev.score:
            best[row.sha1] = row
    return list(best.values())


def rank_small_gate_candidate(
    median_height_px: float,
    contrast_std: float,
    blur_var: float,
    max_height_px: float,
    small_height_threshold: float,
) -> float:
    # Smaller median/max gate heights are preferred.
    size_score = max(0.0, (small_height_threshold - median_height_px) / max(small_height_threshold, 1.0))
    max_size_score = max(0.0, (small_height_threshold - max_height_px) / max(small_height_threshold, 1.0))
    # Lower contrast and lower Laplacian variance are harder conditions.
    contrast_score = max(0.0, (55.0 - contrast_std) / 55.0)
    blur_score = max(0.0, (220.0 - blur_var) / 220.0)
    return 1.6 * size_score + 0.8 * max_size_score + 0.7 * contrast_score + 0.7 * blur_score


def score_negatives_with_model(
    model: YOLO,
    candidates: List[Candidate],
    conf: float,
    iou: float,
    batch_size: int = 32,
) -> None:
    if not candidates:
        return

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        sources = [str(row.image_path) for row in batch]
        stream = model.predict(
            source=sources,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False,
            stream=True,
        )
        by_name: Dict[str, float] = {}
        for prediction in stream:
            image_name = Path(prediction.path).name
            max_conf = 0.0
            if prediction.boxes is not None and len(prediction.boxes) > 0:
                confs = prediction.boxes.conf.detach().cpu().numpy().tolist()
                max_conf = max(float(x) for x in confs)
            by_name[image_name] = max_conf

        for row in batch:
            row.model_max_conf = by_name.get(row.image_path.name, 0.0)


def rank_negative_candidate(model_max_conf: float, vertical_edge_strength: float) -> float:
    edge_norm = min(vertical_edge_strength / 35.0, 1.0)
    return 0.75 * model_max_conf + 0.25 * edge_norm


def ensure_clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)


def safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._-") or "img"


def write_dataset(output_dir: Path, selected: List[Candidate]) -> Dict[str, int]:
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    counts = {"small_gate": 0, "hard_negative": 0}

    for idx, row in enumerate(selected, start=1):
        prefix = "sg" if row.kind == "small_gate" else "hn"
        stem = safe_stem(row.image_path.stem)
        out_stem = f"{prefix}_{idx:04d}_{stem}_{row.sha1[:8]}"
        out_img = images_out / f"{out_stem}{row.image_path.suffix.lower()}"
        out_lbl = labels_out / f"{out_stem}.txt"

        shutil.copy2(row.image_path, out_img)
        out_lbl.write_text("\n".join(row.label_lines), encoding="utf-8")
        counts[row.kind] += 1

    return counts


def write_manifest(output_dir: Path, selected: List[Candidate]) -> Path:
    manifest_path = output_dir / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "score",
                "source_root",
                "split",
                "image_path",
                "label_path",
                "sha1",
                "label_count",
                "median_height_px",
                "contrast_std",
                "blur_var",
                "vertical_edge_strength",
                "model_max_conf",
            ],
        )
        writer.writeheader()
        for row in sorted(selected, key=lambda r: (r.kind, -r.score)):
            writer.writerow(
                {
                    "kind": row.kind,
                    "score": f"{row.score:.6f}",
                    "source_root": str(row.source_root),
                    "split": row.split,
                    "image_path": str(row.image_path),
                    "label_path": str(row.label_path),
                    "sha1": row.sha1,
                    "label_count": row.label_count,
                    "median_height_px": f"{row.median_height_px:.4f}",
                    "contrast_std": f"{row.contrast_std:.4f}",
                    "blur_var": f"{row.blur_var:.4f}",
                    "vertical_edge_strength": f"{row.vertical_edge_strength:.4f}",
                    "model_max_conf": f"{row.model_max_conf:.4f}",
                }
            )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build targeted small-gate + hard-negative dataset")
    parser.add_argument(
        "--base-dataset",
        required=True,
        help="Base dataset used for duplicate filtering (images already present are skipped).",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source dataset root (repeat --source for multiple datasets).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path (.pt) used to score hard negatives by false-positive confidence.",
    )
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--small-target", type=int, default=40, help="Number of small-gate images to select")
    parser.add_argument("--negative-target", type=int, default=20, help="Number of hard-negative images to select")
    parser.add_argument(
        "--small-height-threshold",
        type=float,
        default=120.0,
        help="Max median gate height (px) to consider as small/far.",
    )
    parser.add_argument("--negative-conf", type=float, default=0.20, help="Detection conf for hard-negative mining")
    parser.add_argument("--negative-iou", type=float, default=0.55, help="NMS IoU for hard-negative mining")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_dataset = Path(args.base_dataset).resolve()
    sources = [Path(p).resolve() for p in args.source]
    output_dir = Path(args.output).resolve()
    model_path = Path(args.model).resolve()

    for path in [base_dataset, model_path, *sources]:
        if not path.exists():
            raise FileNotFoundError(f"Missing path: {path}")

    base_hashes = collect_base_sha1s(base_dataset)

    small_candidates: List[Candidate] = []
    negative_candidates: List[Candidate] = []

    for source in sources:
        for split, image_path, label_path in iter_split_samples(source):
            label_lines = parse_label_lines(label_path)
            try:
                image_sha = sha1_file(image_path)
            except OSError:
                continue

            if image_sha in base_hashes:
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            image_h = frame.shape[0]
            heights = yolo_heights_px(label_lines, image_h)
            contrast_std, blur_var, vertical_edge_strength = compute_image_quality_metrics(image_path)

            if label_lines:
                median_h = float(np.median(heights)) if heights else 0.0
                max_h = float(np.max(heights)) if heights else 0.0
                if median_h >= args.small_height_threshold:
                    continue
                score = rank_small_gate_candidate(
                    median_height_px=median_h,
                    contrast_std=contrast_std,
                    blur_var=blur_var,
                    max_height_px=max_h,
                    small_height_threshold=float(args.small_height_threshold),
                )
                small_candidates.append(
                    Candidate(
                        kind="small_gate",
                        source_root=source,
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        sha1=image_sha,
                        label_lines=label_lines,
                        label_count=len(label_lines),
                        median_height_px=median_h,
                        contrast_std=contrast_std,
                        blur_var=blur_var,
                        vertical_edge_strength=vertical_edge_strength,
                        score=score,
                    )
                )
            else:
                negative_candidates.append(
                    Candidate(
                        kind="hard_negative",
                        source_root=source,
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        sha1=image_sha,
                        label_lines=[],
                        label_count=0,
                        median_height_px=0.0,
                        contrast_std=contrast_std,
                        blur_var=blur_var,
                        vertical_edge_strength=vertical_edge_strength,
                    )
                )

    small_candidates = dedupe_best_by_sha(small_candidates)
    negative_candidates = dedupe_best_by_sha(negative_candidates)

    model = YOLO(str(model_path))
    score_negatives_with_model(
        model=model,
        candidates=negative_candidates,
        conf=float(args.negative_conf),
        iou=float(args.negative_iou),
    )
    for row in negative_candidates:
        row.score = rank_negative_candidate(
            model_max_conf=row.model_max_conf,
            vertical_edge_strength=row.vertical_edge_strength,
        )

    selected_small = sorted(small_candidates, key=lambda r: r.score, reverse=True)[: args.small_target]
    selected_negatives = sorted(negative_candidates, key=lambda r: r.score, reverse=True)[: args.negative_target]
    selected = selected_small + selected_negatives

    ensure_clean_output(output_dir)
    counts = write_dataset(output_dir, selected)
    manifest_path = write_manifest(output_dir, selected)

    summary = {
        "base_dataset": str(base_dataset),
        "sources": [str(s) for s in sources],
        "model": str(model_path),
        "output": str(output_dir),
        "small_height_threshold": float(args.small_height_threshold),
        "targets": {
            "small_target": int(args.small_target),
            "negative_target": int(args.negative_target),
        },
        "candidate_pool": {
            "small_gate": len(small_candidates),
            "hard_negative": len(negative_candidates),
        },
        "selected_counts": counts,
        "manifest": str(manifest_path),
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Created targeted dataset at: {output_dir}")


if __name__ == "__main__":
    main()

