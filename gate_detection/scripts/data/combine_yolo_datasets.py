"""
Combine one or more YOLO-format datasets into a single dataset.

Supported source layouts:
1) Split layout:
   <src>/train/images, <src>/train/labels
   <src>/valid/images, <src>/valid/labels (or val/)
   <src>/test/images,  <src>/test/labels
2) Flat layout:
   <src>/images, <src>/labels

Features:
- Keeps image/label pairs only
- Optional class merge (all class ids -> 0)
- Optional drop-empty behavior
- Exact image deduplication via SHA1 (prefers richer labels on conflicts)
- Writes YOLO data.yaml + merge report + manifest

Example:
    python scripts/combine_yolo_datasets.py \
      --src data/annotations/v7_curated \
      --src data/annotations/your_new_batch \
      --dst data/annotations/combined_v8_final \
      --overwrite \
      --merge-classes \
      --unsplit-target train
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
TARGET_SPLITS = ("train", "valid", "test")


@dataclass
class Sample:
    source_image: Path
    source_label: Path
    source_root: Path
    source_split: str
    label_lines: List[str]
    label_count: int
    sha1: str
    image_suffix: str


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80] or "img"


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    for ext in IMG_EXTS:
        img = images_dir / f"{label_path.stem}{ext}"
        if img.exists():
            return img
    return None


def _normalize_split_name(split: str) -> str:
    s = split.lower().strip()
    if s == "val":
        return "valid"
    if s in TARGET_SPLITS:
        return s
    return "unsplit"


def _parse_label_lines(label_path: Path, merge_classes: bool) -> Tuple[List[str], int]:
    text = label_path.read_text(encoding="utf-8", errors="replace")
    lines: List[str] = []
    invalid = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            invalid += 1
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            invalid += 1
            continue

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            invalid += 1
            continue
        if w <= 0.0 or h <= 0.0:
            invalid += 1
            continue
        x1, y1 = x - w / 2.0, y - h / 2.0
        x2, y2 = x + w / 2.0, y + h / 2.0
        if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
            invalid += 1
            continue

        if merge_classes:
            cls = 0
        parts[0] = str(cls)
        lines.append(" ".join(parts))

    return lines, invalid


def _iter_split_samples(src: Path, split: str) -> Iterable[Tuple[Path, Path, str]]:
    images_dir = src / split / "images"
    labels_dir = src / split / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _find_image_for_label(label_path, images_dir)
        if image_path is None:
            continue
        yield image_path, label_path, _normalize_split_name(split)


def _iter_flat_samples(src: Path) -> Iterable[Tuple[Path, Path, str]]:
    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _find_image_for_label(label_path, images_dir)
        if image_path is None:
            continue
        yield image_path, label_path, "unsplit"


def _collect_samples(
    sources: List[Path],
    merge_classes: bool,
    keep_empty: bool,
) -> Tuple[List[Sample], Dict[str, int]]:
    raw_samples: List[Sample] = []
    stats = {
        "source_count": len(sources),
        "pairs_seen": 0,
        "pairs_kept_before_dedup": 0,
        "dropped_empty": 0,
        "dropped_invalid_labels": 0,
        "dropped_unreadable_images": 0,
    }

    for src in sources:
        has_split = any((src / split / "images").exists() and (src / split / "labels").exists()
                        for split in ("train", "valid", "val", "test"))

        iterator: Iterable[Tuple[Path, Path, str]]
        if has_split:
            iterators = [
                _iter_split_samples(src, "train"),
                _iter_split_samples(src, "valid"),
                _iter_split_samples(src, "val"),
                _iter_split_samples(src, "test"),
            ]
            combined: List[Tuple[Path, Path, str]] = []
            for it in iterators:
                if it is not None:
                    combined.extend(list(it))
            iterator = combined
        else:
            iterator = list(_iter_flat_samples(src) or [])

        for image_path, label_path, split in iterator:
            stats["pairs_seen"] += 1
            label_lines, invalid = _parse_label_lines(label_path, merge_classes=merge_classes)
            if invalid > 0 and not label_lines:
                stats["dropped_invalid_labels"] += 1
                continue
            if not keep_empty and len(label_lines) == 0:
                stats["dropped_empty"] += 1
                continue

            try:
                digest = _sha1_file(image_path)
            except OSError:
                stats["dropped_unreadable_images"] += 1
                continue

            raw_samples.append(
                Sample(
                    source_image=image_path,
                    source_label=label_path,
                    source_root=src,
                    source_split=split,
                    label_lines=label_lines,
                    label_count=len(label_lines),
                    sha1=digest,
                    image_suffix=image_path.suffix.lower(),
                )
            )
            stats["pairs_kept_before_dedup"] += 1

    return raw_samples, stats


def _dedupe_samples(samples: List[Sample]) -> Tuple[List[Sample], Dict[str, int]]:
    best_by_sha: Dict[str, Sample] = {}
    replaced_for_richer_labels = 0
    duplicate_sha_dropped = 0

    for sample in samples:
        existing = best_by_sha.get(sample.sha1)
        if existing is None:
            best_by_sha[sample.sha1] = sample
            continue

        # Keep richer annotation if the same image appears in multiple sources.
        if sample.label_count > existing.label_count:
            best_by_sha[sample.sha1] = sample
            replaced_for_richer_labels += 1
        else:
            duplicate_sha_dropped += 1

    stats = {
        "unique_images_after_dedup": len(best_by_sha),
        "duplicate_sha_dropped": duplicate_sha_dropped,
        "replaced_for_richer_labels": replaced_for_richer_labels,
    }
    return list(best_by_sha.values()), stats


def _write_outputs(
    samples: List[Sample],
    dst: Path,
    unsplit_target: str,
    merge_classes: bool,
) -> Dict[str, object]:
    split_dirs = {
        split: {
            "images": dst / split / "images",
            "labels": dst / split / "labels",
        }
        for split in TARGET_SPLITS
    }
    for split in TARGET_SPLITS:
        split_dirs[split]["images"].mkdir(parents=True, exist_ok=True)
        split_dirs[split]["labels"].mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    split_counts = {split: 0 for split in TARGET_SPLITS}
    box_counts = {split: 0 for split in TARGET_SPLITS}

    for idx, sample in enumerate(sorted(samples, key=lambda s: (str(s.source_root), str(s.source_image))), start=1):
        split = sample.source_split if sample.source_split in TARGET_SPLITS else unsplit_target
        stem = _safe_stem(sample.source_image.stem)
        out_stem = f"{idx:06d}_{stem}_{sample.sha1[:8]}"
        out_img = split_dirs[split]["images"] / f"{out_stem}{sample.image_suffix}"
        out_lbl = split_dirs[split]["labels"] / f"{out_stem}.txt"

        shutil.copy2(sample.source_image, out_img)
        out_lbl.write_text("\n".join(sample.label_lines), encoding="utf-8")

        split_counts[split] += 1
        box_counts[split] += sample.label_count
        manifest_rows.append(
            {
                "source_root": str(sample.source_root),
                "source_split": sample.source_split,
                "source_image": str(sample.source_image),
                "source_label": str(sample.source_label),
                "target_split": split,
                "target_image": str(out_img),
                "target_label": str(out_lbl),
                "sha1": sample.sha1,
                "box_count": str(sample.label_count),
            }
        )

    # data.yaml
    if merge_classes:
        names = ["gate"]
        nc = 1
    else:
        # Generic fallback when classes are not merged.
        names = ["class_0", "class_1"]
        nc = 2

    data_yaml = "\n".join(
        [
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            f"nc: {nc}",
            f"names: {names}",
            "",
        ]
    )
    (dst / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # manifest.csv
    manifest_path = dst / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source_root",
            "source_split",
            "source_image",
            "source_label",
            "target_split",
            "target_image",
            "target_label",
            "sha1",
            "box_count",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    return {
        "split_image_counts": split_counts,
        "split_box_counts": box_counts,
        "manifest_path": str(manifest_path),
        "data_yaml_path": str(dst / "data.yaml"),
    }


def combine_datasets(
    src_dirs: List[str],
    dst_dir: str,
    keep_empty: bool,
    merge_classes: bool,
    unsplit_target: str,
    overwrite: bool,
) -> Dict[str, object]:
    sources = [Path(p) for p in src_dirs]
    dst = Path(dst_dir)
    unsplit_target = _normalize_split_name(unsplit_target)
    if unsplit_target not in TARGET_SPLITS:
        raise ValueError(f"--unsplit-target must be one of {TARGET_SPLITS}, got: {unsplit_target}")

    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    raw_samples, collect_stats = _collect_samples(
        sources=sources,
        merge_classes=merge_classes,
        keep_empty=keep_empty,
    )
    deduped, dedupe_stats = _dedupe_samples(raw_samples)
    output_stats = _write_outputs(
        samples=deduped,
        dst=dst,
        unsplit_target=unsplit_target,
        merge_classes=merge_classes,
    )

    report = {
        "source_roots": [str(s) for s in sources],
        "destination": str(dst),
        "keep_empty": keep_empty,
        "merge_classes": merge_classes,
        "unsplit_target": unsplit_target,
        **collect_stats,
        **dedupe_stats,
        **output_stats,
    }
    (dst / "combine_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine YOLO datasets into one destination dataset")
    parser.add_argument("--src", action="append", required=True,
                        help="Source YOLO dataset directory (repeat --src for multiple datasets)")
    parser.add_argument("--dst", required=True, help="Destination directory for combined dataset")
    parser.add_argument("--keep-empty", action="store_true",
                        help="Keep empty-label images (background negatives)")
    parser.add_argument("--merge-classes", action="store_true",
                        help="Merge all classes into class id 0")
    parser.add_argument("--unsplit-target", default="train",
                        help="Where to put flat-source samples without split info (train|valid|test)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite destination if it exists")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    report = combine_datasets(
        src_dirs=args.src,
        dst_dir=args.dst,
        keep_empty=args.keep_empty,
        merge_classes=args.merge_classes,
        unsplit_target=args.unsplit_target,
        overwrite=args.overwrite,
    )

    print(json.dumps(report, indent=2))
    print(f"Combined dataset written to: {args.dst}")


if __name__ == "__main__":
    main()
