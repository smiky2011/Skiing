"""
Prepare a filtered dataset for training.

Features:
- Drop empty-label images (background-only)
- Optionally merge classes into a single "gate" class

Usage:
    python scripts/prepare_dataset.py --src data/annotations/ski-gate-detection.v2i.yolov8 --dst data/annotations/ski-gate-detection.v2i.yolov8.filtered
    python scripts/prepare_dataset.py --src ... --dst ... --merge-classes
"""
import argparse
import shutil
from pathlib import Path


def _find_image_for_label(label_path, images_dir):
    for ext in (".jpg", ".png", ".jpeg"):
        img = images_dir / f"{label_path.stem}{ext}"
        if img.exists():
            return img
    return None


def prepare_dataset(src_dir, dst_dir, drop_empty=True, merge_classes=False):
    src = Path(src_dir)
    dst = Path(dst_dir)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    dropped_empty = 0

    for split in ["train", "valid", "test"]:
        src_images = src / split / "images"
        src_labels = src / split / "labels"
        dst_images = dst / split / "images"
        dst_labels = dst / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for label_path in src_labels.glob("*.txt"):
            total += 1
            label_text = label_path.read_text().strip()
            if drop_empty and label_text == "":
                dropped_empty += 1
                continue

            img_path = _find_image_for_label(label_path, src_images)
            if img_path is None:
                continue

            shutil.copy2(img_path, dst_images / img_path.name)

            if merge_classes and label_text:
                new_lines = []
                for line in label_text.splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    parts[0] = "0"
                    new_lines.append(" ".join(parts))
                (dst_labels / label_path.name).write_text("\n".join(new_lines))
            else:
                shutil.copy2(label_path, dst_labels / label_path.name)

            kept += 1

    # Write data.yaml with paths relative to dataset root
    names = ["gate"] if merge_classes else ["blue_gate", "red_gate"]
    nc = 1 if merge_classes else 2
    data_yaml = "\n".join([
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {nc}",
        f"names: {names}",
        "",
    ])
    (dst / "data.yaml").write_text(data_yaml)

    print(f"Total labels: {total}")
    print(f"Kept: {kept}")
    if drop_empty:
        print(f"Dropped empty-label images: {dropped_empty}")
    print(f"Prepared dataset at: {dst}")


def main():
    parser = argparse.ArgumentParser(description="Prepare filtered dataset")
    parser.add_argument("--src", required=True, help="Source dataset dir")
    parser.add_argument("--dst", required=True, help="Destination dataset dir")
    parser.add_argument("--keep-empty", action="store_true",
                        help="Keep empty-label images (backgrounds)")
    parser.add_argument("--merge-classes", action="store_true",
                        help="Merge red/blue gates into a single class")
    args = parser.parse_args()

    prepare_dataset(
        src_dir=args.src,
        dst_dir=args.dst,
        drop_empty=not args.keep_empty,
        merge_classes=args.merge_classes,
    )


if __name__ == "__main__":
    main()
