"""
Frame extraction from ski racing videos.
Supports uniform extraction and balanced extraction (for addressing class imbalance).
"""
import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from a video at a specified FPS rate.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        fps: Frames per second to extract (default: 1).

    Returns:
        Number of frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        count += 1

    cap.release()
    return saved


def extract_frames_balanced(video_path, output_dir, dense_seconds=10, dense_interval=10, sparse_interval=30):
    """
    Extract frames with balanced sampling to address class imbalance.

    Gates are most visible in the early portion of each run (course inspection,
    pre-start, and early gates). This function samples densely at the start
    and sparsely afterward.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        dense_seconds: How many seconds to sample densely at the start.
        dense_interval: Extract every Nth frame during dense period.
        sparse_interval: Extract every Nth frame during sparse period.

    Returns:
        Number of frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    dense_frame_limit = int(dense_seconds * video_fps)

    frame_count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Dense sampling early (more gate-heavy frames)
        if frame_count < dense_frame_limit:
            if frame_count % dense_interval == 0:
                cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
                saved += 1
        else:
            # Sparse sampling for the rest
            if frame_count % sparse_interval == 0:
                cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
                saved += 1

        frame_count += 1

    cap.release()
    return saved


def check_class_distribution(annotations_dir):
    """
    Check class distribution in annotation files to verify balance.
    Supports YOLO format (.txt files with class_id x y w h).

    Args:
        annotations_dir: Directory containing YOLO-format annotation files.

    Returns:
        Dictionary of class counts.
    """
    class_counts = {}

    for label_file in Path(annotations_dir).glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

    return class_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from ski racing videos")
    parser.add_argument("video_path", help="Path to video file or directory of videos")
    parser.add_argument("--output-dir", default="data/frames", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--balanced", action="store_true", help="Use balanced extraction (dense early, sparse late)")
    parser.add_argument("--dense-seconds", type=int, default=10, help="Seconds of dense sampling (balanced mode)")
    args = parser.parse_args()

    video_path = Path(args.video_path)

    if video_path.is_dir():
        videos = [
            p for p in video_path.rglob("*")
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        ]
    else:
        videos = [video_path]

    total_saved = 0
    for video in videos:
        if video_path.is_dir():
            rel = video.relative_to(video_path).with_suffix("")
            out_dir = Path(args.output_dir) / rel
        else:
            out_dir = Path(args.output_dir) / video.stem
        if args.balanced:
            count = extract_frames_balanced(str(video), str(out_dir), dense_seconds=args.dense_seconds)
        else:
            count = extract_frames(str(video), str(out_dir), fps=args.fps)
        print(f"Extracted {count} frames from {video.name}")
        total_saved += count

    print(f"\nTotal: {total_saved} frames extracted from {len(videos)} video(s)")
