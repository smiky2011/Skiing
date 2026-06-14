#!/usr/bin/env python3
"""
Extract per-frame PTS sidecars for raw videos.

Primary path: ffprobe packet timestamps.
Fallback path: OpenCV frame decode timestamps when ffprobe is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
from jsonschema import Draft7Validator


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
FAILURE_LABELS = {"occlusion", "rolling_shutter", "eis_jump", "snow_glare", "track_swap"}
READOUT_LUT_MS = {"1080p": 16.0, "4K": 33.0, "unknown": 33.0}


@dataclass
class VideoPTS:
    pts_seconds: List[float]
    fps_nominal: float
    width: int
    height: int
    source: str


def repo_root_from_script() -> Path:
    # tracks/A_eval_harness/scripts/extract_pts.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3]


def parse_ratio(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            num = float(left)
            den = float(right)
            return num / den if den else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def classify_resolution(width: int, height: int) -> str:
    max_dim = max(width, height)
    min_dim = min(width, height)
    if max_dim >= 3000 or min_dim >= 2000:
        return "4K"
    if max_dim >= 1600 and min_dim >= 900:
        return "1080p"
    return "unknown"


def make_clip_id(video_path: Path, input_dir: Path, stem_counts: Counter) -> str:
    stem = video_path.stem
    if stem_counts[stem] == 1:
        return stem

    rel_no_ext = video_path.relative_to(input_dir).with_suffix("")
    return "__".join(rel_no_ext.parts)


def discover_videos(input_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            candidates.append(path)
    return sorted(candidates)


def ffprobe_video(video_path: Path, ffprobe_bin: str = "ffprobe") -> Optional[VideoPTS]:
    cmd = [
        ffprobe_bin,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_packets",
        "-show_streams",
        "-select_streams",
        "v:0",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None

    if proc.returncode != 0:
        return None

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None

    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps_nominal = parse_ratio(stream.get("avg_frame_rate")) or parse_ratio(stream.get("r_frame_rate"))

    time_base = parse_ratio(stream.get("time_base")) or 0.0
    pts: List[float] = []
    for packet in payload.get("packets") or []:
        pts_time = packet.get("pts_time")
        if pts_time is not None:
            try:
                pts.append(float(pts_time))
                continue
            except (TypeError, ValueError):
                pass
        pts_raw = packet.get("pts")
        if pts_raw is not None and time_base > 0:
            try:
                pts.append(float(pts_raw) * time_base)
            except (TypeError, ValueError):
                continue

    if not pts:
        return None

    pts = enforce_monotonic_pts(pts, fps_nominal=fps_nominal)
    if fps_nominal <= 0:
        fps_nominal = estimate_fps_from_pts(pts)

    return VideoPTS(
        pts_seconds=pts,
        fps_nominal=float(fps_nominal),
        width=width,
        height=height,
        source="ffprobe",
    )


def enforce_monotonic_pts(pts: Sequence[float], fps_nominal: float) -> List[float]:
    if not pts:
        return []
    fallback_dt = 1.0 / fps_nominal if fps_nominal > 1e-9 else 1.0 / 30.0
    fixed = [float(pts[0])]
    for idx in range(1, len(pts)):
        current = float(pts[idx])
        prev = fixed[-1]
        if current <= prev:
            current = prev + fallback_dt
        fixed.append(current)
    return fixed


def estimate_fps_from_pts(pts: Sequence[float]) -> float:
    deltas = [pts[i] - pts[i - 1] for i in range(1, len(pts)) if pts[i] > pts[i - 1]]
    if not deltas:
        return 0.0
    median_dt = sorted(deltas)[len(deltas) // 2]
    if median_dt <= 1e-9:
        return 0.0
    return 1.0 / median_dt


def opencv_video(video_path: Path) -> VideoPTS:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video via OpenCV: {video_path}")

    fps_nominal = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fallback_dt = 1.0 / fps_nominal if fps_nominal > 1e-9 else 1.0 / 30.0

    pts: List[float] = []
    idx = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break

        msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if msec > 0:
            current = msec / 1000.0
        else:
            current = idx * fallback_dt

        if pts and current <= pts[-1]:
            current = pts[-1] + fallback_dt
        pts.append(current)
        idx += 1

    cap.release()

    if not pts:
        raise RuntimeError(f"No frames decoded from {video_path}")

    if fps_nominal <= 0:
        fps_nominal = estimate_fps_from_pts(pts) or 30.0

    return VideoPTS(
        pts_seconds=pts,
        fps_nominal=fps_nominal,
        width=width,
        height=height,
        source="opencv",
    )


def compute_deltas(pts: Sequence[float]) -> List[float]:
    if not pts:
        return []
    deltas = [0.0]
    for idx in range(1, len(pts)):
        dt = float(pts[idx]) - float(pts[idx - 1])
        deltas.append(max(0.0, dt))
    return deltas


def detect_vfr(deltas: Sequence[float], threshold_ratio: float = 1.05) -> bool:
    positive = [dt for dt in deltas[1:] if dt > 1e-9]
    if len(positive) < 2:
        return False
    mn = min(positive)
    mx = max(positive)
    if mn <= 0:
        return False
    return (mx / mn) > threshold_ratio


def load_schema(schema_path: Path) -> Draft7Validator:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft7Validator(payload)


def validate_sidecar(validator: Draft7Validator, sidecar: Dict[str, Any]) -> None:
    errors = sorted(validator.iter_errors(sidecar), key=lambda err: list(err.path))
    if not errors:
        return
    first = errors[0]
    location = ".".join(str(item) for item in first.path) or "<root>"
    raise ValueError(f"Schema validation failed at {location}: {first.message}")


def build_sidecar(
    clip_id: str,
    extracted: VideoPTS,
    failure_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    deltas = compute_deltas(extracted.pts_seconds)
    resolution = classify_resolution(extracted.width, extracted.height)
    fps_nominal = float(extracted.fps_nominal)
    slow_motion = fps_nominal >= 120.0
    readout_ms = READOUT_LUT_MS.get(resolution, READOUT_LUT_MS["unknown"])
    is_vfr = detect_vfr(deltas)

    frames = []
    for idx, (pts, dt) in enumerate(zip(extracted.pts_seconds, deltas)):
        frames.append(
            {
                "frame_idx": idx,
                "pts_seconds": float(pts),
                "delta_t_s": float(dt),
            }
        )

    sidecar: Dict[str, Any] = {
        "clip_id": clip_id,
        "resolution": resolution,
        "fps_nominal": fps_nominal,
        "is_vfr": bool(is_vfr),
        "readout_time_ms": float(readout_ms),
        "slow_motion": bool(slow_motion),
        "frames": frames,
    }

    if failure_labels:
        valid_labels = [label for label in failure_labels if label in FAILURE_LABELS]
        if valid_labels:
            sidecar["failure_labels"] = valid_labels
    return sidecar


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def parse_only_paths(only_values: Sequence[str], input_dir: Path) -> List[Path]:
    resolved: List[Path] = []
    for value in only_values:
        path = Path(value)
        candidate = path if path.is_absolute() else (input_dir / path)
        if candidate.exists() and candidate.is_file():
            resolved.append(candidate.resolve())
        else:
            raise FileNotFoundError(f"Requested --only path does not exist: {value}")
    return resolved


def read_failure_map(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--failure-map must be a JSON object: {clip_id: [labels...]}")
    parsed: Dict[str, List[str]] = {}
    for clip_id, labels in payload.items():
        if not isinstance(clip_id, str):
            continue
        if not isinstance(labels, list):
            continue
        clean = [str(label) for label in labels if str(label) in FAILURE_LABELS]
        parsed[clip_id] = clean
    return parsed


def process_video(
    video_path: Path,
    clip_id: str,
    validator: Draft7Validator,
    output_path: Path,
    ffprobe_bin: str,
    failure_labels: Optional[List[str]],
) -> Tuple[Dict[str, Any], str]:
    extracted = ffprobe_video(video_path=video_path, ffprobe_bin=ffprobe_bin)
    if extracted is None:
        extracted = opencv_video(video_path=video_path)
    sidecar = build_sidecar(clip_id=clip_id, extracted=extracted, failure_labels=failure_labels)
    validate_sidecar(validator, sidecar)
    write_json(output_path, sidecar)
    return sidecar, extracted.source


def format_status(
    clip_id: str,
    sidecar: Dict[str, Any],
    source: str,
    output_path: Path,
) -> str:
    frame_count = len(sidecar.get("frames") or [])
    fps = sidecar.get("fps_nominal")
    vfr = sidecar.get("is_vfr")
    slow = sidecar.get("slow_motion")
    return (
        f"[ok] clip_id={clip_id} frames={frame_count} fps_nominal={fps:.3f} "
        f"is_vfr={vfr} slow_motion={slow} source={source} -> {output_path}"
    )


def run(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    schema_path = args.schema.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    if args.only:
        videos = parse_only_paths(args.only, input_dir=input_dir)
    else:
        videos = discover_videos(input_dir=input_dir)

    if args.max_clips is not None and args.max_clips > 0:
        videos = videos[: args.max_clips]

    if not videos:
        print("No videos found.")
        return 0

    stem_counts = Counter(path.stem for path in videos)
    validator = load_schema(schema_path=schema_path)
    failure_map = read_failure_map(args.failure_map)

    processed = 0
    skipped_existing = 0
    slow_motion_count = 0
    vfr_count = 0
    source_counts: Counter = Counter()
    failed = 0

    for video_path in videos:
        clip_id = make_clip_id(video_path=video_path, input_dir=input_dir, stem_counts=stem_counts)
        output_path = output_dir / f"{clip_id}.json"

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"[skip] Existing sidecar: {output_path}")
            continue

        try:
            sidecar, source = process_video(
                video_path=video_path,
                clip_id=clip_id,
                validator=validator,
                output_path=output_path,
                ffprobe_bin=args.ffprobe_bin,
                failure_labels=failure_map.get(clip_id),
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[error] {video_path}: {exc}", file=sys.stderr)
            continue

        processed += 1
        source_counts[source] += 1
        if sidecar.get("slow_motion"):
            slow_motion_count += 1
        if sidecar.get("is_vfr"):
            vfr_count += 1
        print(format_status(clip_id=clip_id, sidecar=sidecar, source=source, output_path=output_path))

    print("")
    print("Extraction summary")
    print(f"- processed: {processed}")
    print(f"- skipped_existing: {skipped_existing}")
    print(f"- failed: {failed}")
    print(f"- slow_motion_clips: {slow_motion_count}")
    print(f"- vfr_clips: {vfr_count}")
    print(f"- source_counts: {dict(source_counts)}")

    if failed:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Extract sidecar PTS JSON for raw videos.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "data" / "raw_videos",
        help="Directory containing source videos (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "eval_sidecars,
        help="Directory where sidecar JSON files are written.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=root / "shared" / "interfaces" / "sidecar_pts.schema.json",
        help="Path to sidecar JSON schema.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help=(
            "Optional explicit list of video files to process. "
            "Each path may be absolute or relative to --input-dir."
        ),
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional hard cap on number of clips processed after filtering.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
        help="ffprobe binary name/path. Falls back to OpenCV when unavailable.",
    )
    parser.add_argument(
        "--failure-map",
        type=Path,
        default=None,
        help="Optional JSON map: {clip_id: [failure_label,...]}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sidecar files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = run(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
