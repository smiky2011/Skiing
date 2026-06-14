"""
Gate-only live detection test runner.

Generates overlay videos with live gate detections (cached between inference
frames) and writes per-video JSON summaries (counts + timing).

Example:
  python scripts/test_live_gate_detection.py \\
    "tests/re-test videos" \\
    --gate-model models/gate_detector_best.pt \\
    --stride 3 --conf 0.20 --iou 0.45 --infer-width 1280 \\
    --output-dir outputs/gate_live_retest
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path (so `import ski_racing` works when run as a script)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _percentile(values, q):
    if not values:
        return None
    arr = np.array(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _safe_mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _scale_dets(dets, sx, sy):
    scaled = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        out = dict(d)
        if "center_x" in out:
            out["center_x"] = float(out["center_x"]) * sx
        if "center_y" in out:
            out["center_y"] = float(out["center_y"]) * sy
        if "base_y" in out:
            out["base_y"] = float(out["base_y"]) * sy
        if "bbox" in out and isinstance(out["bbox"], (list, tuple)) and len(out["bbox"]) == 4:
            x0, y0, x1, y1 = out["bbox"]
            out["bbox"] = [float(x0) * sx, float(y0) * sy, float(x1) * sx, float(y1) * sy]
        scaled.append(out)
    return scaled


def run_one(
    video_path: Path,
    gate_model_path: Path,
    output_dir: Path,
    stride: int,
    conf: float,
    iou: float,
    infer_width: int | None,
    max_frames: int | None,
):
    from ski_racing.detection import GateDetector

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = float(total_frames / fps) if fps > 1e-6 else None

    out_video_path = output_dir / f"{video_path.stem}_live_gates.mp4"
    out_json_path = output_dir / f"{video_path.stem}_live_gates_summary.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps if fps > 1e-6 else 30.0, (width, height))

    detector = GateDetector(str(gate_model_path))

    stride = max(1, int(stride))
    infer_times_ms = []
    counts = []
    call_rows = []

    cached = []
    last_infer_frame = -10_000
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= int(max_frames):
            break

        if frame_idx - last_infer_frame >= stride:
            infer_frame = frame
            sx = sy = 1.0
            if infer_width is not None and int(infer_width) > 0 and width > int(infer_width):
                target_w = int(infer_width)
                target_h = int(round(height * (target_w / width)))
                infer_frame = cv2.resize(infer_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                sx = float(width / target_w)
                sy = float(height / target_h)

            t0 = time.perf_counter()
            dets = detector.detect_in_frame(infer_frame, conf=float(conf), iou=float(iou))
            t1 = time.perf_counter()
            infer_times_ms.append(1000.0 * (t1 - t0))
            if sx != 1.0 or sy != 1.0:
                dets = _scale_dets(dets, sx=sx, sy=sy)
            cached = dets
            last_infer_frame = frame_idx

            c = int(len(dets))
            counts.append(c)
            mean_conf = _safe_mean([float(d.get("confidence", 0.0)) for d in dets if isinstance(d, dict)])
            call_rows.append({"frame": int(frame_idx), "count": c, "mean_conf": float(mean_conf)})

        # Draw cached gates on every frame
        for i, d in enumerate(cached):
            if not isinstance(d, dict):
                continue
            try:
                gx = int(round(float(d.get("center_x", 0.0))))
                gy = int(round(float(d.get("base_y", 0.0))))
            except Exception:
                continue
            if gx <= 0 and gy <= 0:
                continue
            color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
            cv2.circle(frame, (gx, gy), 10, color, -1)
            cv2.circle(frame, (gx, gy), 15, color, 2)

        # Text overlay
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Gates (cached): {len(cached)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Infer stride: {stride}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if infer_times_ms:
            cv2.putText(frame, f"Infer avg: {(_safe_mean(infer_times_ms)):.1f} ms",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    processed_frames = int(frame_idx)
    infer_calls = int(len(infer_times_ms))
    wall_s = float(sum(infer_times_ms) / 1000.0)  # only model time (excludes I/O + encoding)
    # Note: "effective fps" here is a rough indicator; actual wall time depends on video I/O + encoding.
    effective_fps = float(infer_calls / wall_s) if wall_s > 1e-9 else None
    realtime_x = None
    if duration_s is not None and wall_s > 1e-9:
        realtime_x = float(duration_s / wall_s)

    summary = {
        "video": str(video_path),
        "video_info": {
            "width": int(width),
            "height": int(height),
            "fps": float(fps),
            "total_frames": int(total_frames),
            "processed_frames": int(processed_frames),
            "duration_s": duration_s,
        },
        "params": {
            "gate_model": str(gate_model_path),
            "stride": int(stride),
            "conf": float(conf),
            "iou": float(iou),
            "infer_width": int(infer_width) if infer_width is not None else None,
            "max_frames": int(max_frames) if max_frames is not None else None,
        },
        "inference_timing_ms": {
            "calls": infer_calls,
            "avg": _safe_mean(infer_times_ms),
            "p50": _percentile(infer_times_ms, 50),
            "p95": _percentile(infer_times_ms, 95),
            "p99": _percentile(infer_times_ms, 99),
            "effective_fps_est": effective_fps,
            "realtime_x_est": realtime_x,
        },
        "gate_counts": {
            "calls": int(len(counts)),
            "min": int(min(counts)) if counts else None,
            "max": int(max(counts)) if counts else None,
            "mean": _safe_mean(counts),
            "p10": _percentile(counts, 10),
            "p50": _percentile(counts, 50),
            "p90": _percentile(counts, 90),
        },
        "per_call": call_rows,
        "artifacts": {
            "overlay_video": str(out_video_path),
            "summary_json": str(out_json_path),
        },
    }
    out_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Test live gate detection (gate-only overlays + timing).")
    parser.add_argument("input", help="Video file or directory")
    parser.add_argument("--gate-model", required=True, help="Path to trained gate detector weights")
    parser.add_argument("--output-dir", required=True, help="Output directory for overlays + summaries")
    parser.add_argument("--stride", type=int, default=3, help="Run inference every N frames (default 3)")
    parser.add_argument("--conf", type=float, default=0.20, help="Gate detection confidence (default 0.20)")
    parser.add_argument("--iou", type=float, default=0.45, help="Gate detection NMS IoU (default 0.45)")
    parser.add_argument("--infer-width", type=int, default=1280,
                        help="Resize width for inference (default 1280; set 0 to disable)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames")
    args = parser.parse_args()

    input_path = Path(args.input)
    gate_model = Path(args.gate_model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
        videos = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in video_exts
        )
    else:
        videos = [input_path]

    run_rows = []
    for v in videos:
        print(f"Processing {v.name}...")
        summary = run_one(
            video_path=v,
            gate_model_path=gate_model,
            output_dir=out_dir,
            stride=int(args.stride),
            conf=float(args.conf),
            iou=float(args.iou),
            infer_width=(int(args.infer_width) if int(args.infer_width) > 0 else None),
            max_frames=args.max_frames,
        )
        run_rows.append({
            "video": v.name,
            "min_gates": summary["gate_counts"]["min"],
            "p50_gates": summary["gate_counts"]["p50"],
            "max_gates": summary["gate_counts"]["max"],
            "avg_infer_ms": summary["inference_timing_ms"]["avg"],
            "p95_infer_ms": summary["inference_timing_ms"]["p95"],
            "overlay_video": summary["artifacts"]["overlay_video"],
            "summary_json": summary["artifacts"]["summary_json"],
        })

    (out_dir / "run_summary.json").write_text(json.dumps(run_rows, indent=2), encoding="utf-8")
    print(f"✓ Wrote {len(run_rows)} summaries to {out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
