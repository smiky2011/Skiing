"""Image-space course-corridor prototype (issue #20; prototype-only, NO pipeline changes).

Per Codex: gate-detect at conf=0.45 -> DEDUP gates (don't fit raw per-frame boxes;
1571 double-boxes poles) -> fit an IMAGE-SPACE centerline x=f(y) (no homography;
transform.py self-warns it's wrong under panning) -> debug overlay + JSON corridor
coords + gate-support diagnostics.

The centerline weaves through the (deduped) gates sorted by depth (base_y) -- that's
the course path; `lateral_error_px` is the off-course distance a later racer-selection
prior would use. Per-frame fit (follows the panning camera); carry-forward the last
good centerline on frames with too few gates.

Reuses ski_racing.detection.GateDetector and the _build_course_centerline approach
(replicated here to stay standalone). Dedup ratios from configs/course_gate_defaults.yaml.

Usage:
  python gate_detection/fit_corridor.py --model gate_yolo11s.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from ski_racing.detection import GateDetector  # noqa: E402

DEFAULT_CASES = [
    "indoor_far_qiaobo_day2_moving",
    "outdoor_clear_1592_standard",
    "outdoor_moving_1571_far",
    "gate_occlusion_1575_recovery",
]
DEDUP_DX_RATIO = 0.03   # course_gate_defaults.yaml
DEDUP_DY_RATIO = 0.05


def detect(det, frame, conf, iou, imgsz):
    gates = []
    for b in det.model(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0].boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].cpu().numpy())
        gates.append({"bbox": [x1, y1, x2, y2], "confidence": float(b.conf[0]),
                      "center_x": (x1 + x2) / 2, "base_y": y2})
    return gates


def dedup(gates, W, H):
    """Merge boxes that are the same gate (centers within dx/dy ratios); keep max conf."""
    dx, dy = DEDUP_DX_RATIO * W, DEDUP_DY_RATIO * H
    kept = []
    for g in sorted(gates, key=lambda g: -g["confidence"]):
        if any(abs(g["center_x"] - k["center_x"]) < dx and abs(g["base_y"] - k["base_y"]) < dy
               for k in kept):
            continue
        kept.append(g)
    return kept


def centerline(gates):
    """Course-SPINE fit: smoothed least-squares polynomial x=f(y) through gate
    centers (not piecewise-linear through every gate — GS gates alternate sides,
    so interpolation zigzags; a low-order LS poly recovers the course direction
    and the band covers the weave). Degree adapts to gate count.
    """
    pts = sorted(((g["base_y"], g["center_x"]) for g in gates), key=lambda t: t[0])
    ys, xs, keep = [p[0] for p in pts], [p[1] for p in pts], [0] if pts else []
    for i in range(1, len(ys)):
        if abs(ys[i] - ys[keep[-1]]) >= 2.0:
            keep.append(i)
    ys = np.array([ys[i] for i in keep], float); xs = np.array([xs[i] for i in keep], float)
    if len(ys) < 2:
        return None

    def _fit(yy, xx):
        deg = 2 if len(yy) >= 5 else 1
        c = np.polyfit(yy, xx, deg)
        return c if deg == 2 else np.array([0.0, c[0], c[1]])   # pad linear -> [c2,c1,c0]

    c = _fit(ys, xs)
    inl = np.ones(len(ys), bool)
    # one robust pass: drop outlier gates (a stray off-line gate tilts the spine)
    if len(ys) >= 4:
        resid = np.abs(xs - np.polyval(c, ys))
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
        inl = resid <= 2.5 * mad
        if 2 <= inl.sum() < len(ys):
            c = _fit(ys[inl], xs[inl])
    rms = float(np.sqrt(np.mean((xs[inl] - np.polyval(c, ys[inl])) ** 2))) if inl.sum() else 0.0
    return {"coeffs": c.tolist(), "y0": float(ys[0]), "y1": float(ys[-1]),
            # reliability fields (Codex): support, vertical extent, fit tightness
            "n_gates": int(inl.sum()), "y_span": float(ys[-1] - ys[0]), "residual_px": round(rms, 1)}


def x_at_y(cl, y):
    y = float(min(max(y, cl["y0"]), cl["y1"]))   # clamp to observed gate depth range
    return float(np.polyval(cl["coeffs"], y))


def ema(prev, cur, a):
    """Temporal smoothing of the spine (EMA on coeffs + y-range) to kill jitter."""
    if prev is None:
        return cur
    if cur is None:
        return prev
    pc, cc = np.array(prev["coeffs"]), np.array(cur["coeffs"])
    return {"coeffs": (a * cc + (1 - a) * pc).tolist(),
            "y0": a * cur["y0"] + (1 - a) * prev["y0"],
            "y1": a * cur["y1"] + (1 - a) * prev["y1"],
            # reliability reflects the CURRENT frame's detections
            "n_gates": cur.get("n_gates", 0), "y_span": cur.get("y_span", 0.0),
            "residual_px": cur.get("residual_px", 0.0)}


def draw(frame, gates, cl, band_px):
    for g in gates:
        x1, y1, x2, y2 = (int(v) for v in g["bbox"])
        col = (0, 200, 0) if g["confidence"] >= 0.5 else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    if cl is not None:
        ys = list(range(int(cl["y0"]), int(cl["y1"]) + 1, 8))
        pts = [(int(x_at_y(cl, y)), y) for y in ys]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (255, 80, 0), 3, cv2.LINE_AA)          # course centerline (blue)
        for sgn in (-1, 1):                                              # band edges (faint cyan)
            edge = [(int(x_at_y(cl, y) + sgn * band_px), y) for y in ys]
            for a, b in zip(edge, edge[1:]):
                cv2.line(frame, a, b, (200, 200, 0), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conf", type=float, default=0.45)   # Codex: 0.45 baseline
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--band-ratio", type=float, default=0.25, help="corridor half-width / frame width")
    ap.add_argument("--smooth-alpha", type=float, default=0.35, help="EMA weight on the new frame (lower = smoother)")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--cases-file", default=str(REPO / "eval" / "cases.json"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "gate_corridor"))
    args = ap.parse_args()

    cases = {c["id"]: c for c in json.load(open(args.cases_file, encoding="utf-8"))["cases"]}
    det = GateDetector(args.model)
    out_root = Path(args.out)
    summary = {}

    for cid in args.cases:
        case = cases.get(cid)
        if not case:
            print(f"[skip] {cid}: not in cases.json"); continue
        video = Path(str(case["video"])).expanduser()
        start = int(case.get("start_frame", 0)); maxf = case.get("max_frames")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[skip] {cid}: cannot open {video}"); continue
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        band_px = args.band_ratio * W
        out_dir = out_root / cid; out_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_dir / f"{cid}_corridor.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        per_frame, ema_cl, i, support, carried = [], None, 0, [], 0
        while True:
            if maxf is not None and i >= int(maxf):
                break
            ok, frame = cap.read()
            if not ok:
                break
            gates = dedup(detect(det, frame, args.conf, args.iou, args.imgsz), W, H)
            raw_cl = centerline(gates)
            if raw_cl is None:                   # too few gates -> carry the smoothed corridor
                if ema_cl is not None:
                    carried += 1
            else:                                # temporal EMA -> kills frame-to-frame jitter
                ema_cl = ema(ema_cl, raw_cl, args.smooth_alpha)
            cl = ema_cl
            support.append(len(gates))
            writer.write(draw(frame, gates, cl, band_px))
            per_frame.append({
                "frame_index": start + i, "n_gates": len(gates),
                "centerline": [[round(y, 1), round(x_at_y(cl, y), 1)]
                               for y in range(0, H, 40)] if cl else None,
            })
            i += 1
        writer.release(); cap.release()
        json.dump({"case": cid, "video": str(video), "conf": args.conf,
                   "band_ratio": args.band_ratio, "frames": per_frame},
                  open(out_dir / "corridor.json", "w"), indent=2)
        n = len(support)
        summary[cid] = {"frames": n, "avg_gates_after_dedup": round(sum(support) / n, 2) if n else 0,
                        "frames_with_corridor": sum(1 for f in per_frame if f["centerline"]),
                        "frames_carried_forward": carried}
        print(f"[{cid}] {n} frames, avg {summary[cid]['avg_gates_after_dedup']} deduped gates/frame, "
              f"corridor on {summary[cid]['frames_with_corridor']} frames ({carried} carried) -> {out_dir}")

    json.dump(summary, open(out_root / "summary.json", "w"), indent=2)
    print(f"\nCorridor overlays + JSON in {out_root}. Review: does the line follow the GS course "
          f"and exclude the off-course/barrier side? (issue #20)")


if __name__ == "__main__":
    main()
