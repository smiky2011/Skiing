"""Corridor lateral-error SEPARATION TEST (issue #20 / PR #21 follow-up).

THE decision gate before any ski_pose_overlay change: does the course corridor's
`lateral_error` actually separate the racer from the confusing other-skier on
qiaobo f0-135?

Per frame: build the corridor (gate-detect 0.45 -> dedup -> spine + EMA, with
reliability), and run the v1 pose detector for ALL person candidates. Each person
gets lateral_error = |center_x - corridor_x_at_y(center_y)| (only on reliable
corridor frames). Persons are clustered into tracks by position continuity; per
track we report median lateral_error, persistence, and on-course fraction.

Read it as: ONE persistent low-lateral_error track -> corridor isolates the racer
(GO, soft prior). TWO+ low tracks -> the confuser is also on-course, corridor
can't disambiguate (NO-GO). Overlay (green=on-course, red=off, grey=unreliable
corridor) lets the user confirm which track is the racer.

Usage:
  python gate_detection/separation_test.py --gate-model gate_yolo11s.pt --pose-model ski_pose_v1.pt
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
from ski_racing.detection import GateDetector              # noqa: E402
import fit_corridor as FC                                  # noqa: E402
from ultralytics import YOLO                               # noqa: E402


def reliable(cl, W):
    return cl is not None and cl.get("n_gates", 0) >= 3 and cl.get("residual_px", 1e9) < 0.12 * W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-model", required=True)
    ap.add_argument("--pose-model", default=str(REPO / "ski_pose_v1.pt"))
    ap.add_argument("--case", default="indoor_far_qiaobo_day2_moving")
    ap.add_argument("--window", nargs=2, type=int, default=[0, 135])
    ap.add_argument("--gate-conf", type=float, default=0.45)
    ap.add_argument("--pose-conf", type=float, default=0.05)   # far/tiny skiers need low conf
    ap.add_argument("--imgsz", type=int, default=960)           # gate detection imgsz
    ap.add_argument("--pose-imgsz", type=int, default=1536)     # person detection imgsz (tiny skiers)
    ap.add_argument("--band-ratio", type=float, default=0.25)
    ap.add_argument("--smooth-alpha", type=float, default=0.35)
    ap.add_argument("--match-ratio", type=float, default=0.06, help="track-match center dist / W")
    ap.add_argument("--out", default=str(REPO / "outputs" / "corridor_separation"))
    args = ap.parse_args()

    case = {c["id"]: c for c in json.load(open(REPO / "eval" / "cases.json"))["cases"]}[args.case]
    video = Path(str(case["video"])).expanduser()
    lo, hi = args.window
    gate_det = GateDetector(args.gate_model)
    pose = YOLO(args.pose_model)
    out_dir = Path(args.out) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    band_px = args.band_ratio * W
    writer = cv2.VideoWriter(str(out_dir / f"{args.case}_separation.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracks = []   # each: {"cx","cy","les":[], "frames":[], "id"}
    ema_cl, fi, per_frame = None, lo, []
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    while fi <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        gates = FC.dedup(FC.detect(gate_det, frame, args.gate_conf, 0.45, args.imgsz), W, H)
        raw = FC.centerline(gates)
        if raw is not None:
            ema_cl = FC.ema(ema_cl, raw, args.smooth_alpha)
        cl = ema_cl
        rel = reliable(cl, W)
        # corridor draw
        if cl is not None:
            ys = list(range(int(cl["y0"]), int(cl["y1"]) + 1, 8))
            col = (255, 120, 0) if rel else (130, 130, 130)
            pts = [(int(FC.x_at_y(cl, y)), y) for y in ys]
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, a, b, col, 3, cv2.LINE_AA)
        # persons
        res = pose(frame, conf=args.pose_conf, imgsz=args.pose_imgsz, verbose=False)[0]
        frame_persons = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            le = abs(cx - FC.x_at_y(cl, cy)) if rel else None
            on = (le is not None and le <= band_px)
            color = (130, 130, 130) if le is None else ((0, 200, 0) if on else (0, 0, 230))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if le is not None:
                cv2.putText(frame, f"{le:.0f}", (int(x1), max(12, int(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            frame_persons.append({"cx": cx, "cy": cy, "le": le})
            # track association
            m = min(tracks, key=lambda t: abs(t["cx"] - cx) + abs(t["cy"] - cy), default=None)
            if m is not None and abs(m["cx"] - cx) < args.match_ratio * W and abs(m["cy"] - cy) < 0.10 * H:
                m["cx"], m["cy"] = cx, cy
                m["frames"].append(fi)
                if le is not None:
                    m["les"].append(le)
            else:
                tracks.append({"cx": cx, "cy": cy, "les": ([le] if le is not None else []),
                               "frames": [fi], "id": len(tracks)})
        writer.write(frame)
        per_frame.append({"frame": fi, "reliable": rel, "n_persons": len(frame_persons),
                          "lateral_errors": [round(p["le"], 1) if p["le"] is not None else None for p in frame_persons]})
        fi += 1
    writer.release(); cap.release()

    # track summary: persistent tracks with their median lateral_error
    rows = []
    for t in tracks:
        if len(t["frames"]) < 8 or not t["les"]:        # ignore blips
            continue
        med = float(np.median(t["les"]))
        rows.append({"id": t["id"], "n_frames": len(t["frames"]), "n_le": len(t["les"]),
                     "median_le": round(med, 1), "median_le_pct_W": round(100 * med / W, 1),
                     "on_course_frac": round(sum(1 for e in t["les"] if e <= band_px) / len(t["les"]), 2)})
    rows.sort(key=lambda r: r["median_le"])
    json.dump({"case": args.case, "window": [lo, hi], "band_px": round(band_px, 1),
               "tracks": rows, "per_frame": per_frame}, open(out_dir / "separation.json", "w"), indent=2)

    band_pct = round(100 * args.band_ratio, 1)
    print(f"\n=== {args.case} f{lo}-{hi}: persistent tracks (>=8 frames), sorted by median lateral_error ===")
    print(f"band = {band_pct}% of width ({band_px:.0f}px). on-course = lateral_error <= band.\n")
    print(f"{'track':>5} {'frames':>6} {'medLE_px':>9} {'medLE_%W':>9} {'on_course_frac':>14}")
    for r in rows:
        print(f"{r['id']:>5} {r['n_frames']:>6} {r['median_le']:>9} {r['median_le_pct_W']:>9} {r['on_course_frac']:>14}")
    oncourse = [r for r in rows if r["median_le"] <= band_px]
    print(f"\n{len(oncourse)} persistent track(s) sit on-course (median LE <= band). "
          f"Overlay: {out_dir}/{args.case}_separation.mp4")
    print("Read: 1 on-course track -> corridor isolates the racer (GO). 2+ -> confuser also on-course (NO-GO).")


if __name__ == "__main__":
    main()
