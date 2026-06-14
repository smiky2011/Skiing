"""Gate-detector generalization probe (issue #20, prototype-only — no pipeline changes).

Runs a re-trained 1-class gate detector on sampled frames of OUR ski clips and
emits overlays + JSON, so we can eyeball the go/no-go:
  - qiaobo  -> detects the indoor GS gates AND rejects right-wall/barrier graphics?
  - 1592/1571 (outdoor) -> no hallucinated gates?
  - 1575 -> sane (guard)?

Reuses `ski_racing.detection.GateDetector` for model loading; calls the model
with an explicit imgsz (the module's detect_in_frame hardcodes YOLO's 640 default,
too low for far/small gates — we train at 960). Clip list + windows come from
eval/cases.json (the source of truth for our review clips).

Usage:
  python gate_detection/probe_gates.py --model gate_yolo11s.pt
  python gate_detection/probe_gates.py --model best.pt --conf 0.25 --imgsz 960 --stride 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent          # gate_detection/
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))               # so `ski_racing` imports
from ski_racing.detection import GateDetector   # noqa: E402

DEFAULT_CASES = [
    "indoor_far_qiaobo_day2_moving",   # the residual: gates among barriers + many skiers
    "outdoor_clear_1592_standard",     # outdoor far — must not hallucinate
    "outdoor_moving_1571_far",         # outdoor — must not hallucinate
    "gate_occlusion_1575_recovery",    # guard
]


def draw(frame, gates):
    for g in gates:
        x1, y1, x2, y2 = (int(v) for v in g["bbox"])
        c = g["confidence"]
        col = (0, 200, 0) if c >= 0.5 else (0, 165, 255)  # green high / orange low
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f'{g.get("class_name","gate")}:{c:.2f}', (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="re-trained gate .pt (kept out of git)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=960, help="inference imgsz (match training)")
    ap.add_argument("--stride", type=int, default=15, help="frames between sampled overlays")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--cases-file", default=str(REPO / "eval" / "cases.json"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "gate_probe"))
    args = ap.parse_args()

    cases = {c["id"]: c for c in json.load(open(args.cases_file, encoding="utf-8"))["cases"]}
    det = GateDetector(args.model)
    out_root = Path(args.out)
    summary = {}

    for cid in args.cases:
        case = cases.get(cid)
        if not case:
            print(f"[skip] {cid}: not in cases.json")
            continue
        video = Path(str(case["video"])).expanduser()
        start = int(case.get("start_frame", 0))
        maxf = case.get("max_frames")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[skip] {cid}: cannot open {video}")
            continue
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        out_dir = out_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)

        per_frame, counts, i = [], [], 0
        while True:
            if maxf is not None and i >= int(maxf):
                break
            ok, frame = cap.read()
            if not ok:
                break
            fi = start + i
            if i % args.stride == 0:
                res = det.model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
                gates = []
                for b in res[0].boxes:
                    xy = b.xyxy[0].cpu().numpy()
                    gates.append({
                        "class_name": det.model.names[int(b.cls[0])],
                        "bbox": [float(v) for v in xy],
                        "confidence": float(b.conf[0]),
                        "center_x": float((xy[0] + xy[2]) / 2),
                        "base_y": float(xy[3]),
                    })
                cv2.imwrite(str(out_dir / f"f{fi:05d}.jpg"), draw(frame, gates))
                per_frame.append({"frame_index": fi, "n_gates": len(gates), "gates": gates})
                counts.append(len(gates))
            i += 1
        cap.release()
        json.dump({"case": cid, "video": str(video), "model": args.model,
                   "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                   "frames": per_frame}, open(out_dir / "detections.json", "w"), indent=2)
        n = len(counts)
        summary[cid] = {"frames_sampled": n,
                        "gates_per_frame_avg": round(sum(counts) / n, 2) if n else 0,
                        "gates_per_frame_max": max(counts) if counts else 0,
                        "frames_with_>=1_gate": sum(1 for c in counts if c)}
        print(f"[{cid}] sampled {n}: avg {summary[cid]['gates_per_frame_avg']} gates/frame, "
              f"max {summary[cid]['gates_per_frame_max']}, "
              f"{summary[cid]['frames_with_>=1_gate']} frames with a gate -> {out_dir}")

    json.dump(summary, open(out_root / "summary.json", "w"), indent=2)
    print(f"\nOverlays + detections in {out_root}. Review against the go/no-go (issue #20).")


if __name__ == "__main__":
    main()
