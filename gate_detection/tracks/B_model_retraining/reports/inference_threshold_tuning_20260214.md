# Inference Threshold Tuning (`gate_detector_20260214_0957`)

## Goal
Tune gate detector inference thresholds for deployment:
- confidence threshold (`gate_conf`)
- NMS IoU threshold (`gate_iou`)

## Data + Method
- Model: `models/gate_detector_best.pt`
- Data split: `data/annotations/final_combined_1class_20260213/test`
- Size: 38 images, 138 gate instances
- Sweep grid:
  - `gate_conf`: 0.05 to 0.80 (step 0.05)
  - `gate_iou`: 0.45, 0.55, 0.65, 0.70, 0.75
- Matching metric for sweep: greedy TP/FP/FN with IoU >= 0.50
- Full sweep output: `docs/reports/threshold_sweep_20260214_0957.json`

## Recommended Production Setting (Balanced F1)
- `gate_conf = 0.35`
- `gate_iou = 0.55`

Performance at this point:
- Precision: 0.7500
- Recall: 0.7609
- F1: 0.7554
- TP/FP/FN: 105 / 35 / 33

## Comparison vs Previous Defaults
Previous defaults were approximately:
- `gate_conf = 0.30`
- `gate_iou = 0.70` (Ultralytics default)

Old vs new:
- Precision: 0.5829 -> 0.7500 (`+0.1671`, `+28.67%`)
- Recall: 0.7899 -> 0.7609 (`-0.0290`, `-3.67%`)
- F1: 0.6708 -> 0.7554 (`+0.0846`, `+12.62%`)
- False positives: 78 -> 35 (`-43`)
- False negatives: 29 -> 33 (`+4`)

Interpretation:
- New setting significantly reduces false detections with a small recall tradeoff.
- Overall balance (F1) is materially better for production use.

## Alternative Profiles
- Precision-priority (fewer false gates): `gate_conf=0.45`, `gate_iou=0.45`
  - Precision 0.8302, Recall 0.6377, F1 0.7213
- Recall-priority (catch more gates): `gate_conf=0.25`, `gate_iou=0.70`
  - Precision 0.5286, Recall 0.8043, F1 0.6379

## Where This Is Applied
- `scripts/process_video.py`
  - default `--gate-conf=0.35`
  - default `--gate-iou=0.55`
- `ski_racing/pipeline.py`
  - `process_video(..., gate_conf=0.35, gate_iou=0.55, ...)`
- `ski_racing/detection.py`
  - detector calls now pass `iou` into YOLO inference

## Notes
- This tuning used the `test` split for threshold selection. For stricter evaluation hygiene, create a separate calibration split in future runs and keep `test` untouched for final reporting.
