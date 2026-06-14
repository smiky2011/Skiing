# Evaluation Summary (Hard-Frame Repairs Only)

- Generated: 2026-02-15
- Dataset: `tracks/B_model_retraining/artifacts/final_combined_1class_20260215_hardfix`
- Frozen test split: `tracks/B_model_retraining/artifacts/final_combined_1class_20260215/test`
- Target: **F1 >= 0.8727** on frozen test

## What Was Run

1. Built repairs-only training dataset (baseline + repaired calibration hard frames; no hard negatives, no mined small gates).
2. Training attempt A:
   - Model: `runs/detect/gate_detector_20260215_1113/weights/best.pt`
   - Calibration sweep best: conf=0.25, nms_iou=0.65, F1=0.4916
   - Frozen-test performance: not promoted (calibration too weak; run interrupted early due downward trend).
3. Training attempt B (conservative low-LR fine-tune):
   - Model: `runs/detect/gate_detector_hardfix_lowlr_20260215_1120/weights/best.pt`
   - Calibration sweep best: conf=0.25, nms_iou=0.45, F1=0.5402
   - Frozen-test (selected conf/iou): Precision=0.7857, Recall=0.8148, **F1=0.8000** (TP=44, FP=12, FN=10)
   - Frozen-test best observed across tested thresholds: **F1=0.8000**

## Baseline Reference

- Baseline model: `models/gate_detector_best.pt`
- Frozen-test at conf=0.35, nms_iou=0.55:
  - Precision=0.8571, Recall=0.8889, **F1=0.8727** (TP=48, FP=8, FN=6)

## Regression Suite Status

- Full `scripts/run_eval.py` execution is currently blocked by an existing syntax error in `ski_racing/transform.py` (`IndentationError` at line 1154).
- Per prompt constraints, `ski_racing/*.py` was not modified.
- Existing baseline regression reference remains:
  - `reports/eval_20260215_1009_02/eval_result.json`
  - Mean auto-cal correction: **5.54x**

## Decision

- **Do not promote** either repairs-only retrained checkpoint from this cycle.
- Retrained F1 does not meet target (0.8727).
- Revert to baseline checkpoint for deployment parity.
- Baseline copy saved as:
  - `tracks/B_model_retraining/artifacts/models/gate_detector_best_20260215_hardfix_revert.pt`

## Artifacts

- Dataset combine report:
  - `tracks/B_model_retraining/artifacts/final_combined_1class_20260215_hardfix/combine_report.json`
- Calibration sweeps:
  - `tracks/B_model_retraining/reports/threshold_sweep_20260215_hardfix.json`
  - `tracks/B_model_retraining/reports/threshold_sweep_20260215_hardfix_lowlr.json`
- Frozen-test eval JSONs:
  - `tracks/B_model_retraining/reports/eval_20260215_hardfix_stage1.json`
  - `tracks/B_model_retraining/reports/eval_20260215_hardfix_stage1_nms055.json`
  - `tracks/B_model_retraining/reports/eval_20260215_hardfix_baseline_stage1.json`
