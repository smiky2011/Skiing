# Codex Prompt for Track B — Model Retraining Loop

Paste everything below into the Codex thread that has access to `tracks/B_model_retraining/`.

---

Improve the YOLOv8 gate detector through data-centric retraining. The model detects ski racing gates (poles) in video frames. Your job is to fix annotation issues, add targeted training data, retrain, and verify the new model is better.

## Current baseline

- Model: YOLOv8n (6.3 MB), file: `models/gate_detector_best.pt`
- Training data: `data/annotations/final_combined_1class_20260213/` (~420 images)
- Test set: 38 images, 138 gate instances
- Metrics: Precision=0.7500, Recall=0.7609, F1=0.7554, TP=105, FP=35, FN=33
- Thresholds: gate_conf=0.35, gate_iou=0.55

## Known failure patterns (already analyzed — read the reports)

1. **Missed gates are smaller/farther:** FN median height is 119px vs GT median 170px. The model misses distant gates.
2. **18 of 33 FN are complete misses** — not even close detections. Many gates simply not detected.
3. **FP are not just noise:** FP confidence median ~0.48, P90 ~0.67. Some are near-match boxes (IoU 0.40–0.49) failing the 0.50 threshold. Could be annotation alignment issues.
4. **Errors cluster in ~20 hard frames.** See `tracks/B_model_retraining/reports/error_cases_20260214/hard_cases_top20.csv`.

## What to do (in order)

### Step 1: Create calibration split
Take 40% of the current test set (`data/annotations/final_combined_1class_20260213/test/`) and move it to a new `calibration/` folder. Keep the remaining 60% as the frozen test set. Threshold tuning happens on calibration, final metrics on test. This prevents test set contamination.

### Step 2: Fix top-20 hard frames
Open each image listed in `tracks/B_model_retraining/reports/error_cases_20260214/hard_cases_top20.csv`. Cross-reference with the overlay images in `tracks/B_model_retraining/reports/error_cases_20260214/` (they show GT boxes in green, TP in blue, FP in red, FN in yellow). Verify every bounding box is tight around the gate pole. Remove false annotations. Fix misaligned boxes.

### Step 3: Mine small gate images
Find 30–50 frames from existing videos where gate height is roughly <120px. These are the distant gates the model misses. Prioritize low-contrast (snow/fog) and motion-blur scenes. The pure-miss examples in `tracks/B_model_retraining/reports/error_analysis_findings_20260214.md` list specific frames to start with. Annotate them in YOLO format.

### Step 4: Add hard negatives
Find 15+ frames containing non-gate vertical structures that might confuse the model — fence posts, tree trunks, timing poles, ski pole handles. Make sure these have NO gate annotations (or correct annotations if gates are also present). Include these in training to reduce high-confidence false positives.

### Step 5: Combine and train
```bash
python scripts/combine_yolo_datasets.py \
  --sources data/annotations/final_combined_1class_20260213 data/annotations/<new_data> \
  --output data/annotations/final_combined_1class_YYYYMMDD

python scripts/train_detector.py \
  --data data/annotations/final_combined_1class_YYYYMMDD/data.yaml \
  --model yolov8n.pt --epochs 150 --imgsz 960 --batch 8 --freeze 10 --cos-lr
```

### Step 6: Threshold sweep on calibration split
Sweep gate_conf from 0.05–0.80 (step 0.05) and gate_iou from 0.45–0.75 (step 0.10). Pick the conf/iou that maximizes F1 on the calibration split. Do NOT use the test split for this.

### Step 7: Evaluate on frozen test split
Run evaluation with the new checkpoint + best thresholds. Record P/R/F1/TP/FP/FN.

### Step 8: Run 3-video regression
Process videos 2907, 2909, 2911 (in `tracks/E_evaluation_ci/regression_videos/`) through the full pipeline with `--stabilize`. Record max speed, max G-force, max jump, auto-cal correction. Compare to previous checkpoint.

### Step 9: Decide
Promote the new checkpoint ONLY if: (a) F1 on test split >= 0.755, AND (b) regression video metrics don't degrade significantly.

## Files to read first

1. `tracks/B_model_retraining/Track_B_Model_Retraining_Loop.docx` — full spec
2. `tracks/B_model_retraining/reports/error_analysis_findings_20260214.md` — what's broken and why
3. `tracks/B_model_retraining/reports/error_cases_20260214/hard_cases_top20.csv` — the hard frames
4. `tracks/B_model_retraining/reports/inference_threshold_tuning_20260214.md` — current threshold analysis
5. `data/annotations/README_FINAL_DATASET.md` — dataset structure

## Files you own

- `data/annotations/` (create new dataset versions)
- `models/` (save new checkpoints)
- `scripts/train_detector.py`, `scripts/evaluate.py`, `scripts/combine_yolo_datasets.py`

## Do NOT modify

- `ski_racing/*.py` (not your responsibility)
- `data/annotations/final_combined_1class_20260213/test/` (frozen test set — only split off calibration subset)

## Deliverables per cycle

- New checkpoint: `models/gate_detector_best_YYYYMMDD_HHMM.pt`
- Threshold sweep: `tracks/B_model_retraining/reports/threshold_sweep_YYYYMMDD.json`
- Test evaluation: `tracks/B_model_retraining/reports/eval_YYYYMMDD.json`
- Regression comparison: `tracks/B_model_retraining/reports/regression_YYYYMMDD.md`
- Updated `shared/docs/MODEL_REGISTRY.md`

## Pass criteria

- F1 > 0.755 on frozen test split
- Regression video metrics (max speed, max G, max jump) do not degrade
