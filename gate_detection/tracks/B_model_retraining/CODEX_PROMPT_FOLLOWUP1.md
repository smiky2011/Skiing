# Track B Follow-up #1 — Conservative Retrain (Hard Frames Only)

## What happened last time

The previous retraining cycle **failed** — F1 dropped from 0.8727 to 0.6796 (22% regression). The model lost 13 true positives and gained 13 false negatives. Root cause: the combined dataset added **20 hard negatives** (images with no gates) but **0 new small gate images** (the mining script found 0 qualifying candidates). This biased the model toward suppressing detections.

## What to do differently this time

**Strategy: only use the hard-frame label repairs.** No hard negatives, no mined small gates. Just fix the annotations that were wrong and retrain.

### Step 1: Verify hard frame repairs

The previous round created repaired labels in `tracks/B_model_retraining/artifacts/`. Verify the repair quality:

```bash
# Check that repaired label files have reasonable box counts
for f in tracks/B_model_retraining/artifacts/hard_case_repairs_20260215/labels/*.txt; do
  echo "$(wc -l < "$f") boxes: $f"
done
```

Manually spot-check 3-5 files: each line should be `0 cx cy w h` format with coordinates in 0-1 range. Any line with class != 0 or coordinates outside 0-1 is corrupt.

### Step 2: Build minimal dataset

Combine ONLY the original baseline + repaired hard frames:

```bash
python scripts/combine_yolo_datasets.py \
  --sources data/annotations/final_combined_1class_20260213 \
            tracks/B_model_retraining/artifacts/hard_case_repairs_20260215 \
  --output data/annotations/final_combined_1class_$(date +%Y%m%d)_hardfix \
  --keep-empty false
```

**Critical:** Do NOT include `targeted_addon_20260215` (the hard negatives) or `mined_small_gates_20260215` this time.

The `--keep-empty false` flag ensures no empty-label images sneak in.

### Step 3: Verify dataset integrity before training

```bash
# Count images and labels per split
for split in train valid test; do
  imgs=$(ls data/annotations/final_combined_1class_*_hardfix/images/$split/ 2>/dev/null | wc -l)
  lbls=$(ls data/annotations/final_combined_1class_*_hardfix/labels/$split/ 2>/dev/null | wc -l)
  echo "$split: $imgs images, $lbls labels"
done

# Verify no empty label files in train split
find data/annotations/final_combined_1class_*_hardfix/labels/train -name "*.txt" -empty | wc -l
# Should be 0
```

### Step 4: Train (fine-tune from existing best model)

Fine-tune from the current best checkpoint — NOT from scratch. The baseline model (F1=0.87) already works well; we just need to nudge it with corrected labels. 40 epochs is enough.

```bash
python scripts/train_detector.py \
  --data data/annotations/final_combined_1class_*_hardfix/data.yaml \
  --model models/gate_detector_best.pt --epochs 40 --imgsz 960 --batch 8 --freeze 10 --cos-lr
```

If `train_detector.py` does not accept a `.pt` checkpoint as `--model`, use the ultralytics CLI directly:
```bash
yolo detect train data=data/annotations/final_combined_1class_*_hardfix/data.yaml \
  model=models/gate_detector_best.pt epochs=40 imgsz=960 batch=8 freeze=10 cos_lr=True
```

### Step 5: Threshold sweep on calibration split

Use the same calibration split from last time (15 images):
```bash
# Sweep conf 0.20-0.60 step 0.05, iou 0.45-0.65 step 0.10
```

Pick conf/iou maximizing F1 on calibration. Expected: conf~0.35, iou~0.55 (same as before).

### Step 6: Evaluate on frozen test

```bash
python scripts/run_eval.py \
  --gate-model models/gate_detector_best_$(date +%Y%m%d)_*.pt \
  --videos tracks/E_evaluation_ci/regression_videos/IMG_2907.mp4 \
           tracks/E_evaluation_ci/regression_videos/IMG_2909.mp4 \
           tracks/E_evaluation_ci/regression_videos/IMG_2911.mp4 \
  --discipline slalom --stabilize --summary \
  --output-dir tracks/B_model_retraining/reports/eval_$(date +%Y%m%d)_hardfix
```

### Step 7: Decide

**Promote** the new model ONLY if:
1. F1 on frozen test ≥ 0.8727 (match or beat baseline — NOT the failed 0.6796)
2. Regression video auto-cal correction does not increase vs baseline (was 5.54× mean)

If F1 improves even slightly (e.g., 0.88+), that's a win — the hard frame repairs fixed real annotation errors.

If F1 is about the same (~0.87), the annotations were already OK and we've confirmed the baseline is solid.

If F1 drops again, the repair process itself is introducing errors — revert and keep the baseline model.

## Files you own
- `data/annotations/` (new dataset version)
- `models/` (new checkpoint)
- `tracks/B_model_retraining/reports/` (evaluation results)

## Do NOT modify
- `ski_racing/*.py` (not your files)
- `data/annotations/final_combined_1class_20260213/test/` (frozen test — read only)

## Save report to
`tracks/B_model_retraining/reports/eval_YYYYMMDD_hardfix/summary.md`
