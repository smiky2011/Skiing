# Dataset Audit (2026-02-17)

Audited dataset:
- `tracks/B_model_retraining/artifacts/final_combined_1class_20260215`
- `tracks/B_model_retraining/artifacts/final_combined_1class_20260215_hardfix`

## Integrity checks (YOLO format)

`final_combined_1class_20260215`
- Train: 342 images, 342 labels, 50 empty-label files, 1945 boxes
- Val: 84 images, 84 labels, 12 empty-label files, 389 boxes
- Test: 23 images, 23 labels, 4 empty-label files, 54 boxes
- Calibration (not referenced in `data.yaml`): 15 images, 15 labels, 0 empty-label files, 84 boxes
- Missing image/label pairs: 0
- Content-duplicate images across splits (md5): 0
- Label parsing sanity: no out-of-range coords or invalid lines detected in scan

`final_combined_1class_20260215_hardfix`
- Train: 260 images, 260 labels, 0 empty-label files, 1423 boxes
- Val: 72 images, 72 labels, 0 empty-label files, 389 boxes
- Test: 26 images, 26 labels, 0 empty-label files, 76 boxes
- Missing image/label pairs: 0
- Content-duplicate images across splits (md5): 0
- Label parsing sanity: no out-of-range coords or invalid lines detected in scan

## Object size snapshot (normalized box area = `w*h`)

For `final_combined_1class_20260215`:
- Train median area ~0.0057 (0.57% of image), p10 ~0.0014 (0.14%)
- A large fraction of boxes are tiny: `< 0.01` area counted as “small” for quick triage

## Empty-label images: “safe negatives” vs “missed positives”

On the 66 empty-label images in `final_combined_1class_20260215`, the current model (`models/gate_detector_best.pt`) predicts at least one box with max confidence:
- ≥ 0.50 on 18/66
- ≥ 0.35 on 24/66

These are worth manual review: either they are true hard negatives (useful later, but must be balanced), or they are unlabeled positives that should be fixed.

## Split leakage risk (by inferred video/source id)

Heuristic source IDs extracted from filenames overlap heavily between splits (train/val/test/calibration). Treat the dataset’s internal `test` as a light sanity check; rely on your frozen regression videos for real “holdout” evaluation, or create a video-disjoint holdout split.

## Spot-check artifacts

Generated quick overlays (GT in green; preds from `gate_detector_best.pt` for empty-label frames):
- `gt_random_*.jpg`: random labeled frames with GT boxes
- `gt_small_*.jpg`: smallest-area labeled boxes (good for tiny-object label sanity)
- `empty_pred_*.jpg`: highest-confidence predictions on empty-label frames

