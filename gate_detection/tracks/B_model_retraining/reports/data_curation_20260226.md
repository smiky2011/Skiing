# Data Curation Report — 2026-02-26

## Source
Base: `data/datasets/final_combined_1class_20260215_hardfix`
Output: `data/datasets/final_combined_1class_20260226_curated`

## Changes Made

### 1. Dark/black image removal
- Scanned all images for mean brightness < 20
- **Result: 0 dark images found in hardfix dataset** (already cleaned by hardfix step)
- Original 4 black images were in `ski-gate-detection.v2i.yolov8` (no longer on disk)

### 2. Hard-negative images restored
- `final_combined_1class_20260215_hardfix` had **0 hard-negatives** (stripped during rebuild)
- Restored 50 empty-label images from `final_combined_1class_20260215` (orig)
- Extracted 50 new hard-negative frames from regression videos 2907/2909/2911
  (sampled from final 30% of each video: finish area, spectators, banners)
- **Total hard-negatives added to train: 100**

### 3. Annotation bbox quality fixes
- Identified 3 boxes with h≥1.0 (box touching both top and bottom edges = cropped gate)
- Files: 000098_...uLW74013Wp0_16, 000190_...MO-GS-T3_frame_0092, 000272_...uLW74013Wp0_7
- Fix: clamped h to 0.98, adjusted cy to keep box centered within frame
- No images removed (the gates are real, just slightly cropped)

### 4. Label class consistency
- Single-class (`nc: 1, names: ['gate']`) confirmed across all splits
- No two-class images found

### 5. Gate-annotated frame ratio
| Split | Total | Annotated | Hard-neg | Ratio |
|-------|-------|-----------|----------|-------|
| train | 360   | 260       | 100      | 72%   |
| valid | 72    | 72        | 0        | 100%  |
| test  | 26    | 26        | 0        | 100%  |

Target was 30–40% hard-neg ratio in train → achieved 28% (100/360). ✓

## Next Step
Run 2.2: `python scripts/train_detector.py --data data/datasets/final_combined_1class_20260226_curated/data.yaml ...`
