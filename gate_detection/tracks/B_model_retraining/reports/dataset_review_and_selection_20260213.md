# Dataset Review And Selection (2026-02-13)

## Scope Reviewed
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/combined_1class`
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/ski-gate-detection-dataset1.v6i.yolov8`
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/ski-gate-detection-dataset3.v1i.yolov8`
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/ski-gate-detection.v2i.yolov8.filtered`
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/ski-gate-detection.v2i.yolov8.filtered_1class`
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/v7_curated`
- `/Users/quan/Documents/personal/Stanford application project/data/ski-gate-detection-dataset1.v7i.yolov8 (1)`

Audit outputs:
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/_audit_reports/dataset_audit_summary.csv`
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/_audit_reports/dataset_audit_summary.json`

## Key Findings
- `combined_1class`: strong positive set, no empty labels, minor invalid geometry corrected during merge filtering.
- `v7_curated`: high quality, but some empty-label images were likely unlabeled positives.
- `dataset3.v1`: many empty labels; model check flagged many as likely missed gates.
- `dataset1.v6`: many invalid label lines, low quality for direct use.
- `dataset1.v7`: very high empty-label ratio and many risky empties.
- `filtered` and `filtered_1class`: mostly overlap with `combined_1class`.

## Cleaning Performed
- Built cleaned sources by dropping only risky empty-label images (`max_conf >= 0.50` by current detector):
  - `/Users/quan/Documents/personal/Stanford application project/data/annotations/_cleaned_sources/v7_curated.clean_conf50`
  - `/Users/quan/Documents/personal/Stanford application project/data/annotations/_cleaned_sources/ski-gate-detection-dataset3.v1i.yolov8.clean_conf50`

## Final Selected Sources For Training
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/combined_1class`
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/_cleaned_sources/v7_curated.clean_conf50`
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/_cleaned_sources/ski-gate-detection-dataset3.v1i.yolov8.clean_conf50`

## Final Combined Dataset (Use This)
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/final_combined_1class_20260213`
- Data config: `/Users/quan/Documents/personal/Stanford application project/data/annotations/final_combined_1class_20260213/data.yaml`

Stats:
- Images: `404`
- Boxes: `1890`
- Empty labels kept as safe negatives: `46`
- Invalid labels in final set: `0`

Merge report:
- `/Users/quan/Documents/personal/Stanford application project/data/annotations/final_combined_1class_20260213/combine_report.json`

## Training Command
```bash
yolo detect train \
  data="/Users/quan/Documents/personal/Stanford application project/data/annotations/final_combined_1class_20260213/data.yaml" \
  model="/Users/quan/Documents/personal/Stanford application project/models/yolov8s.pt" \
  imgsz=640 \
  epochs=120
```
