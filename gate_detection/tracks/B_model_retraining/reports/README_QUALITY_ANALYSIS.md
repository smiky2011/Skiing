# SKI GATE DETECTION YOLOv8 DATASET - IMAGE QUALITY ANALYSIS

## Overview

A comprehensive image quality analysis has been performed on the ski gate detection YOLOv8 dataset. This analysis identifies corrupted, dark, small, and low-contrast images that may impact model training.

## Files Generated

1. **QUALITY_ANALYSIS_REPORT.txt** - Comprehensive report with detailed findings and recommendations
2. **image_quality_issues.csv** - Detailed CSV file with all quality metrics for every image
3. **IMAGES_TO_REMOVE.txt** - Quick reference guide for problematic images
4. **README_QUALITY_ANALYSIS.md** - This file

## Quick Summary

```
Total Images Analyzed: 420
├── Train split:  293 images
├── Valid split:   85 images
└── Test split:    42 images

Quality Issues Found:
├── Dark images (brightness < 40):      4 images (0.95%)
├── Corrupted/unloadable images:        0 images (0.00%)
├── Small images (< 100x100):           0 images (0.00%)
└── Low contrast images (std dev < 15): 224 images (53.3%)
```

## Critical Issues

### COMPLETELY BLACK IMAGES (2 images) - DELETE IMMEDIATELY

These images contain NO information and must be removed:

1. `train/images/GS__Lucas-GS-T2__frame_0007_jpg.rf.8e675b93243457020496ae1e497948d3.jpg`
   - Brightness: 0.14 (nearly black)
   - Contrast: 2.74 (no variation)

2. `train/images/GS__MO-GS-T3__frame_0040_jpg.rf.7179bf5df51f3b5c7f358f6b047fda11.jpg`
   - Brightness: 0.42 (nearly black)
   - Contrast: 5.98 (minimal variation)

### NEARLY BLACK IMAGES (2 images) - REVIEW BEFORE TRAINING

These images are extremely dark and may not be useful:

1. `train/images/GS__Lucas-GS-T2__frame_0000_jpg.rf.b6ba779e51e2ba550356bd86850e6604.jpg`
   - Brightness: 3.67 (very dark)
   - Contrast: 19.07

2. `train/images/GS__Lucas-GS-T2__frame_0001_jpg.rf.d3303e09618e2eb1894b686bf4733dee.jpg`
   - Brightness: 3.93 (very dark)
   - Contrast: 19.10

**Action**: Check the annotations for these images. If annotations are sparse or missing, delete them.

## Dataset Quality Assessment

### Positive Findings

- ✓ No corrupted/unloadable images
- ✓ No images smaller than 100x100 pixels
- ✓ All images have correct 640x640 dimensions (good for YOLOv8)
- ✓ Good distribution of brightness values (average: 99.51)
- ✓ 99% of images are usable after removing dark images

### Areas of Concern

- 224 images have low contrast (std dev < 15)
  - These appear to be snow/fog scenes
  - May represent legitimate challenging detection cases
  - Should be reviewed to ensure proper annotations
  - Can be valuable for model robustness in poor visibility

## Dataset Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Brightness Range | 0.14 - 126.64 | Majority between 90-120 |
| Average Brightness | 99.51 | Good overall visibility |
| Contrast Range | 2.74 - 38.89 | Wide variation |
| Average Contrast | 16.73 | Most images have good detail |
| Image Dimensions | 640x640 | Consistent across dataset |
| Corrupted Images | 0 | Excellent file integrity |

## Recommendations

### Immediate Actions (Must Do)

1. Remove 2 completely black images (see IMAGES_TO_REMOVE.txt)
2. This removes only 0.48% of the dataset
3. Result: 418 usable images (99.5% of original)

### Recommended Actions (Should Do)

1. Review annotations for 2 nearly black images
2. Remove if annotations are sparse
3. Result: 416-418 usable images (99-99.5% of original)

### Optional Actions (Consider)

1. Analyze 224 low-contrast images for annotation quality
2. Keep those with proper annotations (good for robustness)
3. Remove those with sparse/missing annotations
4. This could improve quality but may reduce dataset size

## How to Use This Analysis

### Step 1: Review the Findings

Read **QUALITY_ANALYSIS_REPORT.txt** for comprehensive analysis and context.

### Step 2: Quick Reference

Use **IMAGES_TO_REMOVE.txt** for specific filenames and removal commands.

### Step 3: Detailed Data

Open **image_quality_issues.csv** in Excel/Python to:
- Sort by issue type
- Filter by split (train/valid/test)
- Analyze specific quality metrics
- Export for further processing

### Step 4: Implement Cleaning

```bash
# Navigate to dataset directory
cd "/sessions/admiring-magical-albattani/mnt/Stanford application project/data/annotations/ski-gate-detection.v2i.yolov8"

# Remove completely black images (CRITICAL)
rm train/images/GS__Lucas-GS-T2__frame_0007_jpg.rf.8e675b93243457020496ae1e497948d3.jpg
rm train/images/GS__MO-GS-T3__frame_0040_jpg.rf.7179bf5df51f3b5c7f358f6b047fda11.jpg

# Remove nearly black images (RECOMMENDED)
rm train/images/GS__Lucas-GS-T2__frame_0000_jpg.rf.b6ba779e51e2ba550356bd86850e6604.jpg
rm train/images/GS__Lucas-GS-T2__frame_0001_jpg.rf.d3303e09618e2eb1894b686bf4733dee.jpg
```

## Analysis Methodology

### Brightness Calculation
- Converted each image to grayscale
- Calculated mean pixel value (0-255 scale)
- Values < 40 indicate very dark/nearly black images
- Threshold is typical for unusable dark frames

### Contrast Calculation
- Calculated standard deviation of grayscale pixel values
- Values < 15 indicate low visual variation
- Can indicate blank frames, fog/snow scenes, or corrupted data

### Image Dimensions
- Checked width and height of all images
- Threshold < 100x100 would indicate small/corrupted images
- All images in this dataset are 640x640 (optimal)

## Next Steps

1. **For immediate use**: Remove the 2 completely black images and proceed with training
2. **For optimal quality**: Follow recommendations in IMAGES_TO_REMOVE.txt
3. **For detailed analysis**: Use image_quality_issues.csv to build custom filtering logic
4. **For robustness**: Consider keeping low-contrast images if they have good annotations

## Additional Notes

- The high number of low-contrast images (224) may be normal for ski racing footage
- Snow and fog naturally create low-contrast scenes
- These images can be valuable for training robustness if properly annotated
- Always verify annotations are present before removing training data

## Questions?

Refer to the detailed analysis files:
- Technical details: See "TECHNICAL DETAILS" section in QUALITY_ANALYSIS_REPORT.txt
- Specific filenames: See IMAGES_TO_REMOVE.txt
- Full metrics: See image_quality_issues.csv

---

**Analysis Date**: February 9, 2025  
**Dataset Path**: `/sessions/admiring-magical-albattani/mnt/Stanford application project/data/annotations/ski-gate-detection.v2i.yolov8`  
**Total Images Analyzed**: 420
