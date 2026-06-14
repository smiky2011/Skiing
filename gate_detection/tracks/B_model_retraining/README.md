# Track B: Model Retraining Loop

## Owner
Assigned to: _______________

## Goal
Improve gate detection accuracy (currently F1 = 0.755) through data-centric retraining cycles.

## Your Spec Document
**Read this first:** `Track_B_Model_Retraining_Loop.docx` (in this folder)

## Files You Own (edit these)
| File | What it does |
|------|-------------|
| `data/annotations/` | Training datasets — create new versions here |
| `models/` | Save new checkpoints here |
| `scripts/train_detector.py` | Training script (modify only if ablating hyperparams) |
| `scripts/evaluate.py` | Evaluation script |
| `scripts/combine_yolo_datasets.py` | Dataset merging tool |

## Files You Read (do NOT edit)
| File | Why you need it |
|------|----------------|
| `ski_racing/detection.py` | Understand how the model is used at inference |
| `ski_racing/pipeline.py` | Understand end-to-end pipeline parameters |

## Your Reports Folder
`tracks/B_model_retraining/reports/` — contains all existing analysis:
- Error analysis (which frames are hard, what FP/FN look like)
- Threshold sweep results
- Model comparison reports
- Dataset quality analysis

## Key Commands
```bash
# Train a new model
python scripts/train_detector.py \
  --data data/annotations/final_combined_1class_YYYYMMDD/data.yaml \
  --model yolov8n.pt --epochs 150 --imgsz 960 --batch 8 --freeze 10 --cos-lr

# Evaluate on test split
python scripts/evaluate.py \
  --model models/gate_detector_best_NEW.pt \
  --data data/annotations/final_combined_1class_YYYYMMDD/test

# Run full pipeline on a regression video
python scripts/process_video.py data/test_videos_unseen/VIDEO.mp4 \
  --gate-model models/gate_detector_best_NEW.pt \
  --discipline slalom --stabilize --summary
```

## Dependencies
- **Depends on Track E**: Use their evaluation pipeline to measure your checkpoints
- **Depends on Track D** (soft): Cleaner trajectories mean better end-to-end metrics
