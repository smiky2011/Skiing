# Proposed `MODEL_REGISTRY.md` Update (Track B)

Target file (read-only in this run):  
`/Users/quan/Documents/personal/Stanford application project/shared/docs/MODEL_REGISTRY.md`

Append the section below:

---

## Wave 2 Candidate Pose Model (2026-02-19, Not Promoted)
- Checkpoint: `tracks/B_model_retraining/artifacts/models/gate_pose_best_20260219.pt`
- Source run: `tracks/B_model_retraining/runs/pose/gate_pose_20260219_1844/weights/best.pt`
- Dataset: `tracks/B_model_retraining/artifacts/pose_1class_20260219/data.yaml`
- Training mode: `yolov8n-pose.yaml` fallback (offline environment, no pretrained pose weights download)
- Keypoint config: `kpt_shape: [2,3]` (`base`, `tip`)
- Notes:
  - Dataset bootstrapped from bbox labels plus keypoint scaffolding manifest.
  - Per-frame detections JSONs generated and schema-validated:
    - `tracks/B_model_retraining/outputs/2907_1765738705(Video in Original Quality)_detections.json`
    - `tracks/B_model_retraining/outputs/2909_1765738725(Video in Original Quality)_detections.json`
    - `tracks/B_model_retraining/outputs/2911_1765738746(Video in Original Quality)_detections.json`
  - Ablation report: `tracks/B_model_retraining/reports/ablation_fallback_20260219.md`
  - Status: **Not promoted** to active detector alias.

---
