# Wave 2 Track B Execution Summary (2026-02-19)

## Completed in this workspace

1. Pose dataset bootstrap
- Output dataset: `artifacts/pose_1class_20260219/`
- Data config: `artifacts/pose_1class_20260219/data.yaml` (`kpt_shape: [2, 3]`)
- Manifest: `artifacts/pose_1class_20260219/annotation_manifest.csv`
- Summary: `artifacts/pose_1class_20260219/dataset_summary.json`

2. Pose model training
- Training script: `tools/train_pose_model_v2.py`
- Run directory: `runs/pose/gate_pose_20260219_1844/`
- Checkpoint: `artifacts/models/gate_pose_best_20260219.pt`
- Training report: `reports/pose_training_20260219.json`

3. Tiered fallback + geometry + emission output writer
- Implementation script: `tools/generate_per_frame_detections_v2.py`
- Includes:
  - `resolve_gate_base()` Tier-1/2/3 logic
  - rolling-shutter geometry check
  - `emission_log_prob` calculation
  - schema validation (`shared/interfaces/per_frame_detections.schema.json`)

4. Per-clip outputs
- `outputs/2907_1765738705(Video in Original Quality)_detections.json`
- `outputs/2909_1765738725(Video in Original Quality)_detections.json`
- `outputs/2911_1765738746(Video in Original Quality)_detections.json`

5. Ablation and run reports
- Fallback ablation: `reports/ablation_fallback_20260219.md`
- Detection run summary: `reports/per_frame_detection_run_20260219_185424.json`

6. Interface + registry docs (track-local due write boundary)
- Interface agreement copy/sign-off: `reports/interface_agreement_trackC_trackB_20260219.md`
- Proposed model registry update snippet: `reports/model_registry_update_proposed_20260219.md`
- Proposed upstream `ski_racing/detection.py` change note: `reports/proposed_ski_racing_detection_py_changes_20260219.md`

## Acceptance checks from generated outputs

- Tier-2 jitter lower than Tier-3 counterfactual: **PASS** (see ablation report).
- `geometry_check_passed` includes both true and false cases: **PASS**.
- `emission_log_prob <= 0` for all detections: **PASS**.
- Output JSON schema validation: **PASS** for all three clips.

## Constraint note

This run had write access restricted to `tracks/B_model_retraining/` only.  
Direct edits to:
- `ski_racing/detection.py`
- `tracks/C_bev_egomotion/INTERFACE_AGREEMENT.md`
- `shared/docs/MODEL_REGISTRY.md`
were not performed; track-local equivalents are included above.
