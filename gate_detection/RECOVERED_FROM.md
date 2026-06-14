# Recovered course-gate detection system

Recovered 2026-06-14 into `ski_project(codex)` for the **gate → course-corridor → racer-selection prior** work (issue #20). This is @smiky2011's prior Stanford project's gate-detection system, which had been removed from that repo during a "skiing cleanup" and survived only on old branches.

## Source
- Repo: `/Users/quan/Documents/personal/Stanford` (git)
- Commit: `72ec6a90` ("Test live gate detection videos", 2026-03-03); present on branches `backup/main-before-skiing-cleanup-20260320` and several `feature/*`. NOT on the current `improvements-v2` branch.
- Recovered via `git archive 72ec6a90 <paths> | tar -x`.

## What this is
A complete **single-class (`gate`) course-gate detection → tracking → counting** system:
- **`ski_racing/`** — core package: `detection.py`, `tracking.py`, `transform.py` (geometry/BEV), `pipeline.py`, `initialiser.py`, `physics.py` (gate spacing), `decoder.py`, `safety.py`, `visualize.py`.
- **`scripts/`** — `train_detector.py`, `inference/process_video.py`, `eval_unseen_course_gate_counts.py`, `tune_course_gate_counter.py`, `promote_model.py`, `run_eval.py`, etc.
- **`tests/`** — `test_course_gate_counter.py`, `test_gate_consensus.py`, `test_detection.py`, `test_pipeline_gate_tracking_quality.py`, `test_initialiser.py`, `test_physics.py`, …
- **`configs/`** — `course_gate_defaults.yaml` (tuned: conf 0.2, stride 2, min-hits 3, dedup/track params), `regression_defaults.yaml` (references the trained model below).
- **`data/datasets/`** — labeled YOLO data, **1 class `gate`**:
  - `final_combined_1class_20260215/` — 342 train / 84 val / 23 test.
  - `ablation_neg0/` — negative/ablation set. (814 images + 814 labels total recovered.)
- **`tracks/B_model_retraining/`** — the retraining track: README, CODEX prompts, curation/audit reports (`.md`).

## The trained model is NOT here
The config references `models/gate_detector_neg20_ensemble.pt` (an **ensemble**, conf ≈ 0.36, 1-class `gate`). `*.pt` was gitignored in the source repo, so the weights are not recoverable from git. **Re-train from the dataset above** (standard 1-class Ultralytics YOLO training; fast) or locate the original weights (possibly on Colab / external).

## Pruned during recovery (regenerable; still in Stanford history)
Test videos (`*.MOV`, ~105M), tuning-sweep / unseen-eval result dumps (`tests/sweeps_*`, `tests/unseen_eval_*`, `tests/verify_*`), large per-video detection/analysis JSONs (`tracks/.../outputs/`, `>1M *.json`), audit-image dumps, `*.cache`. Kept: all code, configs, the labeled dataset, and the `.md` reports.

## Why we want it (issue #20)
An alpine racer is defined by running the gated course; recreational skiers/crowd/barriers are off-course. A trained gate detector → fitted course corridor is the **racer-selection prior** that color segmentation couldn't deliver (it can't separate gates from same-colored barriers; a learned detector can). Next: re-train the gate detector, run on our clips (qiaobo / 1592 / 1571), verify it generalizes, then fit a corridor and feed it as the prior alongside the existing discriminator.
