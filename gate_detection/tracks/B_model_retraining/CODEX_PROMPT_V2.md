# Codex Prompt — Track B v2: YOLOv8-Pose + 3-Tier Keypoint Fallback (Wave 2, Worker B)

Paste everything below into a Codex environment with access to the full project root.

> **Note:** This is the v2.1 prompt. The earlier `CODEX_PROMPT.md` and `CODEX_PROMPT_FOLLOWUP1.md` in this folder covered the original bbox detector retraining loop. That work is complete. This prompt covers the new Wave 2 work: upgrading to YOLOv8-Pose (2 keypoints) and implementing the 3-tier fallback hierarchy from the v2.1 spec.

---

## Your role

You are **Wave 2, Worker B**. You run in parallel with Track C (Worker A). You own the gate detector upgrade and the keypoint fallback hierarchy. Your outputs (per-frame detections with emission log-probabilities) are consumed by Track D (Kalman), Track F (Viterbi), and Track G (initialisation). **Coordinate the VP_t interface with Track C before writing any code** — see "Critical coordination" below.

---

## Context

The existing detector (`models/gate_detector_best.pt`) is a standard YOLOv8 bbox detector. It detects gate poles but has no keypoints. The v2.1 spec (Phase 3) upgrades this to YOLOv8-Pose with 2 keypoints per gate:
- **Base** (kp index 0): where the pole meets the snow surface — the ground truth position we track
- **Tip** (kp index 1): the top of the gate panel — used for Tier-2 fallback projection

The critical insight driving this upgrade: when snow spray or the skier's body occludes the gate base, the old bbox bottom-centre fallback was tracking the bottom of the occluding object — not the gate. The 3-tier hierarchy fixes this by using the pole tip + vanishing point to project the occluded base geometrically.

---

## Files to read first (in this order)

1. `tracks/README.md` — architecture and your role
2. `shared/interfaces/per_frame_detections.schema.json` — your PRIMARY output schema, must conform exactly
3. `shared/interfaces/per_frame_bev.schema.json` — your input from Track C (VP_t and horizon_y_px)
4. `shared/interfaces/sidecar_pts.schema.json` — for resolution LUT (tr) used in geometry checks
5. `tracker_spec_v2.docx` — Section 4.2 (rolling shutter LUT), Section 4.3 (fallback hierarchy table), Phase 3 (full spec)
6. `tracks/B_model_retraining/README.md` — acceptance criteria
7. `tracks/B_model_retraining/CODEX_PROMPT.md` — prior work context (read for background, don't redo it)
8. `ski_racing/detection.py` — existing detection code you will extend

---

## Critical coordination (do this FIRST, before writing code)

**With Track C (Wave 2 Worker A):** Your Tier-2 fallback uses `vp_t` (vanishing point) and `horizon_y_px` from Track C's BEV output. You need to agree on the coordinate system before either of you writes any code. Read `tracks/C_bev_egomotion/CODEX_PROMPT.md` section "Critical coordination", then sign `tracks/C_bev_egomotion/INTERFACE_AGREEMENT.md` once it is written.

The Tier-2 fallback is only available when Track C's `alpha_t > 0.0`. When `alpha_t = 0.0` (VP fully frozen or unavailable), you must fall through to Tier-3.

---

## What to build

### Step 1: Dataset annotation for YOLOv8-Pose

The existing annotation dataset (`data/annotations/final_combined_1class_20260215/`) has bbox labels only. You need to add keypoint annotations for the best available images:

For each image, annotate:
- Keypoint 0 (base): pixel coordinate at the snow surface contact point of the pole. If obscured, mark as `(0, 0, 0)` (invisible).
- Keypoint 1 (tip): pixel coordinate at the top of the gate panel. If obscured, mark as `(0, 0, 0)`.

YOLOv8-Pose label format (per object, space-separated):
```
<class_id> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <kp0_vis> <kp1_x> <kp1_y> <kp1_vis>
```
All coordinates normalised 0–1. Visibility: 0=not labelled, 1=occluded, 2=visible.

Prioritise annotating the top 50 images from the existing training set (use the hard cases from `tracks/B_model_retraining/reports/`). Save annotated dataset to `data/annotations/pose_1class_YYYYMMDD/`.

### Step 2: Train YOLOv8-Pose

```bash
python scripts/train_detector.py \
  --data data/annotations/pose_1class_YYYYMMDD/data.yaml \
  --model yolov8n-pose.pt \
  --epochs 150 --imgsz 960 --batch 8
```

The `data.yaml` must declare `kpt_shape: [2, 3]` (2 keypoints, 3 values each: x, y, visibility).

Save checkpoint to `models/gate_pose_best_YYYYMMDD.pt`.

### Step 3: Implement the 3-tier fallback hierarchy

In `ski_racing/detection.py`, implement `resolve_gate_base(detection, bev_frame, tau_kp=0.5)`:

**Tier 1 — Keypoint base** (use when `kp0_conf >= tau_kp`):
```
base_px = keypoint_base pixel coords
base_fallback_tier = 1
is_degraded = False
```

**Tier 2 — VP-Projected base** (use when `kp0_conf < tau_kp` AND `bev_frame.alpha_t > 0`):
```
# Line through pole tip (kp1) and VP_t
# Parameterised as: P = kp1_px + t * (VP_t - kp1_px)
# Find t where P.y == horizon_y_px
t = (horizon_y_px - kp1_py) / (VP_t.y - kp1_py)
base_px.x = kp1_px + t * (VP_t.x - kp1_px)
base_px.y = horizon_y_px
base_fallback_tier = 2
is_degraded = False
```
If kp1 is also invisible (kp1_conf < tau_kp), fall through to Tier 3.

**Tier 3 — Bbox bottom-centre** (last resort: `kp0_conf < tau_kp` AND (`alpha_t == 0` OR `kp1_conf < tau_kp`)):
```
base_px = (bbox_x1 + bbox_x2) / 2, bbox_y2
base_fallback_tier = 3
is_degraded = True  # always flag DEGRADED for Tier 3
```

### Step 4: Rolling shutter geometry check

For each detection where both keypoints are visible, compute the pole vector angle from vertical:
```
dx = kp1_x - kp0_x  (tip minus base, x component)
dy = kp1_y - kp0_y  (tip minus base, y component — negative because tip is above base)
pole_vector_angle_deg = degrees(arctan2(dx, -dy))  # 0° = perfectly vertical
```

Look up `rolling_shutter_theta_deg` from the corresponding BEV frame output (Track C provides this per frame). Accept the detection if:
```
|pole_vector_angle_deg| <= theta_deg + 5.0
```
Set `geometry_check_passed = True/False`. Do NOT hard-reject failed checks — set `geometry_check_passed=False` and let the tracker downstream soft-penalise it.

### Step 5: Compute emission log-probabilities

For each detection, compute per-state log emission probabilities for the HMM (Track F will read these directly from your output — it must NOT re-run the detector):

```python
# Simple colour-based prior using class confidence
# Adjust these priors after seeing flat-light performance
log_prob_red  = log(conf_class) if class_label == "red"  else log(1 - conf_class + 1e-9)
log_prob_blue = log(conf_class) if class_label == "blue" else log(1 - conf_class + 1e-9)
log_prob_dnf  = log(0.05)  # small prior — DNF is rare, triggered by geometry not colour
```

Store these in `emission_log_prob` in the output schema. All values must be <= 0.

### Step 6: Output writer

Write one JSON file per clip to `tracks/B_model_retraining/outputs/<clip_id>_detections.json` conforming exactly to `shared/interfaces/per_frame_detections.schema.json`. Validate against schema before writing.

---

## Files you own

- `ski_racing/detection.py` — extend with keypoint fallback and geometry check logic
- `data/annotations/pose_1class_YYYYMMDD/` — new keypoint dataset
- `models/gate_pose_best_YYYYMMDD.pt` — new model checkpoint
- `tracks/B_model_retraining/outputs/` — per-clip detection JSONs
- `tracks/B_model_retraining/reports/` — ablation and geometry check reports

## Do NOT modify

- `shared/interfaces/` — READ ONLY
- `ski_racing/tracking.py` — owned by Track D
- `ski_racing/transform.py` — owned by Track C
- `tracks/C_bev_egomotion/` — read only (consume their BEV outputs, don't edit them)

---

## Deliverables

- `ski_racing/detection.py` — with `resolve_gate_base()` and geometry check implemented
- `models/gate_pose_best_YYYYMMDD.pt` — YOLOv8-Pose checkpoint
- `tracks/B_model_retraining/outputs/<clip_id>_detections.json` — one per clip
- `tracks/B_model_retraining/reports/ablation_fallback_YYYYMMDD.md` — Tier 1 vs Tier 2 vs Tier 3 jitter comparison
- Updated `shared/docs/MODEL_REGISTRY.md` with new pose model entry

---

## Pass criteria

1. Tier 2 (VP-projected base) produces lower temporal jitter std(||p_t - p_{t-1}||) than Tier 3 (bbox) on clips where the base keypoint is occluded but `alpha_t > 0`. Document this comparison in the ablation report.
2. `geometry_check_passed` is correctly set to False for poles leaning beyond `theta + 5°` and True for valid poles.
3. All `emission_log_prob` values are <= 0 (valid log probabilities).
4. All output JSONs validate against `shared/interfaces/per_frame_detections.schema.json`.
5. `tracks/C_bev_egomotion/INTERFACE_AGREEMENT.md` is signed by you before you implement Tier 2.
