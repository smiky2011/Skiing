# Proposed `ski_racing/detection.py` Changes (Track B, v2.1)

Write access in this run is restricted to `tracks/B_model_retraining/`, so the implementation was added in:
- `tracks/B_model_retraining/tools/generate_per_frame_detections_v2.py`

The following functions are the intended upstream additions to `ski_racing/detection.py`:
- `resolve_gate_base(detection, bev_frame, tau_kp=0.5)`
- `compute_geometry_check(kp0, kp1, bev_frame, tau_kp=0.5)`
- `emission_log_prob(class_label, conf_class)`

## Function Contracts

1. `resolve_gate_base(...)`
- Tier 1: use `keypoint_base_px` when `kp0_conf >= tau_kp`.
- Tier 2: when `kp0_conf < tau_kp` and `alpha_t > 0` and `kp1_conf >= tau_kp`, project through `vp_t` to `horizon_y_px`.
- Tier 3: fallback to bbox bottom-center.
- Output: `base_px`, `base_fallback_tier`, `is_degraded`.

2. `compute_geometry_check(...)`
- If both keypoints are confident, compute:
  `pole_vector_angle_deg = degrees(arctan2(kp1_x-kp0_x, -(kp1_y-kp0_y)))`
- Pass condition:
  `abs(pole_vector_angle_deg) <= rolling_shutter_theta_deg + 5.0`
- Return `(angle_or_none, geometry_check_passed_bool)`.

3. `emission_log_prob(...)`
- Returns `log_prob_red`, `log_prob_blue`, `log_prob_dnf`.
- All values clamped to `<= 0`.

## Reference Implementation

See:
- `tracks/B_model_retraining/tools/generate_per_frame_detections_v2.py`
