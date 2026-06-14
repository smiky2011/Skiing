# Wave 2 Interface Agreement (Track B Copy)

Date: 2026-02-19  
Participants: Track B Worker B (this workspace), Track C Worker A (spec reference)

## Agreed VP/Horizon Contract

1. Coordinate system
- Image pixel coordinates.
- Origin: top-left.
- `x_px`: rightward positive.
- `y_px`: downward positive.

2. `vp_t`
- `vp_t.x_px`, `vp_t.y_px` are image-space vanishing-point coordinates after Track C smoothing.

3. `horizon_y_px`
- Image-space horizon y coordinate used by Track B Tier-2 projection:
  `t = (horizon_y_px - kp1_y) / (vp_t.y_px - kp1_y)`
  `base_x = kp1_x + t * (vp_t.x_px - kp1_x)`
  `base_y = horizon_y_px`

4. Tier-2 availability
- Tier-2 is enabled only when `alpha_t > 0.0`.
- If `alpha_t == 0.0`, Track B falls through to Tier-3.

5. File naming
- Track C output naming convention expected by Track B:
  `tracks/C_bev_egomotion/outputs/<clip_id>_bev.json`

## Sign-off

- Track B Worker B: **SIGNED** (2026-02-19)
- Track C Worker A: pending in Track C workspace

## Note

Write access in this run is restricted to `tracks/B_model_retraining/`, so this sign-off copy is stored in Track B reports. Mirror this content into `tracks/C_bev_egomotion/INTERFACE_AGREEMENT.md` when Track C workspace writes are available.
