# Fallback Ablation Report (2026-02-19)

## Scope
- Compares Tier-1/2/3 base resolution usage and temporal jitter proxies.
- Validates geometry check flags and emission log-probability constraints.
- Uses BEV from Track C when present; otherwise uses deterministic local BEV stub.

## Aggregate
- Clips processed: 3
- Detections total: 13224
- Tier 1 count: 1324
- Tier 2 count: 6228
- Tier 3 count: 5672
- Geometry check failures: 481
- Clips with stub BEV usage: 3

## Jitter Comparison
- Tier-2 jitter std median: 60.7170
- Tier-3 jitter std median: 74.3808
- Tier-3 counterfactual on Tier-2 events (median std): 109.4121
- Pass criterion (Tier-2 lower jitter than Tier-3 baseline): PASS
- Detail: Tier-2 median jitter std (60.717) < Tier-3 counterfactual median jitter std (109.412).

## Per-clip Summary
### 2907_1765738705(Video in Original Quality)
- Frames written: 180
- Detections: 3453
- Tier counts: {'1': 650, '2': 1572, '3': 1231}
- Geometry failures: 192
- Jitter std: {'tier2': 60.71695459044935, 'tier3': 74.38077625349266, 'tier2_counterfactual_tier3': 120.21535043744646}
- Output JSON: `/Users/quan/Documents/personal/Stanford application project/tracks/B_model_retraining/outputs/2907_1765738705(Video in Original Quality)_detections.json`

### 2909_1765738725(Video in Original Quality)
- Frames written: 180
- Detections: 5484
- Tier counts: {'1': 455, '2': 2832, '3': 2197}
- Geometry failures: 211
- Jitter std: {'tier2': 49.068790061026995, 'tier3': 70.38896357484252, 'tier2_counterfactual_tier3': 107.65044010280603}
- Output JSON: `/Users/quan/Documents/personal/Stanford application project/tracks/B_model_retraining/outputs/2909_1765738725(Video in Original Quality)_detections.json`

### 2911_1765738746(Video in Original Quality)
- Frames written: 180
- Detections: 4287
- Tier counts: {'1': 219, '2': 1824, '3': 2244}
- Geometry failures: 78
- Jitter std: {'tier2': 60.94269281617522, 'tier3': 83.87806863739372, 'tier2_counterfactual_tier3': 109.41212707847177}
- Output JSON: `/Users/quan/Documents/personal/Stanford application project/tracks/B_model_retraining/outputs/2911_1765738746(Video in Original Quality)_detections.json`
