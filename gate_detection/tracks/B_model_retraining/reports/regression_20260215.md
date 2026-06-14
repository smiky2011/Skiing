# Regression Comparison - 2026-02-15

- Generated: 2026-02-15 10:49:32
- Baseline eval: `/Users/quan/Documents/personal/Stanford application project/tracks/B_model_retraining/reports/eval_20260215_1009_02/eval_result.json`
- Candidate eval: `/Users/quan/Documents/personal/Stanford application project/tracks/B_model_retraining/reports/eval_20260215_1045/eval_result.json`

## Stage 1 (Frozen Test)

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Precision | 0.8571 | 0.7143 | -0.1429 |
| Recall | 0.8889 | 0.6481 | -0.2407 |
| F1 | 0.8727 | 0.6796 | -0.1931 |
| TP | 48 | 35 | -13 |
| FP | 8 | 14 | +6 |
| FN | 6 | 19 | +13 |

## Stage 2 (3-Video Regression)

| Video | Metric | Baseline | Candidate | Delta |
|---|---|---:|---:|---:|
| 2907 | Max speed (km/h) | 21868.870 | 1902.388 | -19966.482 |
| 2907 | Max G-force | 2124.857 | 2030.393 | -94.464 |
| 2907 | Max jump (m) | 202.621 | 17.626 | -184.995 |
| 2907 | Auto-cal correction | 5.180 | 1.995 | -3.185 |
| 2909 | Max speed (km/h) | 5839.027 | 497.580 | -5341.447 |
| 2909 | Max G-force | 8510.910 | 488.175 | -8022.735 |
| 2909 | Max jump (m) | 54.124 | 4.612 | -49.512 |
| 2909 | Auto-cal correction | 5.058 | 2.591 | -2.466 |
| 2911 | Max speed (km/h) | 8765.386 | 1781.307 | -6984.079 |
| 2911 | Max G-force | 6090.954 | 2403.226 | -3687.728 |
| 2911 | Max jump (m) | 81.171 | 16.496 | -64.676 |
| 2911 | Auto-cal correction | 6.377 | 8.053 | +1.675 |

## Decision

- Candidate FAILS promotion criteria and is not promoted.
- F1 decreased (0.6796 < baseline 0.8727).

## Artifacts

- Checkpoint: `tracks/B_model_retraining/artifacts/models/gate_detector_best_20260215_1043.pt`
- Threshold sweep: `tracks/B_model_retraining/reports/threshold_sweep_20260215.json`
- Frozen-test eval: `tracks/B_model_retraining/reports/eval_20260215.json`
- Full run-eval report: `tracks/B_model_retraining/reports/eval_20260215_1045/eval_result.json`
