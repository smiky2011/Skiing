# Evaluation Summary

- Generated: 2026-02-15 10:48:33
- Model: `/Users/quan/Documents/personal/Stanford application project/tracks/B_model_retraining/runs/detect/gate_detector_20260215_1016/weights/best.pt`
- Git commit: `f3b113c`
- Baseline: `/Users/quan/Documents/personal/Stanford application project/tracks/B_model_retraining/reports/eval_20260215_1009_02/eval_result.json`
- Verdict: **FAIL**

## Stage 1 - Holdout Detection Metrics

| Confidence | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.6500 | 0.7222 | 0.6842 | 39 | 21 | 15 |
| 0.35 | 0.7143 | 0.6481 | 0.6796 | 35 | 14 | 19 |
| 0.45 | 0.7647 | 0.4815 | 0.5909 | 26 | 8 | 28 |
| 0.55 | 0.7826 | 0.3333 | 0.4675 | 18 | 5 | 36 |

## Stage 2 - Regression Suite

| Video | Gates | Coverage | P90 speed (km/h) | Max speed (km/h) | Max G-force | Max jump (m) | Auto-cal correction | Physics issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2907` | 6 | 0.989 | 33.82 | 1902.39 | 2030.39 | 17.63 | 1.99 | 5 |
| `2909` | 5 | 0.997 | 27.45 | 497.58 | 488.18 | 4.61 | 2.59 | 4 |
| `2911` | 6 | 0.968 | 149.76 | 1781.31 | 2403.23 | 16.50 | 8.05 | 5 |
| **Mean** | **5.667** | **0.985** | **70.34** | **1393.76** | **1640.60** | **12.91** | **4.21** | **4.667** |

## Delta vs Baseline

| Metric | Baseline | Current | Delta | Delta % |
|---|---:|---:|---:|---:|
| F1 | 0.8727 | 0.6796 | -0.1931 | -22.13% |
| Gates detected | 5.3333 | 5.6667 | +0.3333 | +6.25% |
| Trajectory coverage | 0.9846 | 0.9846 | +0.0000 | +0.00% |
| P90 speed (km/h) | 153.8709 | 70.3430 | -83.5279 | -54.28% |
| Max speed (km/h) | 12157.7612 | 1393.7583 | -10764.0029 | -88.54% |
| Max G-force | 5575.5739 | 1640.5984 | -3934.9755 | -70.58% |
| Max jump (m) | 112.6391 | 12.9114 | -99.7277 | -88.54% |
| Auto-cal correction | 5.5382 | 4.2128 | -1.3254 | -23.93% |
| Physics issue count | 5.0000 | 4.6667 | -0.3333 | -6.67% |

## Verdict

**FAIL**

Reasons:
- F1 decreased (0.6796 < baseline 0.8727).
