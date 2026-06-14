# Evaluation Summary

- Generated: 2026-02-15 10:13:20
- Model: `/Users/quan/Documents/personal/Stanford application project/models/gate_detector_best.pt`
- Git commit: `f3b113c`
- Verdict: **PASS**

## Stage 1 - Holdout Detection Metrics

| Confidence | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.8033 | 0.9074 | 0.8522 | 49 | 12 | 5 |
| 0.35 | 0.8571 | 0.8889 | 0.8727 | 48 | 8 | 6 |
| 0.45 | 0.8182 | 0.6667 | 0.7347 | 36 | 8 | 18 |
| 0.55 | 0.8710 | 0.5000 | 0.6353 | 27 | 4 | 27 |

## Stage 2 - Regression Suite

| Video | Gates | Coverage | P90 speed (km/h) | Max speed (km/h) | Max G-force | Max jump (m) | Auto-cal correction | Physics issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2907` | 7 | 0.989 | 119.86 | 21868.87 | 2124.86 | 202.62 | 5.18 | 5 |
| `2909` | 4 | 0.997 | 175.57 | 5839.03 | 8510.91 | 54.12 | 5.06 | 5 |
| `2911` | 5 | 0.968 | 166.18 | 8765.39 | 6090.95 | 81.17 | 6.38 | 5 |
| **Mean** | **5.333** | **0.985** | **153.87** | **12157.76** | **5575.57** | **112.64** | **5.54** | **5.000** |

## Verdict

**PASS**

Reasons:
- No baseline provided; skipped regression delta checks.
