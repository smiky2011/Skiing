# Error Analysis Findings (`gate_detector_20260214_0957`)

## Scope
- Model: `models/gate_detector_best.pt`
- Thresholds: `gate_conf=0.35`, `gate_iou=0.55`
- Dataset: `data/annotations/final_combined_1class_20260213/test`
- Match rule for analysis: IoU >= 0.50

## Overall Result (at tuned thresholds)
- TP: `105`
- FP: `35`
- FN: `33`
- Precision: `0.7500`
- Recall: `0.7609`
- F1: `0.7554`

## Key Failure Patterns

1. Missed gates are generally smaller/farther.
- GT median normalized area: `0.01237`
- FN median normalized area: `0.00861` (smaller)
- GT median height: `169.94 px`
- FN median height: `119.00 px` (smaller)

2. FN errors are mostly true detection misses, with some localization misses.
- FN total: `33`
- Near-localization misses (`0.30 <= IoU < 0.50` with some prediction): `12`
- Far misses (`IoU < 0.10` to any prediction): `18`
- Interpretation: most FN are not just box misalignment; many gates are not detected at all.

3. False positives are not only low-confidence noise.
- FP confidence median: ~`0.48`
- FP confidence p90: ~`0.67`
- Several high-confidence FP are near-match boxes (`IoU ~0.40-0.49`) that fail the 0.50 match threshold.
- Interpretation: some FP are likely borderline localization/annotation alignment issues, not random background blobs.

4. Errors cluster in a small set of frames.
- Hardest frame has `TP=1, FP=2, FN=5` (error score 7).
- Top hard frames are listed in:
  - `docs/reports/error_cases_20260214/hard_cases_top20.csv`

## Notable Hard Cases
- `000246_uLW74013Wp0_4_uLW74013Wp0_4_00014_png.rf.5d362d2e166cb5aced04b6763611c3f3_40ccae57.jpg` (`FP=2`, `FN=5`)
- `000159_unseen_2911_SL_frame0403_jpg.rf.6a4401d11d50d658ff76ea4b6701a02c_f3988d94.jpg` (`FP=3`, `FN=4`)
- `000154_unseen_2907_SL_frame0132_jpg.rf.c245e1534bce98d2b3241dc3ca379380_38ec7d20.jpg` (`FP=4`, `FN=2`)
- `000153_unseen_2907_SL_frame0033_jpg.rf.c691c80019d2683aaa4dbc8ce825864c_af52e14d.jpg` (`FP=3`, `FN=2`)

Pure misses (no predictions but GT exists):
- `000006_v8_div_balv2_0143__Videos__5UHRvqx1iuQ__0-mp4__t2-41_jpg.rf.56214351a2ce4fce5581_4111112f.jpg`
- `000007_v8_div_balv2_0150__Videos__he3w2n9WvrI__16-mp4__t5-98_jpg.rf.90c766a6cd466c2bfc4_d339fc30.jpg`
- `000008_v8_div_balv2_0161__Videos__he3w2n9WvrI__10-mp4__t5-98_jpg.rf.19ba2b430791579b997_edd50986.jpg`

## What To Do Next (Data-Centric Priority)

1. Relabel/verify top-20 hard frames first.
- Focus on tight box placement and ambiguous partial/occluded poles.
- This should reduce both near-localization FP and FN.

2. Add targeted training data for small/far gates.
- Mine frames where gate height is roughly `<120 px`.
- Prioritize low-contrast and motion-blur scenes from the pure-miss clips above.

3. Add hard-negative review around high-confidence FP frames.
- Review non-gate vertical structures/snow artifacts that trigger `conf > 0.6`.
- Include corrected negatives in next training batch.

4. Keep current inference defaults for now.
- `gate_conf=0.35`, `gate_iou=0.55` remains the best balanced production point.
- Re-run threshold sweep after next retrain.

## Where To Inspect Results
- Machine-readable error stats:
  - `docs/reports/error_analysis_20260214_0957.json`
- Hard-case overlays (GT/TP/FP/FN drawn):
  - `docs/reports/error_cases_20260214/`
- Top 20 hard-case index:
  - `docs/reports/error_cases_20260214/hard_cases_top20.csv`
