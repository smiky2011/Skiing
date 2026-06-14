# Manager Dispatch: Worker B HOLD

Timestamp: 2026-02-14 20:56:01
Status: HOLD
Reason: Track B starts only after Track D and Track E make progress.

## Release Conditions
- Track D shows progress beyond baseline in `tracks/D_tracking_outlier/reports/`:
  - New report includes Kalman Q sweep (`kalman_tuning_YYYYMMDD.md`), and
  - Output diagnostics include `outlier_count`, `bytetrack_coverage`, `track_id_switches`, and
  - 2D max jump is trending toward `< 50px` on all 3 regression videos.
- Track E shows eval pipeline progress:
  - `scripts/run_eval.py` exists and runs,
  - deterministic repeated runs,
  - backup model produces FAIL.

## Note
Do not begin retraining cycles until this hold is lifted by manager.
