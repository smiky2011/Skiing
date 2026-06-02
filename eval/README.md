# Detection Evaluation Set

This folder defines a small review set for improving detection quality before adding ski-specific metrics.

## Purpose

The goal is to compare detection changes against the same short clips every time. This prevents tuning the pipeline for one video while accidentally making far-skier, occlusion, or multi-person behavior worse elsewhere.

Each case in `cases.json` records:

- the source video
- the frame window to process
- the preset or target hint to use
- the failure mode to review

## Reviewer Workflow

1. Run one case and inspect the generated overlay MP4, keypoint JSON, and report JSON.
2. Decide whether the intended skier stayed selected.
3. Check whether hips, knees, and ankles are believable enough for later metrics.
4. Mark obvious failures in notes before changing model or tracking code.

Example command:

```bash
python3 run_overlay.py "/Users/quan/Documents/石泉/ski/1575_raw.MP4" \
  --preset occlusion \
  --start-frame 900 \
  --max-frames 120 \
  --output-dir outputs/eval_gate_occlusion_1575_recovery
```

## Success Criteria

A detection-quality change is useful only if it improves one target failure without causing clear regressions in the other cases. Prioritize stable target identity and reliable lower-body keypoints over visually smooth but incorrect skeletons.
