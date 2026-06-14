# Ski Skeleton Overlay

First usable version for the workflow in `IMPLEMENTATION_STEPS.md`:

`ski video -> detect/track one skier -> estimate 2D body keypoints -> smooth -> render confidence-aware overlay MP4 + JSON`.

Uses Ultralytics YOLO pose with ByteTrack by default. MediaPipe is available as an experimental backend in the code, but it can fail in headless environments because MediaPipe tries to create an OpenGL context.

## Quick Start

Run a short test on one of your videos:

```bash
python3 run_overlay.py path/to/video.mp4 --max-frames 100
```

Open the UI:

```bash
streamlit run app.py
```

Outputs are written to `outputs/`:

- `*_skeleton_overlay.mp4`
- `*_keypoints.json`
- `*_report.json`

## CLI Examples

Process a whole clip:

```bash
python3 run_overlay.py path/to/video.mp4
```

Try the experimental MediaPipe backend:

```bash
python3 run_overlay.py path/to/video.mp4 --backend mediapipe
```

If MediaPipe raises an OpenGL/NSOpenGL error, use the default YOLO backend.

Use a larger YOLO input size for far skiers:

```bash
python3 run_overlay.py path/to/video.mp4 --imgsz 1600 --conf 0.20
```

Use the far-skier preset for distant footage:

```bash
python3 run_overlay.py path/to/video.mp4 --preset far
```

If several people are visible, give a target hint. The point is an `x,y` coordinate near the skier on the selected frame:

```bash
python3 run_overlay.py path/to/video.mp4 --preset far --target-frame 225 --target-point 650,210
```

Use the gate-occlusion preset when the skier is briefly hidden by a gate or snow spray:

```bash
python3 run_overlay.py path/to/video.mp4 --preset occlusion
```

Test only a gate-crossing window:

```bash
python3 run_overlay.py path/to/video.mp4 --preset occlusion --start-frame 940 --max-frames 60
```

If the model jumps to the wrong person, seed the first skier with a box:

```bash
python3 run_overlay.py path/to/video.mp4 --initial-box 420,160,720,860
```

## Status Labels

- `tracking_ok`: the tracked skier has enough reliable core joints.
- `low_confidence`: the person is detected but core joints are weak; uncertain joints are faded or hidden.
- `tracking_lost`: no target skier was selected on this frame.
- `far_skier`: the skier box is small enough that joints are less trustworthy.
- `skier_too_small`: the selected box is too small for trustworthy hips/knees/ankles.

The overlay intentionally hides weak limbs instead of drawing fake confident skeletons.

## Practical Notes

- Video IO uses OpenCV `VideoCapture`/`VideoWriter`, so `ffmpeg` is not required.
- The default model is `ski_pose_v1.pt` — our Ski-2DPose fine-tuned `yolo11s-pose` ("ski-base v1"), kept as a local runtime asset at the repo root (it is not auto-downloaded). Pass `--model yolo11n-pose.pt` (CLI) or change the model field (app) to fall back to the stock detector, which Ultralytics fetches automatically; `yolo11s-pose.pt` is also supported for comparison.
- Far-skier mode raises YOLO input size, lowers confidence, and enables a second pose pass on the selected skier crop. It improves recall but can pick the wrong person unless you provide a target hint.
- Gate occlusion cannot be fully solved by a 2D model alone. The current tool now tries a predicted crop around the skier when full-frame detection fails, then carries short missing-joint gaps with optical flow as `inferred`, and hides them if the occlusion lasts too long.
- The keypoint JSON records `candidate_source`: `detector`, `crop_refine`, `roi_recovery`, `motion_inference`, or `none`.
- For best accuracy later, validate RTMPose-L/MMPose against this MVP on a labeled ski frame set before changing the production backend.
