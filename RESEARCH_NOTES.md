# Research Notes

Date: 2026-05-28

## Recommendation

Use two tracks:

1. **Usable local MVP now:** Ultralytics YOLO pose + ByteTrack + One Euro smoothing + confidence-aware rendering. This is implemented here because `opencv`, `torch`, `ultralytics`, and `streamlit` are already installed, and the `yolo11n-pose.pt` / `yolo11s-pose.pt` weight files are available locally.
2. **Accuracy target for validation:** RTMDet/RTMPose-L through MMPose/MMDetection + ByteTrack. This better matches the top-down detector/crop/pose pipeline recommended in `IMPLEMENTATION_STEPS.md`, but it is heavier to install and validate on macOS/Python 3.12.

Do not add coaching or technique scoring until hips, knees, ankles, and shoulders are trustworthy on a ski-specific validation set.

## Model And Method Notes

### RTMPose / MMPose

RTMPose is a real-time 2D pose family from OpenMMLab. It is the best candidate for the first accuracy-focused backend because it supports the detector-first, crop-around-person workflow that helps when gate skiers are small in 1080p/4K frames.

Practical drawback: a clean MMPose stack usually means coordinating `mmengine`, `mmcv`, `mmdet`, model configs, and checkpoints. That is worth doing after the MVP, not before a working overlay exists.

Sources:

- RTMPose paper: https://arxiv.org/abs/2303.07399
- MMPose RTMPose configs: https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.md
- MMPose docs: https://mmpose.readthedocs.io/

### ViTPose

ViTPose is a strong offline benchmark for difficult frames. It is useful when you need to know whether RTMPose failures are model-capacity failures or pipeline/tracking/crop failures. It is not the simplest first production backend.

Source: https://arxiv.org/abs/2204.12484

### Ultralytics YOLO Pose

YOLO pose is the pragmatic MVP choice here. It gives a simple Python API, COCO-style keypoints, and built-in tracking mode. It is not the top recommendation for peak ski accuracy, but it is good enough to produce overlay videos and JSON now.

Sources:

- Pose docs: https://docs.ultralytics.com/tasks/pose/
- Tracking docs: https://docs.ultralytics.com/modes/track/

### ByteTrack

ByteTrack is the default tracker because it keeps low-score detections in the association step, which can help through brief occlusion and low contrast. The MVP uses Ultralytics tracking with `bytetrack.yaml` when the YOLO backend is selected.

Source: https://arxiv.org/abs/2110.06864

### BoT-SORT

BoT-SORT is the next tracker to try if ByteTrack has ID switches caused by panning, camera motion, or similar people near the run. It is exposed in the UI/CLI as `botsort.yaml`.

Source: https://arxiv.org/abs/2206.14651

### MediaPipe Pose

MediaPipe Pose is included only as an experimental fallback. It is simple and already installed, but it is not ideal for distant gate footage. In this headless macOS shell it failed while creating an OpenGL context, so the verified local backend is YOLO pose.

Source: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

### AlphaPose

AlphaPose remains a possible route if ski-specific fine-tuning becomes necessary. It is not the first backend because the immediate goal is a reliable baseline and repeatable validation.

Source: https://github.com/MVIG-SJTU/AlphaPose

## Ski-Specific Validation Plan

Build a 100-200 frame validation set from the real videos in `/Users/quan/Documents/石泉/ski`.

Include:

- close, medium, and far skiers
- side, front, and back views
- turn initiation, apex, completion
- gate occlusion, pole/ski occlusion, snow spray
- shadows and low-contrast clothing
- partial out-of-frame cases
- multiple people or confusing background motion

Annotation:

- Use CVAT skeleton annotation.
- Export COCO Keypoints.
- Label person box plus visible/occluded/out-of-frame keypoints.
- Double-label 10-20% of frames to estimate human disagreement.

Metrics:

- Tracking: coverage, missed frames, ID switches, false target frames.
- Pose: OKS/AP, PCK, per-joint pixel error.
- Ski review: hip/knee/ankle/shoulder reliability split by close/medium/far and visible/occluded.

Sources:

- EPFL Ski 2DPose Dataset: https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/
- SkiTB / Tracking Skiers From the Top to the Bottom: https://openaccess.thecvf.com/content/WACV2024/html/Dunnhofer_Tracking_Skiers_From_the_Top_to_the_Bottom_WACV_2024_paper.html
- SkiTraVis: https://openaccess.thecvf.com/content/CVPR2023W/CVSports/html/Dunnhofer_Visualizing_Skiers_Trajectories_in_Monocular_Videos_CVPRW_2023_paper.html
- CVAT skeleton docs: https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/
- CVAT COCO export: https://docs.cvat.ai/docs/manual/advanced/formats/format-coco/

## User-Visible Feedback

The tool should be honest. Missing or weak joints are better than a misleading skeleton.

Current MVP statuses:

- `tracking_ok`: target selected and core-joint confidence is usable.
- `low_confidence`: target selected, but hips/knees/ankles/shoulders are weak.
- `tracking_lost`: no target selected.
- `skier_too_small`: selected box height is below the trustworthy threshold.
- `far_skier`: selected box is small enough that recall improves with high-resolution inference but pose accuracy is less trustworthy.

Future statuses to add after validation:

- `multiple_candidates`
- `possible_id_switch`
- `gate_occlusion`
- `out_of_frame`
- `left_right_swap_risk`

## Iteration Notes From Tested Videos

`乔波day2 2` is a crowded indoor clip with a very small target skier for much of the first half. Lowering confidence and increasing YOLO image size recovers many missed candidates, but it also detects unrelated people. The correct workflow for this case is `--preset far` plus a target point or initial box; otherwise the tracker can lock onto the wrong skier.

`1575_raw` has high person coverage, but gates and snow spray can hide joints. The practical first response is crop refinement plus short, labeled interpolation gaps. Longer occlusions should stay missing rather than being drawn as confident joints.

The next implementation iteration adds predicted-crop recovery and optical-flow motion inference. This addresses the common gate case where the skier track exists before the gate, the full-frame detector briefly loses the person-shaped silhouette, and a local crop still contains enough legs/torso context for pose recovery. It still cannot infer fully hidden anatomy with ground-truth accuracy; those joints must remain `inferred` or `missing`.
