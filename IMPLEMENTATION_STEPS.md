# Alpine Ski Skeleton Detection: Implementation Steps

## 1. Product Scope

Build the first version around one focused workflow:

> Upload alpine skiing gate video -> detect and track the skier -> detect skeleton -> generate overlay video.

Do not start with AI coaching, trajectory scoring, line choice, or full performance analysis. Those features depend on skeleton quality, so they should come later.

The initial goal is accurate, trustworthy 2D skeleton overlay for gate-skiing footage.

## 2. Core Principle

Use the strongest simple pipeline:

1. One skier detector/tracker.
2. One pose estimation model.
3. One temporal smoothing layer.
4. Confidence-aware rendering.

Avoid combining many pose models or fallback systems unless validation proves they improve accuracy.

## 3. Target Input

Prioritize these videos first:

- Fixed-camera or lightly panning gate videos.
- One main skier.
- Full body visible for most of the run.
- 1080p or 4K footage.
- Skier large enough to identify hips, knees, ankles, and shoulders.

Treat these as difficult or later-stage cases:

- POV/body-mounted footage.
- Very distant skier.
- Heavy gate, fence, pole, or snow-spray occlusion.
- Multiple similar skiers.
- Long out-of-frame sections.

## 4. Recommended Technical Pipeline

### Step 1: Video Ingestion

- Accept MP4/MOV files.
- Read metadata: width, height, fps, frame count, duration.
- Keep original resolution available for detection.
- Optionally create a lower-resolution preview for UI and progress display.

### Step 2: Skier Detection

Use a person/skier detector to locate candidate boxes in each frame.

Recommended first candidates:

- RTMDet person detector through MMPose/MMDetection.
- YOLOX or YOLO family detector if custom fine-tuning becomes useful.

Important detail: for 4K gate videos, avoid resizing the whole frame too aggressively. A far skier may disappear if a 4K frame is reduced to 640px before detection.

### Step 3: Skier Tracking

Track the same skier box across frames.

Recommended first choice:

- ByteTrack.

Use BoT-SORT only if ByteTrack has too many ID switches because of camera motion, panning, zooming, or confusing background people.

Output per frame:

- `track_id`
- bounding box
- detection confidence
- tracking confidence/status

### Step 4: Crop Around Skier

For each tracked frame:

- Expand the skier bounding box with margin.
- Crop the skier region.
- Run pose detection on the crop, not only the full frame.
- Map predicted keypoints back to original video coordinates.

This is important because gate skiers are often small in the full frame.

### Step 5: Pose Estimation

Primary model candidate:

- RTMPose-L through MMPose.

Benchmark candidates:

- ViTPose-L/H for hard frames.
- AlphaPose if ski-specific fine-tuning becomes important.

Avoid starting with:

- MediaPipe Pose as the main system. It is strong for close single-person fitness-style footage, but not ideal for distant gate-skiing.
- OpenPose as the main system. It is older and heavier.

Initial keypoints should use COCO-style human joints:

- head/nose/ears/eyes if useful
- shoulders
- elbows
- wrists
- hips
- knees
- ankles

Later ski-specific keypoints can include:

- ski tips
- ski tails
- boot centers
- pole tips/baskets

Do not add ski-specific keypoints until the body skeleton pipeline is reliable.

### Step 6: Confidence Gating

Every joint should keep:

- x coordinate
- y coordinate
- confidence score
- observed/inferred/missing status

Rendering rule:

- Draw confident joints clearly.
- Fade uncertain joints.
- Hide very low-confidence joints.
- Do not draw limbs when one or both endpoint joints are unreliable.

The overlay should be honest. A missing joint is better than a fake confident joint.

### Step 7: Temporal Smoothing

Apply smoothing only after pose detection.

Recommended:

- One Euro Filter per joint.
- Or MMPose temporal smoother if already using MMPose.

Rules:

- Smooth confident observations.
- Allow very short interpolation gaps, for example 1-3 frames.
- Mark interpolated joints as inferred.
- Do not aggressively fill long occlusions.
- Reset or fade after longer tracking loss.

### Step 8: Overlay Video Generation

Generate an output MP4 with:

- Original video frame.
- Skier bounding box, optional.
- Skeleton lines and joints.
- Confidence-aware color/fade.
- Optional frame-level status: tracking OK, low confidence, tracking lost.

Suggested visual language:

- Green: confident.
- Yellow: uncertain.
- Red or hidden: unreliable/lost.
- Dashed/faded: short-gap inferred.

## 5. Accuracy Validation

Do not judge only by visual demos. Build a small ski-specific validation set.

### Validation Dataset

Start with 100-200 frames selected from your real videos.

Include:

- close skier
- medium skier
- far skier
- side view
- front/back view
- turn initiation
- apex
- turn completion
- gate occlusion
- pole/ski occlusion
- snow spray
- shadows/flat light
- skier partially out of frame

Use videos that are not used for tuning when possible.

### Annotation Tool

Recommended:

- CVAT skeleton annotation.
- Export as COCO Keypoints.

Label:

- person bounding box
- visible keypoints
- occluded but inferable keypoints
- outside/out-of-frame keypoints

Double-label 10-20% of frames to estimate human annotation disagreement.

### Metrics

Use three metric groups:

1. Tracking metrics:
   - track coverage
   - missed frames
   - ID switches
   - false target tracking

2. Pose metrics:
   - COCO OKS/AP
   - PCK at practical thresholds
   - per-joint pixel error

3. Ski-specific review metrics:
   - knee reliability
   - ankle reliability
   - hip reliability
   - shoulder reliability
   - close vs far skier performance
   - visible vs occluded joint performance

For the first version, prioritize hips, knees, ankles, and shoulders over face or hand details.

## 6. Model Selection Process

Run the same validation set through each candidate.

Recommended order:

1. RTMPose-L + ByteTrack.
2. RTMPose-L + BoT-SORT if tracking is unstable.
3. ViTPose-L/H as an offline benchmark on hard frames.
4. YOLO-pose L/X only if simplicity/export/custom training matters more than peak accuracy.
5. AlphaPose if ski-specific fine-tuning becomes necessary.

Keep only the simplest system that wins on the validation set.

Do not combine models unless there is a clear measured gain.

## 7. Expected Failure Cases

The system should explicitly handle:

- skier too small
- skier partially hidden by gate
- skier hidden by fence or snow spray
- crossed legs
- strong shadows
- low contrast clothing
- athlete leaving frame
- camera panning quickly
- another person closer to camera
- model swapping left/right limbs

For each failure case, the system should lower confidence or hide joints instead of drawing a misleading skeleton.

## 8. Milestones

### Milestone 1: Offline Prototype

Input:

- one local video file

Output:

- overlay MP4
- raw JSON keypoints per frame

Success:

- skeleton appears on clear close/medium skier frames
- low-confidence frames are marked honestly

### Milestone 2: Tracking Reliability

Add:

- skier ID tracking
- user-selected initial skier if multiple candidates exist
- tracking-lost state

Success:

- same skier is tracked across most of the run
- fewer false jumps to background people or objects

### Milestone 3: Validation Set

Add:

- labeled ski frames
- repeatable evaluation script
- metrics report

Success:

- model changes are judged by data, not impressions

### Milestone 4: Accuracy Improvement

Add only if validation shows need:

- better detector
- BoT-SORT instead of ByteTrack
- ViTPose benchmark
- ski-specific fine-tuning

Success:

- measurable improvement on difficult gate footage

### Milestone 5: Simple User Workflow

Add:

- upload UI
- progress display
- overlay preview
- download output video

Success:

- a user can upload a ski video and receive a skeleton-overlay video without touching technical settings

## 9. Recommended References

- EPFL Ski 2DPose Dataset: https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/
- SkiTB / Tracking Skiers From the Top to the Bottom: https://openaccess.thecvf.com/content/WACV2024/html/Dunnhofer_Tracking_Skiers_From_the_Top_to_the_Bottom_WACV_2024_paper.html
- SkiTraVis: https://openaccess.thecvf.com/content/CVPR2023W/CVSports/html/Dunnhofer_Visualizing_Skiers_Trajectories_in_Monocular_Videos_CVPRW_2023_paper.html
- RTMPose: https://arxiv.org/abs/2303.07399
- MMPose RTMPose configs: https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.md
- ViTPose: https://arxiv.org/abs/2204.12484
- ByteTrack: https://arxiv.org/abs/2110.06864
- BoT-SORT: https://arxiv.org/abs/2206.14651
- CVAT Skeleton Annotation: https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/

## 10. Final Direction

The best first version is not a coaching system.

It is:

> A high-accuracy, confidence-aware alpine skiing skeleton overlay generator for gate videos.

Build trust first. Once skeleton detection is reliable, trajectory analysis, coaching feedback, and athlete comparison can be added on top.
