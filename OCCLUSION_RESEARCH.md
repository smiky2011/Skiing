# Gate Occlusion And Wrong-Target Research

Date: 2026-05-28

## Findings

The gate-hidden frames are not just threshold failures. They are cases where a normal image pose model loses the person-shaped evidence it was trained to expect. The visible skier may be obvious to a human because humans use video continuity, ski/gate context, and body priors. A per-frame COCO-style pose detector usually does not.

The current optical-flow prediction is only a stopgap. It can bridge very short gaps, but it is not anatomically constrained and can drift into unrealistic skeletons. It should remain marked as `inferred`, never `observed`.

## Paper-Backed Direction

### 1. Use video pose, not only frame pose

PoseWarper and DCPose both support the idea that video pose should use neighboring frames and temporal cues instead of independent frame predictions.

- PoseWarper learns dense temporal pose propagation from sparse labels and reports better pose propagation than optical-flow propagation on PoseTrack.
- DCPose explicitly targets video failure modes such as motion blur, defocus, and pose occlusion by using multi-frame temporal context.

Practical implication: optical flow alone is not the right long-term solution. A better next backend is a video-pose model or a temporal refinement model over per-frame RTMPose/ViTPose outputs.

### 2. Use a target-object tracker before pose

SkiTB and SkiTraVis both treat skier localization/tracking as a prerequisite for higher-level ski analysis. That matches what we are seeing: if the skier identity is wrong, the pose overlay is irrelevant even if keypoints look plausible.

Practical implication: target selection should happen before pose. Do not let pose confidence alone decide the skier when there are other people in the frame.

### 3. Use promptable video segmentation for crowded starts

For clips like `1571_raw`, the first frame contains several people near the start. Pure automatic selection cannot know which person is the athlete unless there is a reliable external cue.

The best practical method is:

1. User selects the correct skier in one clear frame.
2. SAM 2 or XMem tracks that skier mask through the video.
3. The pose model runs only inside the tracked skier mask/box.
4. Gate and pole pixels can be excluded from the pose crop or treated as occluders.

SAM 2 is specifically built for promptable image/video segmentation with streaming memory. XMem is designed for long video object segmentation with memory stores. Both are more appropriate than nearest-box prediction for long clips with occlusion.

### 4. Gate detection is useful, but not enough alone

Detecting red/blue gate panels can define a course corridor and suppress people outside the course. This can help reject officials or background skiers at the start.

But gates do not identify the target skier by themselves. A corridor prior should be a scoring feature, not the only decision rule. If many skiers are inside or near the same course region, manual target initialization or identity tracking is still required.

## Recommended vNext Architecture

1. **Target initialization**
   - User picks one frame where the skier is visible.
   - User clicks the skier or draws a box.
   - This becomes the identity anchor.

2. **Video object tracking**
   - Preferred: SAM 2 video predictor if available.
   - Alternative: XMem / Cutie-style video object segmentation.
   - Fallback: BoT-SORT with appearance/ReID plus course-corridor scoring.

3. **Gate/course prior**
   - Detect red/blue gate panels with color + geometry as a first pass.
   - Later train a small YOLO detector for gate panels/poles.
   - Build a course corridor from gate centers.
   - Penalize person tracks outside the corridor.

4. **Pose inside tracked target**
   - Run RTMPose/ViTPose on the target mask/box crop.
   - Keep COCO body joints first.
   - Later fine-tune on EPFL Ski 2DPose and a local gate-occlusion validation set.

5. **Temporal refinement**
   - Short gaps: infer with temporal model or optical flow.
   - Longer gate occlusions: hide or label as uncertain.
   - Never render hidden joints as confident observations.

## Current Code Status

Implemented:

- `--preset occlusion`
- predicted ROI recovery
- optical-flow motion inference
- `candidate_source` in JSON
- `--target-strategy moving` experimental

The `moving` strategy is not sufficient for `1571_raw`; it can still select a wrong moving/background person. It should be treated as experimental, not the final solution.

For `1571_raw`, a better current workaround is to delay target lock until a clear frame and provide a point near the actual racer:

```bash
python3 run_overlay.py "/Users/quan/Documents/石泉/ski/1571_raw.MP4" --preset far --target-frame 500 --target-point 950,425
```

This avoids drawing a skeleton on the wrong people near the start banner before the target racer is identifiable.

## Sources

- SkiTB / Tracking Skiers From the Top to the Bottom: https://openaccess.thecvf.com/content/WACV2024/html/Dunnhofer_Tracking_Skiers_From_the_Top_to_the_Bottom_WACV_2024_paper.html
- SkiTraVis: https://openaccess.thecvf.com/content/CVPR2023W/CVSports/html/Dunnhofer_Visualizing_Skiers_Trajectories_in_Monocular_Videos_CVPRW_2023_paper.html
- EPFL Ski 2DPose Dataset: https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/
- PoseWarper / Learning Temporal Pose Estimation from Sparsely-Labeled Videos: https://arxiv.org/abs/1906.04016
- DCPose / Deep Dual Consecutive Network for Human Pose Estimation: https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Deep_Dual_Consecutive_Network_for_Human_Pose_Estimation_CVPR_2021_paper.html
- ByteTrack: https://arxiv.org/abs/2110.06864
- BoT-SORT: https://arxiv.org/abs/2206.14651
- SAM 2: https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/
- XMem: https://arxiv.org/abs/2207.07115
