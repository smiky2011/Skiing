from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

CORE_JOINTS = [5, 6, 11, 12, 13, 14, 15, 16]

MEDIAPIPE_TO_COCO = {
    0: 0,
    2: 1,
    5: 2,
    7: 3,
    8: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}


@dataclass
class ProcessingSettings:
    backend: str = "auto"
    model: str = "yolo11n-pose.pt"
    tracker: str = "bytetrack.yaml"
    imgsz: int = 1280
    conf: float = 0.25
    device: str = ""
    start_frame: int = 0
    max_frames: int | None = None
    draw_box: bool = True
    draw_labels: bool = True
    smooth: bool = True
    min_draw_conf: float = 0.20
    confident_conf: float = 0.50
    observe_conf: float = 0.15
    max_infer_gap: int = 3
    initial_box: tuple[float, float, float, float] | None = None
    target_point: tuple[float, float] | None = None
    target_frame: int = 0
    target_region: tuple[float, float, float, float] | None = None
    target_strategy: str = "first"
    target_lock_delay_frames: int = 0
    provisional_target: bool = False
    target_min_frames: int = 5
    target_min_displacement_px: float = 65.0
    target_min_downhill_px: float = 20.0
    target_min_downhill_ratio: float = 0.0
    target_min_downhill_rate_px: float = 0.0
    target_max_centerline_distance_ratio: float = 1.0
    target_min_track_confidence: float = 0.0
    target_min_region_progress: float = 0.0
    refine_pose: bool = False
    crop_margin: float = 0.50
    crop_imgsz: int = 960
    too_small_height_ratio: float = 0.08
    far_skier_height_ratio: float = 0.14
    max_reacquire_distance_ratio: float = 0.18
    roi_recovery: bool = True
    motion_inference: bool = True


@dataclass
class ProcessingResult:
    output_video: str
    keypoints_json: str
    report_json: str
    backend: str
    frames_processed: int
    frames_with_pose: int
    low_confidence_frames: int
    tracking_lost_frames: int
    elapsed_sec: float


@dataclass
class CandidatePose:
    bbox: tuple[float, float, float, float] | None
    bbox_conf: float
    track_id: int | None
    keypoints: np.ndarray
    scores: np.ndarray
    source: str = "detector"


@dataclass
class TrackCandidateStats:
    first_center: tuple[float, float]
    last_center: tuple[float, float]
    frames_seen: int = 1
    max_confidence: float = 0.0
    last_frame: int = 0
    path_length: float = 0.0
    downhill_steps: int = 0

    def update(self, center: tuple[float, float], confidence: float, frame_index: int) -> None:
        dx = center[0] - self.last_center[0]
        dy = center[1] - self.last_center[1]
        self.path_length += math.hypot(dx, dy)
        if dy > 0:
            self.downhill_steps += 1
        self.last_center = center
        self.frames_seen += 1
        self.max_confidence = max(self.max_confidence, float(confidence))
        self.last_frame = frame_index

    @property
    def displacement(self) -> float:
        return math.hypot(
            self.last_center[0] - self.first_center[0],
            self.last_center[1] - self.first_center[1],
        )

    @property
    def downhill(self) -> float:
        return self.last_center[1] - self.first_center[1]

    @property
    def downhill_ratio(self) -> float:
        return self.downhill / max(self.displacement, 1.0)

    @property
    def downhill_rate(self) -> float:
        return self.downhill / max(self.frames_seen - 1, 1)

    @property
    def downhill_consistency(self) -> float:
        return self.downhill_steps / max(self.frames_seen - 1, 1)


class LowPassFilter:
    def __init__(self) -> None:
        self.prev: float | None = None

    def __call__(self, value: float, alpha: float) -> float:
        if self.prev is None:
            self.prev = value
            return value
        smoothed = alpha * value + (1.0 - alpha) * self.prev
        self.prev = smoothed
        return smoothed


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
        freq: float = 30.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: float | None = None

    @staticmethod
    def alpha(freq: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, value: float, timestamp: float) -> float:
        if self.last_time is not None:
            dt = max(timestamp - self.last_time, 1e-6)
            self.freq = 1.0 / dt
        self.last_time = timestamp

        prev = self.x_filter.prev
        dx = 0.0 if prev is None else (value - prev) * self.freq
        edx = self.dx_filter(dx, self.alpha(self.freq, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter(value, self.alpha(self.freq, cutoff))


class PoseSmoother:
    def __init__(self, settings: ProcessingSettings, fps: float) -> None:
        self.settings = settings
        self.filters = [
            (OneEuroFilter(freq=fps), OneEuroFilter(freq=fps)) for _ in COCO_KEYPOINTS
        ]
        self.last_points: list[tuple[float, float] | None] = [None] * len(COCO_KEYPOINTS)
        self.last_scores = [0.0] * len(COCO_KEYPOINTS)
        self.gaps = [settings.max_infer_gap + 1] * len(COCO_KEYPOINTS)

    def update(
        self, keypoints: np.ndarray, scores: np.ndarray, timestamp: float
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        out_points = np.full((len(COCO_KEYPOINTS), 2), np.nan, dtype=np.float32)
        out_scores = np.zeros(len(COCO_KEYPOINTS), dtype=np.float32)
        statuses = ["missing"] * len(COCO_KEYPOINTS)

        for idx, (point, score) in enumerate(zip(keypoints, scores, strict=True)):
            x, y = float(point[0]), float(point[1])
            observed = (
                np.isfinite(x)
                and np.isfinite(y)
                and float(score) >= self.settings.observe_conf
            )

            if observed:
                fx, fy = self.filters[idx]
                sx = fx(x, timestamp)
                sy = fy(y, timestamp)
                out_points[idx] = (sx, sy)
                out_scores[idx] = float(score)
                statuses[idx] = "observed"
                self.last_points[idx] = (sx, sy)
                self.last_scores[idx] = float(score)
                self.gaps[idx] = 0
                continue

            if self.last_points[idx] is not None and self.gaps[idx] < self.settings.max_infer_gap:
                self.gaps[idx] += 1
                lx, ly = self.last_points[idx]
                out_points[idx] = (lx, ly)
                out_scores[idx] = max(0.0, self.last_scores[idx] * (0.65**self.gaps[idx]))
                statuses[idx] = "inferred"
                continue

            self.gaps[idx] = self.settings.max_infer_gap + 1

        return out_points, out_scores, statuses


class YoloPoseBackend:
    name = "yolo"

    def __init__(self, settings: ProcessingSettings) -> None:
        from ultralytics import YOLO

        self.settings = settings
        self.model = YOLO(settings.model)
        self.refine_model = YOLO(settings.model) if settings.refine_pose else None

    def infer(self, frame: np.ndarray) -> list[CandidatePose]:
        kwargs: dict[str, Any] = {
            "conf": self.settings.conf,
            "imgsz": self.settings.imgsz,
            "persist": True,
            "tracker": self.settings.tracker,
            "verbose": False,
        }
        if self.settings.device:
            kwargs["device"] = self.settings.device

        results = self.model.track(frame, **kwargs)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        ids = (
            boxes.id.detach().cpu().numpy().astype(int).tolist()
            if boxes.id is not None
            else [None] * len(xyxy)
        )
        kpt_data = keypoints.data.detach().cpu().numpy()

        candidates: list[CandidatePose] = []
        for idx, box in enumerate(xyxy):
            data = kpt_data[idx]
            points = data[:, :2].astype(np.float32)
            scores = (
                data[:, 2].astype(np.float32)
                if data.shape[1] >= 3
                else np.ones(len(COCO_KEYPOINTS), dtype=np.float32)
            )
            candidates.append(
                CandidatePose(
                    bbox=tuple(float(v) for v in box),
                    bbox_conf=float(confs[idx]),
                    track_id=ids[idx],
                    keypoints=points,
                    scores=scores,
                    source="detector",
                )
            )
        return candidates

    def refine(self, frame: np.ndarray, candidate: CandidatePose) -> CandidatePose:
        if self.refine_model is None or candidate.bbox is None:
            return candidate

        crop, offset = crop_around_box(frame, candidate.bbox, self.settings.crop_margin)
        if crop is None:
            return candidate

        kwargs: dict[str, Any] = {
            "conf": max(0.02, min(self.settings.conf, 0.15)),
            "imgsz": self.settings.crop_imgsz,
            "verbose": False,
        }
        if self.settings.device:
            kwargs["device"] = self.settings.device

        results = self.refine_model.predict(crop, **kwargs)
        if not results:
            return candidate

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            return candidate

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        kpt_data = keypoints.data.detach().cpu().numpy()
        crop_center = np.array([crop.shape[1] / 2.0, crop.shape[0] / 2.0], dtype=np.float32)

        best_idx = 0
        best_score = -1.0
        for idx, box in enumerate(xyxy):
            center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)
            center_dist = float(np.linalg.norm(center - crop_center))
            center_score = 1.0 - min(1.0, center_dist / max(crop.shape[0], crop.shape[1], 1))
            scores = kpt_data[idx][:, 2] if kpt_data[idx].shape[1] >= 3 else np.ones(len(COCO_KEYPOINTS))
            score = float(confs[idx]) + 0.35 * mean_confidence(scores, CORE_JOINTS) + 0.35 * center_score
            if score > best_score:
                best_score = score
                best_idx = idx

        ox, oy = offset
        data = kpt_data[best_idx]
        points = data[:, :2].astype(np.float32)
        points[:, 0] += ox
        points[:, 1] += oy
        scores = (
            data[:, 2].astype(np.float32)
            if data.shape[1] >= 3
            else np.ones(len(COCO_KEYPOINTS), dtype=np.float32)
        )
        box = xyxy[best_idx].astype(np.float32)
        box[[0, 2]] += ox
        box[[1, 3]] += oy

        original_core = mean_confidence(candidate.scores, CORE_JOINTS)
        refined_core = mean_confidence(scores, CORE_JOINTS)
        if refined_core + 0.03 < original_core and candidate.bbox_conf >= float(confs[best_idx]):
            return candidate

        return CandidatePose(
            bbox=tuple(float(v) for v in box),
            bbox_conf=max(candidate.bbox_conf, float(confs[best_idx])),
            track_id=candidate.track_id,
            keypoints=points,
            scores=scores,
            source="crop_refine",
        )

    def recover_from_roi(
        self,
        frame: np.ndarray,
        roi_box: tuple[float, float, float, float],
        track_id: int | None,
    ) -> CandidatePose | None:
        if self.refine_model is None:
            return None

        crop, offset = crop_around_box(frame, roi_box, max(self.settings.crop_margin, 0.85))
        if crop is None:
            return None

        kwargs: dict[str, Any] = {
            "conf": max(0.01, min(self.settings.conf, 0.10)),
            "imgsz": max(self.settings.crop_imgsz, 1280),
            "verbose": False,
        }
        if self.settings.device:
            kwargs["device"] = self.settings.device

        results = self.refine_model.predict(crop, **kwargs)
        if not results:
            return None

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        kpt_data = keypoints.data.detach().cpu().numpy()
        expected_center = np.array(
            [
                (roi_box[0] + roi_box[2]) / 2.0 - offset[0],
                (roi_box[1] + roi_box[3]) / 2.0 - offset[1],
            ],
            dtype=np.float32,
        )

        best_idx = -1
        best_score = -1.0
        for idx, box in enumerate(xyxy):
            center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)
            center_dist = float(np.linalg.norm(center - expected_center))
            center_score = 1.0 - min(1.0, center_dist / max(crop.shape[0], crop.shape[1], 1))
            scores = kpt_data[idx][:, 2] if kpt_data[idx].shape[1] >= 3 else np.ones(len(COCO_KEYPOINTS))
            core_conf = mean_confidence(scores, CORE_JOINTS)
            score = 0.55 * float(confs[idx]) + 0.35 * core_conf + 0.65 * center_score
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            return None

        ox, oy = offset
        data = kpt_data[best_idx]
        points = data[:, :2].astype(np.float32)
        points[:, 0] += ox
        points[:, 1] += oy
        scores = (
            data[:, 2].astype(np.float32)
            if data.shape[1] >= 3
            else np.ones(len(COCO_KEYPOINTS), dtype=np.float32)
        )
        if mean_confidence(scores, CORE_JOINTS) < max(0.04, self.settings.observe_conf * 0.6):
            return None

        box = xyxy[best_idx].astype(np.float32)
        box[[0, 2]] += ox
        box[[1, 3]] += oy
        return CandidatePose(
            bbox=tuple(float(v) for v in box),
            bbox_conf=float(confs[best_idx]),
            track_id=track_id,
            keypoints=points,
            scores=scores,
            source="roi_recovery",
        )


class MediaPipePoseBackend:
    name = "mediapipe"

    def __init__(self, settings: ProcessingSettings) -> None:
        import mediapipe as mp

        self.settings = settings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=max(0.1, settings.conf),
            min_tracking_confidence=max(0.1, settings.conf),
        )

    def infer(self, frame: np.ndarray) -> list[CandidatePose]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return []

        height, width = frame.shape[:2]
        points = np.full((len(COCO_KEYPOINTS), 2), np.nan, dtype=np.float32)
        scores = np.zeros(len(COCO_KEYPOINTS), dtype=np.float32)
        for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
            landmark = result.pose_landmarks.landmark[mp_idx]
            points[coco_idx] = (landmark.x * width, landmark.y * height)
            scores[coco_idx] = float(getattr(landmark, "visibility", 0.0))

        visible = points[scores >= self.settings.observe_conf]
        bbox = None
        bbox_conf = float(np.mean(scores[CORE_JOINTS])) if len(scores) else 0.0
        if len(visible):
            x1, y1 = np.nanmin(visible, axis=0)
            x2, y2 = np.nanmax(visible, axis=0)
            pad = max(20.0, 0.12 * max(x2 - x1, y2 - y1))
            bbox = (
                max(0.0, float(x1 - pad)),
                max(0.0, float(y1 - pad)),
                min(float(width - 1), float(x2 + pad)),
                min(float(height - 1), float(y2 + pad)),
            )

        return [
            CandidatePose(
                bbox=bbox,
                bbox_conf=bbox_conf,
                track_id=1,
                keypoints=points,
                scores=scores,
                source="detector",
            )
        ]

    def close(self) -> None:
        self.pose.close()


def process_video(
    video_path: str | Path,
    output_dir: str | Path = "outputs",
    settings: ProcessingSettings | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ProcessingResult:
    settings = settings or ProcessingSettings()
    video_path = Path(video_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count and fps else None
    available_frames = max(0, frame_count - max(0, settings.start_frame))
    limit = min(available_frames, settings.max_frames) if settings.max_frames else available_frames
    if settings.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, settings.start_frame)

    backend = build_backend(settings)
    backend_name = backend.name
    smoother = PoseSmoother(settings, fps) if settings.smooth else None

    output_video = output_dir / f"{video_path.stem}_skeleton_overlay.mp4"
    writer = make_writer(output_video, fps, width, height)

    start = time.time()
    active_track_id: int | None = None
    pending_track_stats: dict[int, TrackCandidateStats] = {}
    prev_bbox: tuple[float, float, float, float] | None = None
    bbox_velocity: tuple[float, float, float, float] | None = None
    prev_gray: np.ndarray | None = None
    last_keypoints: np.ndarray | None = None
    last_scores: np.ndarray | None = None
    last_statuses: list[str] | None = None
    tracking_lost_frames = 0
    low_confidence_frames = 0
    frames_with_pose = 0
    frames_out: list[dict[str, Any]] = []
    metadata = {
        "input_video": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "start_frame": settings.start_frame,
    }

    frame_index = 0
    try:
        while True:
            if settings.max_frames is not None and frame_index >= settings.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            actual_frame_index = settings.start_frame + frame_index
            timestamp = actual_frame_index / fps if fps else 0.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_candidates = backend.infer(frame)
            raw_candidate_count = len(raw_candidates)
            sized_candidates = reject_too_small_candidates(raw_candidates, settings, height)
            target_candidates = filter_candidates_by_region(
                sized_candidates, settings.target_region, width, height
            )
            update_track_candidate_stats(
                pending_track_stats, target_candidates, actual_frame_index
            )
            candidates = target_candidates if active_track_id is None else sized_candidates
            candidate_count = len(candidates)
            corridor_candidate = nearest_candidate_to_corridor_centerline(
                target_candidates, settings.target_region, width, height
            )
            selected: CandidatePose | None = None
            selected_is_provisional = False
            selected_is_reacquired = False
            selection_block_reason = (
                "waiting_for_target_hint"
                if (
                    active_track_id is None
                    and settings.target_point is not None
                    and actual_frame_index < settings.target_frame
                )
                else None
            )
            if should_delay_target_lock(settings, active_track_id, actual_frame_index):
                moving_candidate = None
                if actual_frame_index - settings.start_frame >= settings.target_lock_delay_frames:
                    moving_candidate = choose_moving_candidate(
                        target_candidates, pending_track_stats, settings, width, height
                    )
                if moving_candidate is not None:
                    active_track_id = moving_candidate.track_id
                    selected = moving_candidate
                elif settings.provisional_target:
                    selected = choose_moving_candidate(
                        target_candidates, pending_track_stats, settings, width, height
                    )
                    if selected is None:
                        candidates = []
                    else:
                        selected_is_provisional = True
                else:
                    candidates = []
            elif (
                settings.provisional_target
                and active_track_id is None
                and settings.target_point is not None
                and actual_frame_index < settings.target_frame
            ):
                selected = choose_moving_candidate(
                    target_candidates, pending_track_stats, settings, width, height
                )
                selected_is_provisional = selected is not None

            if selected is None:
                selected = select_candidate(
                    candidates,
                    active_track_id,
                    prev_bbox,
                    settings.initial_box if actual_frame_index == settings.start_frame else None,
                    settings.target_point,
                    settings.target_frame,
                    actual_frame_index,
                    settings.max_reacquire_distance_ratio,
                    width,
                    height,
                    allow_new_track=not should_gate_new_track(settings, active_track_id),
                )
            if selected is None and should_gate_new_track(settings, active_track_id):
                selected = choose_moving_candidate(
                    target_candidates, pending_track_stats, settings, width, height
                )
                selected_is_reacquired = selected is not None
            predicted_box = predict_bbox(prev_bbox, bbox_velocity, width, height)
            if (
                selected is None
                and settings.roi_recovery
                and predicted_box is not None
                and not is_too_small_bbox(predicted_box, settings, height)
                and hasattr(backend, "recover_from_roi")
            ):
                selected = backend.recover_from_roi(frame, predicted_box, active_track_id)

            warnings: list[str] = []
            raw_keypoints = np.full((len(COCO_KEYPOINTS), 2), np.nan, dtype=np.float32)
            raw_scores = np.zeros(len(COCO_KEYPOINTS), dtype=np.float32)
            keypoints = raw_keypoints.copy()
            scores = raw_scores.copy()
            statuses = ["missing"] * len(COCO_KEYPOINTS)
            bbox = None
            bbox_conf = 0.0
            track_id = active_track_id
            candidate_source = "none"
            frame_status = "tracking_lost" if prev_bbox is not None else "no_detection"

            if selected is not None:
                if selected.source == "roi_recovery":
                    warnings.append("roi_recovery")
                elif hasattr(backend, "refine"):
                    selected = backend.refine(frame, selected)
                candidate_source = selected.source
                raw_keypoints = selected.keypoints
                raw_scores = selected.scores
                bbox = selected.bbox
                bbox_conf = selected.bbox_conf
                if bbox is not None and is_too_small_bbox(bbox, settings, height):
                    selected = None
                    warnings.append("skier_too_small_rejected")
                if (
                    selected is not None
                    and should_gate_new_track(settings, active_track_id)
                    and selected.track_id is not None
                    and selected.track_id != active_track_id
                    and not track_passes_moving_gate(
                        selected.track_id, pending_track_stats, settings, width, height
                    )
                ):
                    selected = None
                    warnings.append("relock_candidate_rejected")
                if selected is None:
                    bbox = None
                    bbox_conf = 0.0
                    track_id = active_track_id
                    candidate_source = "none"
                    frame_status = "tracking_lost" if prev_bbox is not None else "no_detection"
                    tracking_lost_frames += 1
                    warnings.append("tracking_lost")
                    propagated = None
                    if (
                        settings.motion_inference
                        and prev_gray is not None
                        and last_keypoints is not None
                        and last_scores is not None
                        and last_statuses is not None
                        and is_usable_motion_box(predicted_box, settings, height)
                    ):
                        propagated = propagate_keypoints(
                            prev_gray,
                            gray,
                            last_keypoints,
                            last_scores,
                            last_statuses,
                            settings,
                        )
                    if propagated is not None:
                        raw_keypoints, raw_scores = propagated
                        if smoother is not None:
                            keypoints, scores, statuses = smoother.update(raw_keypoints, raw_scores, timestamp)
                        else:
                            keypoints, scores = raw_keypoints, raw_scores
                        statuses = [
                            "inferred" if float(score) >= settings.min_draw_conf else "missing"
                            for score in scores
                        ]
                        candidate_source = "motion_inference"
                        warnings.append("motion_inferred")
                        if predicted_box is not None:
                            bbox = predicted_box
                            bbox_conf = 0.0
                    elif smoother is not None:
                        keypoints, scores, statuses = smoother.update(raw_keypoints, raw_scores, timestamp)
                else:
                    track_id = selected.track_id if selected.track_id is not None else active_track_id
                    if not selected_is_provisional:
                        active_track_id = track_id
                    if bbox is not None and not selected_is_provisional:
                        if prev_bbox is not None:
                            bbox_velocity = tuple(float(bbox[i] - prev_bbox[i]) for i in range(4))
                        prev_bbox = bbox
                    frames_with_pose += 1

                    if selected_is_provisional:
                        keypoints = raw_keypoints
                        scores = raw_scores
                        statuses = [
                            "observed" if float(score) >= settings.observe_conf else "missing"
                            for score in scores
                        ]
                        candidate_source = f"{selected.source}_provisional"
                        warnings.append("provisional_target")
                    elif selected_is_reacquired:
                        if smoother is not None:
                            keypoints, scores, statuses = smoother.update(raw_keypoints, raw_scores, timestamp)
                        else:
                            keypoints = raw_keypoints
                            scores = raw_scores
                            statuses = [
                                "observed" if float(score) >= settings.observe_conf else "missing"
                                for score in scores
                            ]
                        warnings.append("racer_reacquired")
                    elif smoother is not None:
                        keypoints, scores, statuses = smoother.update(raw_keypoints, raw_scores, timestamp)
                    else:
                        keypoints = raw_keypoints
                        scores = raw_scores
                        statuses = [
                            "observed" if float(score) >= settings.observe_conf else "missing"
                            for score in scores
                        ]

                    core_conf = mean_confidence(scores, CORE_JOINTS)
                    if bbox is not None:
                        box_height = bbox[3] - bbox[1]
                        if box_height < settings.far_skier_height_ratio * height:
                            warnings.append("far_skier")
                    if core_conf < settings.min_draw_conf:
                        frame_status = "low_confidence"
                        low_confidence_frames += 1
                    elif selected_is_provisional:
                        frame_status = "provisional_target"
                    else:
                        frame_status = "tracking_ok"
            else:
                tracking_lost_frames += 1
                warnings.append("tracking_lost")
                propagated = None
                if (
                    settings.motion_inference
                    and prev_gray is not None
                    and last_keypoints is not None
                    and last_scores is not None
                    and last_statuses is not None
                    and is_usable_motion_box(predicted_box, settings, height)
                ):
                    propagated = propagate_keypoints(
                        prev_gray,
                        gray,
                        last_keypoints,
                        last_scores,
                        last_statuses,
                        settings,
                    )
                if propagated is not None:
                    raw_keypoints, raw_scores = propagated
                    if smoother is not None:
                        keypoints, scores, statuses = smoother.update(raw_keypoints, raw_scores, timestamp)
                    else:
                        keypoints, scores = raw_keypoints, raw_scores
                    statuses = [
                        "inferred" if float(score) >= settings.min_draw_conf else "missing"
                        for score in scores
                    ]
                    candidate_source = "motion_inference"
                    warnings.append("motion_inferred")
                    if predicted_box is not None:
                        bbox = predicted_box
                        bbox_conf = 0.0
                elif smoother is not None:
                    keypoints, scores, statuses = smoother.update(raw_keypoints, raw_scores, timestamp)

            overlay = render_overlay(
                frame,
                keypoints,
                scores,
                statuses,
                bbox,
                bbox_conf,
                track_id,
                actual_frame_index,
                frame_status,
                warnings,
                settings,
            )
            writer.write(overlay)
            prev_gray = gray
            if not selected_is_provisional:
                last_keypoints = keypoints.copy()
                last_scores = scores.copy()
                last_statuses = list(statuses)
            frames_out.append(
                frame_record(
                    actual_frame_index,
                    timestamp,
                    track_id,
                    bbox,
                    bbox_conf,
                    candidate_source,
                    frame_status,
                    warnings,
                    raw_candidate_count,
                    candidate_count,
                    selection_block_reason,
                    corridor_candidate,
                    width,
                    height,
                    raw_keypoints,
                    raw_scores,
                    keypoints,
                    scores,
                    statuses,
                )
            )

            frame_index += 1
            if progress_callback:
                progress_callback(frame_index, limit or frame_count or frame_index)
    finally:
        cap.release()
        writer.release()
        if hasattr(backend, "close"):
            backend.close()

    # OpenCV writes mp4v, which Chromium-based players (VS Code video
    # extensions, browsers) cannot decode. Transcode to H.264 so the overlay
    # is reviewable everywhere; best-effort, leaves mp4v in place if ffmpeg
    # is unavailable.
    transcode_to_h264(output_video)

    elapsed = time.time() - start
    result = ProcessingResult(
        output_video=str(output_video),
        keypoints_json=str(output_dir / f"{video_path.stem}_keypoints.json"),
        report_json=str(output_dir / f"{video_path.stem}_report.json"),
        backend=backend_name,
        frames_processed=frame_index,
        frames_with_pose=frames_with_pose,
        low_confidence_frames=low_confidence_frames,
        tracking_lost_frames=tracking_lost_frames,
        elapsed_sec=elapsed,
    )
    report = {
        "metadata": metadata,
        "settings": asdict(settings),
        "result": asdict(result),
        "summary": {
            "pose_coverage": safe_ratio(frames_with_pose, frame_index),
            "low_confidence_rate": safe_ratio(low_confidence_frames, frame_index),
            "tracking_lost_rate": safe_ratio(tracking_lost_frames, frame_index),
        },
        "diagnostics": summarize_frame_diagnostics(frames_out),
    }
    keypoint_payload = {
        "metadata": metadata,
        "settings": asdict(settings),
        "backend": backend_name,
        "keypoint_format": "COCO-17",
        "keypoint_names": COCO_KEYPOINTS,
        "frames": frames_out,
    }

    write_json(Path(result.keypoints_json), keypoint_payload)
    write_json(Path(result.report_json), report)
    return result


def build_backend(settings: ProcessingSettings) -> Any:
    backend = settings.backend.lower()
    if backend == "auto":
        try:
            return YoloPoseBackend(settings)
        except Exception as exc:
            print(f"YOLO backend unavailable; falling back to MediaPipe. Reason: {exc}")
            return MediaPipePoseBackend(settings)
    if backend == "yolo":
        return YoloPoseBackend(settings)
    if backend == "mediapipe":
        return MediaPipePoseBackend(settings)
    raise ValueError(f"Unknown backend '{settings.backend}'. Use auto, yolo, or mediapipe.")


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for fourcc_name in ("mp4v", "avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not create MP4 writer with OpenCV. Try installing ffmpeg.")


def transcode_to_h264(path: Path) -> bool:
    """Re-encode an OpenCV-written MP4 (OpenCV uses mp4v on macOS) to H.264
    yuv420p in place, so the file plays in Chromium-based players such as the
    VS Code video extensions and web browsers. mp4v is decodable by QuickTime
    but not by Chromium. Transcoding is best-effort: if ffmpeg is missing or
    fails, the original mp4v file is left untouched. Returns True on success.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    tmp = path.with_name(path.stem + "_h264tmp.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "fast",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError):
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


def should_delay_target_lock(
    settings: ProcessingSettings, active_track_id: int | None, frame_index: int
) -> bool:
    return (
        settings.target_strategy == "moving"
        and active_track_id is None
        and settings.initial_box is None
        and settings.target_point is None
        and frame_index >= settings.start_frame
    )


def update_track_candidate_stats(
    stats: dict[int, TrackCandidateStats], candidates: list[CandidatePose], frame_index: int
) -> None:
    for candidate in candidates:
        if candidate.track_id is None or candidate.bbox is None:
            continue
        center = bbox_center(candidate.bbox)
        if candidate.track_id not in stats:
            stats[candidate.track_id] = TrackCandidateStats(
                first_center=center,
                last_center=center,
                max_confidence=float(candidate.bbox_conf),
                last_frame=frame_index,
            )
        else:
            stats[candidate.track_id].update(center, candidate.bbox_conf, frame_index)


def choose_moving_track(
    stats: dict[int, TrackCandidateStats], settings: ProcessingSettings, width: int, height: int
) -> int | None:
    best_id: int | None = None
    best_score = -1.0
    for track_id, item in stats.items():
        if item.frames_seen < settings.target_min_frames:
            continue
        if item.max_confidence < settings.target_min_track_confidence:
            continue
        if item.displacement < settings.target_min_displacement_px:
            continue
        if item.downhill < settings.target_min_downhill_px:
            continue
        if item.downhill_ratio < settings.target_min_downhill_ratio:
            continue
        if item.downhill_rate < settings.target_min_downhill_rate_px:
            continue
        centerline_distance = track_centerline_distance(item, settings.target_region, width, height)
        max_centerline_distance = settings.target_max_centerline_distance_ratio * max(width, 1)
        if centerline_distance > max_centerline_distance:
            continue
        if track_region_progress(item, settings.target_region, width, height) < settings.target_min_region_progress:
            continue
        centerline_score = 1.0 - min(1.0, centerline_distance / max(max_centerline_distance, 1.0))
        displacement_rate = item.displacement / max(item.frames_seen - 1, 1)
        score = (
            displacement_rate
            * max(0.1, item.max_confidence)
            * max(0.35, item.downhill_consistency)
            * (0.5 + centerline_score)
        )
        if score > best_score:
            best_score = score
            best_id = track_id
    return best_id


def choose_moving_candidate(
    candidates: list[CandidatePose],
    stats: dict[int, TrackCandidateStats],
    settings: ProcessingSettings,
    width: int,
    height: int,
) -> CandidatePose | None:
    present_track_ids = {
        candidate.track_id
        for candidate in candidates
        if candidate.track_id is not None and candidate.bbox is not None
    }
    eligible_stats = {
        track_id: item
        for track_id, item in stats.items()
        if track_id in present_track_ids
    }
    return find_candidate_by_track_id(
        candidates, choose_moving_track(eligible_stats, settings, width, height)
    )


def track_passes_moving_gate(
    track_id: int,
    stats: dict[int, TrackCandidateStats],
    settings: ProcessingSettings,
    width: int,
    height: int,
) -> bool:
    item = stats.get(track_id)
    if item is None:
        return False
    return choose_moving_track({track_id: item}, settings, width, height) == track_id


def should_gate_new_track(settings: ProcessingSettings, active_track_id: int | None) -> bool:
    return (
        settings.target_strategy == "moving"
        and active_track_id is not None
        and settings.target_point is None
    )


def track_centerline_distance(
    item: TrackCandidateStats,
    region: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> float:
    if region is not None:
        x1, _, x2, _ = region_to_pixels(region, width, height)
        centerline_x = (x1 + x2) / 2.0
    else:
        centerline_x = width / 2.0
    return abs(item.last_center[0] - centerline_x)


def track_region_progress(
    item: TrackCandidateStats,
    region: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> float:
    if region is None:
        return item.last_center[1] / max(float(height), 1.0)
    _, y1, _, y2 = region_to_pixels(region, width, height)
    return (item.last_center[1] - y1) / max(y2 - y1, 1.0)


def find_candidate_by_track_id(
    candidates: list[CandidatePose], track_id: int | None
) -> CandidatePose | None:
    if track_id is None:
        return None
    for candidate in candidates:
        if candidate.track_id == track_id:
            return candidate
    return None


def filter_candidates_by_region(
    candidates: list[CandidatePose],
    region: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> list[CandidatePose]:
    if region is None:
        return candidates
    x1, y1, x2, y2 = region_to_pixels(region, width, height)
    kept = []
    for candidate in candidates:
        if candidate.bbox is None:
            continue
        cx, cy = bbox_center(candidate.bbox)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            kept.append(candidate)
    return kept


def reject_too_small_candidates(
    candidates: list[CandidatePose], settings: ProcessingSettings, height: int
) -> list[CandidatePose]:
    return [
        candidate
        for candidate in candidates
        if candidate.bbox is not None and not is_too_small_bbox(candidate.bbox, settings, height)
    ]


def is_too_small_bbox(
    bbox: tuple[float, float, float, float], settings: ProcessingSettings, height: int
) -> bool:
    return bbox[3] - bbox[1] < max(24.0, settings.too_small_height_ratio * height)


def is_usable_motion_box(
    bbox: tuple[float, float, float, float] | None, settings: ProcessingSettings, height: int
) -> bool:
    return bbox is None or not is_too_small_bbox(bbox, settings, height)


def nearest_candidate_to_corridor_centerline(
    candidates: list[CandidatePose],
    region: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    if not candidates:
        return {
            "distance_px": None,
            "bbox_height_px": None,
            "bbox_confidence": None,
            "track_id": None,
        }
    if region is not None:
        x1, _, x2, _ = region_to_pixels(region, width, height)
        centerline_x = (x1 + x2) / 2.0
    else:
        centerline_x = width / 2.0

    best: CandidatePose | None = None
    best_distance = float("inf")
    for candidate in candidates:
        if candidate.bbox is None:
            continue
        cx, _ = bbox_center(candidate.bbox)
        distance = abs(cx - centerline_x)
        if distance < best_distance:
            best_distance = distance
            best = candidate

    if best is None or best.bbox is None:
        return {
            "distance_px": None,
            "bbox_height_px": None,
            "bbox_confidence": None,
            "track_id": None,
        }

    return {
        "distance_px": clean_number(best_distance),
        "bbox_height_px": clean_number(best.bbox[3] - best.bbox[1]),
        "bbox_confidence": round(float(best.bbox_conf), 5),
        "track_id": best.track_id,
    }


def region_to_pixels(
    region: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = region
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        return (x1 * width, y1 * height, x2 * width, y2 * height)
    return region


def select_candidate(
    candidates: list[CandidatePose],
    active_track_id: int | None,
    prev_bbox: tuple[float, float, float, float] | None,
    initial_box: tuple[float, float, float, float] | None,
    target_point: tuple[float, float] | None,
    target_frame: int,
    frame_index: int,
    max_reacquire_distance_ratio: float,
    width: int,
    height: int,
    allow_new_track: bool = True,
) -> CandidatePose | None:
    if not candidates:
        return None

    if active_track_id is None and target_point is not None and frame_index < target_frame:
        return None

    diagonal = max(math.hypot(width, height), 1.0)
    if active_track_id is not None:
        for candidate in candidates:
            if candidate.track_id == active_track_id:
                if prev_bbox is not None and candidate.bbox is not None:
                    center_distance = bbox_center_distance(prev_bbox, candidate.bbox)
                    if center_distance > max_reacquire_distance_ratio * diagonal:
                        continue
                return candidate
        if not allow_new_track:
            return None

    reference_box = initial_box or prev_bbox
    best_score = -1.0
    best: CandidatePose | None = None
    frame_area = max(float(width * height), 1.0)
    best_reacquire_distance = float("inf")
    using_previous_box = reference_box is not None and prev_bbox is not None and initial_box is None

    for candidate in candidates:
        box = candidate.bbox
        area_term = (bbox_area(box) / frame_area) if box else 0.0
        confidence = candidate.bbox_conf
        core_conf = mean_confidence(candidate.scores, CORE_JOINTS)
        if reference_box and box:
            iou = bbox_iou(reference_box, box)
            center_distance = bbox_center_distance(reference_box, box)
            center_score = 1.0 - min(1.0, center_distance / (max_reacquire_distance_ratio * diagonal))
            score = 3.0 * iou + 1.25 * center_score + 0.35 * confidence + 0.25 * core_conf
        else:
            center_distance = float("inf")
            score = confidence + 0.35 * core_conf + 0.5 * area_term
        if active_track_id is None and target_point is not None and box is not None:
            cx, cy = bbox_center(box)
            distance = math.hypot(cx - target_point[0], cy - target_point[1])
            point_score = 1.0 - min(1.0, distance / (0.35 * diagonal))
            score += 2.5 * point_score
        if score > best_score:
            best_score = score
            best = candidate
            best_reacquire_distance = center_distance

    if using_previous_box and best_reacquire_distance > max_reacquire_distance_ratio * diagonal:
        return None

    return best


def crop_around_box(
    frame: np.ndarray, box: tuple[float, float, float, float], margin: float
) -> tuple[np.ndarray | None, tuple[float, float]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    side = max(side, 96.0)
    nx1 = max(0, int(round(cx - side / 2.0)))
    ny1 = max(0, int(round(cy - side / 2.0)))
    nx2 = min(width, int(round(cx + side / 2.0)))
    ny2 = min(height, int(round(cy + side / 2.0)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None, (0.0, 0.0)
    return frame[ny1:ny2, nx1:nx2], (float(nx1), float(ny1))


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_center_distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def predict_bbox(
    box: tuple[float, float, float, float] | None,
    velocity: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    if velocity is None:
        predicted = box
    else:
        predicted = tuple(float(box[i] + velocity[i]) for i in range(4))
    return clip_bbox(predicted, width, height)


def clip_bbox(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        min(max(0.0, x1), float(width - 1)),
        min(max(0.0, y1), float(height - 1)),
        min(max(0.0, x2), float(width - 1)),
        min(max(0.0, y2), float(height - 1)),
    )


def propagate_keypoints(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    last_keypoints: np.ndarray,
    last_scores: np.ndarray,
    last_statuses: list[str],
    settings: ProcessingSettings,
) -> tuple[np.ndarray, np.ndarray] | None:
    usable_indexes = [
        idx
        for idx, (point, score, status) in enumerate(zip(last_keypoints, last_scores, last_statuses, strict=True))
        if status in {"observed", "inferred"}
        and float(score) >= settings.observe_conf
        and np.isfinite(point[0])
        and np.isfinite(point[1])
    ]
    if len(usable_indexes) < 3:
        return None

    prev_points = np.array([last_keypoints[idx] for idx in usable_indexes], dtype=np.float32).reshape(-1, 1, 2)
    next_points, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return None

    out_points = np.full((len(COCO_KEYPOINTS), 2), np.nan, dtype=np.float32)
    out_scores = np.zeros(len(COCO_KEYPOINTS), dtype=np.float32)
    good_count = 0
    for local_idx, joint_idx in enumerate(usable_indexes):
        if int(status[local_idx][0]) != 1:
            continue
        if err is not None and float(err[local_idx][0]) > 35.0:
            continue
        x, y = next_points[local_idx][0]
        if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
            continue
        out_points[joint_idx] = (x, y)
        out_scores[joint_idx] = max(0.0, float(last_scores[joint_idx]) * 0.72)
        good_count += 1

    if good_count < 3:
        return None
    return out_points, out_scores


def render_overlay(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    statuses: list[str],
    bbox: tuple[float, float, float, float] | None,
    bbox_conf: float,
    track_id: int | None,
    frame_index: int,
    frame_status: str,
    warnings: list[str],
    settings: ProcessingSettings,
) -> np.ndarray:
    canvas = frame.copy()
    skeleton_layer = frame.copy()

    for a, b in COCO_EDGES:
        if not drawable(scores[a], statuses[a], settings) or not drawable(scores[b], statuses[b], settings):
            continue
        p1 = tuple(np.round(keypoints[a]).astype(int).tolist())
        p2 = tuple(np.round(keypoints[b]).astype(int).tolist())
        color = limb_color(min(float(scores[a]), float(scores[b])), statuses[a], statuses[b], settings)
        thickness = 4 if min(scores[a], scores[b]) >= settings.confident_conf else 2
        if "inferred" in (statuses[a], statuses[b]):
            draw_dashed_line(skeleton_layer, p1, p2, color, thickness=2)
        else:
            cv2.line(skeleton_layer, p1, p2, color, thickness, cv2.LINE_AA)

    for idx, point in enumerate(keypoints):
        if not drawable(scores[idx], statuses[idx], settings):
            continue
        x, y = np.round(point).astype(int)
        color = joint_color(float(scores[idx]), statuses[idx], settings)
        radius = 6 if scores[idx] >= settings.confident_conf else 4
        cv2.circle(skeleton_layer, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(skeleton_layer, (x, y), radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

    cv2.addWeighted(skeleton_layer, 0.88, canvas, 0.12, 0, canvas)

    if settings.draw_box and bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        box_color = (80, 220, 80) if frame_status == "tracking_ok" else (0, 210, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
        if settings.draw_labels:
            label = f"id:{track_id if track_id is not None else '-'} box:{bbox_conf:.2f}"
            cv2.putText(canvas, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    if settings.draw_labels:
        status_color = status_to_color(frame_status)
        text = f"frame {frame_index} | {frame_status}"
        if warnings:
            text += " | " + ",".join(warnings[:2])
        draw_label(canvas, text, (16, 28), status_color)

    return canvas


def frame_record(
    frame_index: int,
    timestamp: float,
    track_id: int | None,
    bbox: tuple[float, float, float, float] | None,
    bbox_conf: float,
    candidate_source: str,
    status: str,
    warnings: list[str],
    raw_candidate_count: int,
    candidate_count: int,
    selection_block_reason: str | None,
    corridor_candidate: dict[str, Any],
    width: int,
    height: int,
    raw_keypoints: np.ndarray,
    raw_scores: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    statuses: list[str],
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp, 6),
        "track_id": track_id,
        "bbox": clean_box(bbox),
        "bbox_confidence": round(float(bbox_conf), 5),
        "candidate_source": candidate_source,
        "status": status,
        "warnings": warnings,
        "diagnostics": detection_diagnostics(
            bbox,
            bbox_conf,
            status,
            raw_candidate_count,
            candidate_count,
            selection_block_reason,
            corridor_candidate,
            raw_scores,
            width,
            height,
        ),
        "raw_keypoints": keypoint_list(raw_keypoints, raw_scores, None),
        "smoothed_keypoints": keypoint_list(keypoints, scores, statuses),
    }


def detection_diagnostics(
    bbox: tuple[float, float, float, float] | None,
    bbox_conf: float,
    status: str,
    raw_candidate_count: int,
    candidate_count: int,
    selection_block_reason: str | None,
    corridor_candidate: dict[str, Any],
    scores: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    failure_type = None
    if status in {"no_detection", "tracking_lost"}:
        failure_type = (
            selection_block_reason
            or ("recall_no_candidate" if candidate_count == 0 else "selection_rejected")
        )
    elif status == "low_confidence":
        failure_type = "low_confidence"

    bbox_height = None
    bbox_area_ratio = None
    if bbox is not None:
        bbox_height = max(0.0, float(bbox[3] - bbox[1]))
        bbox_area_ratio = bbox_area(bbox) / max(float(width * height), 1.0)

    return {
        "raw_candidate_count": raw_candidate_count,
        "candidate_count": candidate_count,
        "selection_block_reason": selection_block_reason,
        "nearest_corridor_candidate": corridor_candidate,
        "selected_bbox_height_px": clean_number(bbox_height)
        if bbox_height is not None
        else None,
        "selected_bbox_area_ratio": clean_number(bbox_area_ratio)
        if bbox_area_ratio is not None
        else None,
        "selected_bbox_confidence": round(float(bbox_conf), 5),
        "selected_core_confidence": round(float(mean_confidence(scores, CORE_JOINTS)), 5),
        "failure_type": failure_type,
    }


def summarize_frame_diagnostics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counts: dict[str, int] = {}
    raw_candidate_total = 0
    candidate_total = 0
    bbox_heights = []
    corridor_distances = []
    for frame in frames:
        diagnostics = frame.get("diagnostics", {})
        raw_candidate_total += int(diagnostics.get("raw_candidate_count") or 0)
        candidate_total += int(diagnostics.get("candidate_count") or 0)
        failure_type = diagnostics.get("failure_type")
        if failure_type:
            failure_counts[str(failure_type)] = failure_counts.get(str(failure_type), 0) + 1
        height = diagnostics.get("selected_bbox_height_px")
        if height is not None:
            bbox_heights.append(float(height))
        corridor_candidate = diagnostics.get("nearest_corridor_candidate") or {}
        distance = corridor_candidate.get("distance_px")
        if distance is not None:
            corridor_distances.append(float(distance))

    frame_count = len(frames)
    return {
        "failure_counts": failure_counts,
        "avg_raw_candidates_per_frame": round(raw_candidate_total / frame_count, 4)
        if frame_count
        else 0.0,
        "avg_candidates_after_filter_per_frame": round(candidate_total / frame_count, 4)
        if frame_count
        else 0.0,
        "avg_selected_bbox_height_px": round(float(np.mean(bbox_heights)), 4)
        if bbox_heights
        else None,
        "min_selected_bbox_height_px": round(float(np.min(bbox_heights)), 4)
        if bbox_heights
        else None,
        "avg_nearest_corridor_candidate_distance_px": round(float(np.mean(corridor_distances)), 4)
        if corridor_distances
        else None,
        "min_nearest_corridor_candidate_distance_px": round(float(np.min(corridor_distances)), 4)
        if corridor_distances
        else None,
    }


def keypoint_list(
    points: np.ndarray, scores: np.ndarray, statuses: list[str] | None
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, name in enumerate(COCO_KEYPOINTS):
        x, y = float(points[idx][0]), float(points[idx][1])
        item = {
            "name": name,
            "x": clean_number(x),
            "y": clean_number(y),
            "confidence": round(float(scores[idx]), 5),
        }
        if statuses is not None:
            item["status"] = statuses[idx]
        items.append(item)
    return items


def drawable(score: float, status: str, settings: ProcessingSettings) -> bool:
    return (
        status in {"observed", "inferred"}
        and float(score) >= settings.min_draw_conf
        and np.isfinite(score)
    )


def joint_color(score: float, status: str, settings: ProcessingSettings) -> tuple[int, int, int]:
    if status == "inferred":
        return (255, 210, 80)
    if score >= settings.confident_conf:
        return (70, 230, 70)
    return (0, 220, 255)


def limb_color(
    score: float, status_a: str, status_b: str, settings: ProcessingSettings
) -> tuple[int, int, int]:
    if "inferred" in (status_a, status_b):
        return (255, 200, 80)
    if score >= settings.confident_conf:
        return (70, 220, 70)
    return (0, 220, 255)


def status_to_color(status: str) -> tuple[int, int, int]:
    if status == "tracking_ok":
        return (50, 210, 50)
    if status == "low_confidence":
        return (0, 210, 255)
    return (50, 80, 240)


def draw_label(
    image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    x, y = origin
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    cv2.rectangle(image, (x - 6, y - h - 8), (x + w + 8, y + baseline + 6), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)


def draw_dashed_line(
    image: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = float(math.hypot(x2 - x1, y2 - y1))
    if length < 1:
        return
    direction = ((x2 - x1) / length, (y2 - y1) / length)
    distance = 0.0
    while distance < length:
        start = distance
        end = min(distance + dash, length)
        s = (int(x1 + direction[0] * start), int(y1 + direction[1] * start))
        e = (int(x1 + direction[0] * end), int(y1 + direction[1] * end))
        cv2.line(image, s, e, color, thickness, cv2.LINE_AA)
        distance += dash + gap


def bbox_area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def mean_confidence(scores: np.ndarray, indexes: list[int]) -> float:
    vals = [float(scores[idx]) for idx in indexes if idx < len(scores) and np.isfinite(scores[idx])]
    return float(np.mean(vals)) if vals else 0.0


def safe_ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / denominator, 5) if denominator else 0.0


def clean_number(value: float) -> float | None:
    return round(float(value), 3) if np.isfinite(value) else None


def clean_box(box: tuple[float, float, float, float] | None) -> list[float] | None:
    if box is None:
        return None
    return [round(float(v), 3) for v in box]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
