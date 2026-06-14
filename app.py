from __future__ import annotations

from pathlib import Path

import streamlit as st

from ski_pose_overlay import ProcessingSettings, process_video


DEFAULT_VIDEO = "/Users/quan/Documents/石泉/ski/1928.MP4"


st.set_page_config(page_title="Ski Skeleton Overlay", layout="wide")
st.title("Ski Skeleton Overlay")

left, right = st.columns([0.36, 0.64], gap="large")

with left:
    st.subheader("Input")
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "MP4", "MOV"])
    local_path = st.text_input("Or local video path", value=DEFAULT_VIDEO)

    st.subheader("Processing")
    preset = st.selectbox(
        "Preset",
        ["standard", "far skier", "gate occlusion"],
        index=0,
        help="Far skier lowers detection thresholds and uses larger inference. Gate occlusion allows short inferred gaps.",
    )
    preset_values = {
        "standard": {
            "imgsz": 1280,
            "conf": 0.25,
            "min_draw_conf": 0.20,
            "observe_conf": 0.15,
            "confident_conf": 0.50,
            "max_infer_gap": 3,
            "refine_pose": False,
            "roi_recovery": True,
            "motion_inference": True,
            "target_strategy": "first",
        },
        "far skier": {
            "imgsz": 1920,
            "conf": 0.05,
            "min_draw_conf": 0.10,
            "observe_conf": 0.08,
            "confident_conf": 0.35,
            "max_infer_gap": 5,
            "refine_pose": True,
            "roi_recovery": True,
            "motion_inference": True,
            "target_strategy": "moving",
        },
        "gate occlusion": {
            "imgsz": 1600,
            "conf": 0.15,
            "min_draw_conf": 0.15,
            "observe_conf": 0.10,
            "confident_conf": 0.40,
            "max_infer_gap": 6,
            "refine_pose": True,
            "roi_recovery": True,
            "motion_inference": True,
            "target_strategy": "first",
        },
    }[preset]
    backend = st.selectbox("Backend", ["auto", "yolo", "mediapipe (experimental)"], index=0)
    model = st.text_input("YOLO model", value=ProcessingSettings().model)
    tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], index=0)
    imgsz = st.select_slider(
        "YOLO image size",
        options=[640, 960, 1280, 1600, 1920],
        value=preset_values["imgsz"],
    )
    conf = st.slider(
        "Confidence threshold",
        min_value=0.02,
        max_value=0.80,
        value=float(preset_values["conf"]),
        step=0.01,
    )
    start_frame = st.number_input("Start frame", min_value=0, value=0, step=25)
    max_frames = st.number_input("Max frames (0 = whole video)", min_value=0, value=0, step=25)
    smooth = st.toggle("Temporal smoothing", value=True)
    refine_pose = st.toggle("Crop refinement", value=bool(preset_values["refine_pose"]))
    roi_recovery = st.toggle("Predicted crop recovery", value=bool(preset_values["roi_recovery"]))
    motion_inference = st.toggle("Motion-inferred joints", value=bool(preset_values["motion_inference"]))
    max_infer_gap = st.number_input(
        "Inferred joint gap",
        min_value=0,
        max_value=15,
        value=int(preset_values["max_infer_gap"]),
        step=1,
    )
    draw_box = st.toggle("Draw skier box", value=True)
    draw_labels = st.toggle("Draw frame status", value=True)
    with st.expander("Target hint"):
        target_strategy = st.selectbox(
            "Target strategy",
            ["first", "moving"],
            index=0 if preset_values["target_strategy"] == "first" else 1,
        )
        target_frame = st.number_input("Target frame", min_value=0, value=0, step=1)
        hint_cols = st.columns(2)
        with hint_cols[0]:
            target_x = st.number_input("Target x", min_value=0.0, value=0.0, step=1.0)
        with hint_cols[1]:
            target_y = st.number_input("Target y", min_value=0.0, value=0.0, step=1.0)
        use_target = st.checkbox("Use target point", value=False)
    run = st.button("Generate overlay", type="primary", use_container_width=True)

with right:
    st.subheader("Output")
    output_area = st.container()


if run:
    input_path: Path
    upload_dir = Path("outputs/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    if uploaded is not None:
        input_path = upload_dir / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())
    else:
        input_path = Path(local_path).expanduser()

    if not input_path.exists():
        st.error(f"Video not found: {input_path}")
        st.stop()

    settings = ProcessingSettings(
        backend="mediapipe" if backend.startswith("mediapipe") else backend,
        model=model,
        tracker=tracker,
        imgsz=int(imgsz),
        conf=float(conf),
        start_frame=int(start_frame),
        max_frames=int(max_frames) if int(max_frames) > 0 else None,
        smooth=bool(smooth),
        min_draw_conf=float(preset_values["min_draw_conf"]),
        observe_conf=float(preset_values["observe_conf"]),
        confident_conf=float(preset_values["confident_conf"]),
        max_infer_gap=int(max_infer_gap),
        draw_box=bool(draw_box),
        draw_labels=bool(draw_labels),
        target_point=(float(target_x), float(target_y)) if use_target else None,
        target_frame=int(target_frame),
        target_strategy=target_strategy,
        refine_pose=bool(refine_pose),
        roi_recovery=bool(roi_recovery),
        motion_inference=bool(motion_inference),
    )

    progress_bar = st.progress(0)
    progress_text = st.empty()

    def on_progress(frame: int, total: int) -> None:
        progress_bar.progress(min(1.0, frame / max(total, 1)))
        progress_text.text(f"Processing frame {frame} / {total}")

    try:
        result = process_video(
            input_path,
            output_dir=Path("outputs"),
            settings=settings,
            progress_callback=on_progress,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    progress_text.text("Done")
    with output_area:
        st.video(result.output_video)
        st.json(
            {
                "backend": result.backend,
                "frames_processed": result.frames_processed,
                "frames_with_pose": result.frames_with_pose,
                "low_confidence_frames": result.low_confidence_frames,
                "tracking_lost_frames": result.tracking_lost_frames,
                "elapsed_sec": round(result.elapsed_sec, 2),
            }
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download overlay MP4",
                Path(result.output_video).read_bytes(),
                file_name=Path(result.output_video).name,
                mime="video/mp4",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Download keypoints JSON",
                Path(result.keypoints_json).read_bytes(),
                file_name=Path(result.keypoints_json).name,
                mime="application/json",
                use_container_width=True,
            )
        with col3:
            st.download_button(
                "Download report JSON",
                Path(result.report_json).read_bytes(),
                file_name=Path(result.report_json).name,
                mime="application/json",
                use_container_width=True,
            )
