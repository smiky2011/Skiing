# Repository Guidelines

## Project Structure & Module Organization

This repository builds a local ski pose overlay workflow. `app.py` provides the Streamlit UI, while `run_overlay.py` is the command-line entry point. Core processing code lives in `ski_pose_overlay/`, with `pipeline.py` containing pose detection, target selection, smoothing, recovery, and rendering logic. Research and implementation notes are kept in `README.md`, `IMPLEMENTATION_STEPS.md`, `RESEARCH_NOTES.md`, and `OCCLUSION_RESEARCH.md`.

Generated artifacts belong in `outputs/`, including uploaded videos, overlay MP4s, keypoint JSON, and report JSON. YOLO model weights such as `yolo11n-pose.pt` and `yolo11s-pose.pt` are local runtime assets and should not be committed.

## Build, Test, and Development Commands

Install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the Streamlit interface:

```bash
streamlit run app.py
```

Run a short CLI smoke test:

```bash
python3 run_overlay.py "/path/to/video.MP4" --max-frames 100
```

Use presets for difficult footage:

```bash
python3 run_overlay.py video.mp4 --preset far
python3 run_overlay.py video.mp4 --preset occlusion --start-frame 940 --max-frames 60
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and type hints, matching the existing code. Prefer `Path` for filesystem paths, `dataclass` for structured settings/results, and explicit CLI argument names such as `--target-point` or `--max-frames`. Keep processing logic inside `ski_pose_overlay/`; use top-level scripts only as UI or CLI adapters.

## Testing Guidelines

There is no dedicated test suite in this snapshot. Validate changes with targeted smoke runs using `--max-frames` to limit runtime, then inspect the generated `*_report.json`, `*_keypoints.json`, and `*_skeleton_overlay.mp4` in `outputs/`. For tracking changes, test at least one standard clip and one hard case using `--preset far` or `--preset occlusion`.

## Commit & Pull Request Guidelines

Local git history is unavailable in this checkout, so no project-specific commit convention can be inferred. Use concise, imperative commit messages such as `Add occlusion recovery smoke test` or `Tune far skier defaults`. Pull requests should describe the scenario tested, include the exact command used, mention affected presets/backends, and attach screenshots or sample output paths when overlay rendering changes.

## Security & Configuration Tips

Do not commit videos, generated outputs, Python caches, or model weights; `.gitignore` already covers `outputs/`, `__pycache__/`, `*.pyc`, and `*.pt`. Keep absolute local media paths out of reusable examples unless they are clearly marked as local-only.
