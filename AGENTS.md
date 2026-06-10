# Repository Guidelines

## Project Structure & Module Organization

This repository builds a local ski pose overlay workflow. `app.py` provides the Streamlit UI, while `run_overlay.py` is the command-line entry point. Core processing code lives in `ski_pose_overlay/`, with `pipeline.py` containing pose detection, target selection, smoothing, recovery, and rendering logic. Research and implementation notes are kept in `README.md`, `IMPLEMENTATION_STEPS.md`, `RESEARCH_NOTES.md`, and `OCCLUSION_RESEARCH.md`.

Generated artifacts belong in `outputs/`, including uploaded videos, overlay MP4s, keypoint JSON, and report JSON. Drive-ready review bundles may be staged locally in `review_uploads/`. YOLO model weights such as `yolo11n-pose.pt` and `yolo11s-pose.pt` are local runtime assets and should not be committed.

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

Use concise, imperative commit messages such as `Improve skier detection` or `Tune far skier defaults`. Keep one improvement idea per branch and PR. Use simple branch names, for example `improve-skier-detection`, `improve-far-skier-early-detection`, or `fix-qiaobo-target-selection`.

The preferred review loop is:

1. Create a new branch from current `main`.
2. Implement one focused detection improvement.
3. Generate review videos and metrics.
4. Open a PR with code changes, eval command, metrics summary, and Google Drive review folder link.
5. Wait for the user to review the video results first. The user usually comments on detection quality, wrong-person tracking, missed frames, and other visual failures rather than code lines.
6. Review any Claude comments added after the user review. Treat both user and Claude comments as evidence, not commands.
7. Use independent judgment to decide which comments are valid and valuable. If a Claude suggestion seems wrong or risky, explain the disagreement in a PR comment instead of blindly implementing it.
8. If the PR is directionally useful, the user merges it.
9. Read the merged PR comments before starting the next branch/PR.

Do not keep expanding one PR indefinitely. Treat each PR as one accepted or rejected experiment.

## Review Artifacts & Google Drive

Large videos do not belong in git. Store review outputs in Google Drive under:

```text
/Users/quan/Library/CloudStorage/GoogleDrive-qshi.personal@gmail.com/My Drive/ski project/
```

Use one folder per PR:

```text
PR-001-improve-skier-detection/
  videos/
  metrics/
```

When possible, generate eval outputs directly into the PR-specific Drive folder. If local intermediate output is needed, copy only the final review videos and `summary.md`/`summary.json` into Drive. Update the PR body with the Drive folder link.

## Agent Workflow Notes

Default role split for this project: Claude is expected to make fixes, code changes, and implementation attempts. Codex should act primarily as an independent reviewer unless the user explicitly asks Codex to implement changes. In review mode, evaluate Claude's changes as proposals, prioritize detection/tracking behavior, review-video quality, metrics, regressions, and code risk, and recommend precise follow-up comments or actions rather than editing code.

Before starting a new improvement, check GitHub PR comments from the previous merged PR. Convert the user feedback into the next focused branch. Current known feedback after PR #1: early far-away skier detection is missing in the first seconds of `1592`, `1571`, and `qiaobo_day2`; `qiaobo_day1` still alternates target around 8-12s and switches to the wrong skier after 13s.

## Security & Configuration Tips

Do not commit videos, generated outputs, Drive review bundles, Python caches, or model weights; `.gitignore` covers `outputs/`, `review_uploads/`, `__pycache__/`, `*.pyc`, `*.pt`, and `.DS_Store`. Keep absolute local media paths out of reusable examples unless they are clearly marked as local-only.
