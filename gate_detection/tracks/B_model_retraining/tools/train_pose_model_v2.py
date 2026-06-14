#!/usr/bin/env python3
"""
Train a YOLOv8-Pose model (2 keypoints) for gate detection.

Writes all artifacts inside Track B:
- Runs: tracks/B_model_retraining/runs/pose/
- Final checkpoint: tracks/B_model_retraining/artifacts/models/gate_pose_best_YYYYMMDD.pt
- Training metadata: tracks/B_model_retraining/reports/pose_training_YYYYMMDD.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from ultralytics import YOLO


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model(model_hint: str) -> Tuple[YOLO, str]:
    """
    Resolve a pose model spec robustly in offline environments.

    Strategy:
    1) Use provided path/spec directly.
    2) Fall back to architecture yaml if pre-trained weights are unavailable.
    """
    try:
        return YOLO(model_hint), model_hint
    except Exception as first_err:
        fallback = "yolov8n-pose.yaml"
        try:
            model = YOLO(fallback)
            print(
                "Warning: could not load pose weights "
                f"'{model_hint}' ({first_err}). Falling back to '{fallback}'."
            )
            return model, fallback
        except Exception as second_err:
            raise RuntimeError(
                f"Failed to create YOLO pose model from '{model_hint}' and fallback '{fallback}'. "
                f"Errors: {first_err} | {second_err}"
            ) from second_err


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wave 2 YOLOv8-Pose gate model.")
    parser.add_argument(
        "--data",
        default="",
        help="Pose dataset data.yaml path. Default: latest artifacts/pose_1class_*/data.yaml",
    )
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Pose model weights or yaml.")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Train image size.")
    parser.add_argument("--batch", type=int, default=8, help="Train batch size.")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument(
        "--output",
        default="",
        help="Output checkpoint path (.pt). Default: artifacts/models/gate_pose_best_YYYYMMDD.pt",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    if args.data:
        data_yaml = (cwd / args.data).resolve()
    else:
        candidates = sorted((cwd / "artifacts").glob("pose_1class_*/data.yaml"))
        if not candidates:
            raise FileNotFoundError("No pose dataset found. Run prepare_pose_dataset_v2.py first.")
        data_yaml = candidates[-1].resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    if args.output:
        out_ckpt = (cwd / args.output).resolve()
    else:
        out_ckpt = (cwd / f"artifacts/models/gate_pose_best_{datetime.now().strftime('%Y%m%d')}.pt").resolve()
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    report_path = cwd / f"reports/pose_training_{datetime.now().strftime('%Y%m%d')}.json"
    runs_project = (cwd / "runs/pose").resolve()
    runs_project.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, model_source = resolve_model(args.model)

    workers = int(args.workers)
    if device == "mps":
        workers = 0

    run_name = f"gate_pose_{datetime.now().strftime('%Y%m%d_%H%M')}"
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        project=str(runs_project),
        name=run_name,
        device=device,
        workers=workers,
        verbose=True,
        save=True,
        patience=30,
        cos_lr=True,
        amp=(device != "mps"),
    )

    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    chosen = best_pt if best_pt.exists() else last_pt
    if not chosen.exists():
        raise FileNotFoundError(f"No checkpoint produced in run directory: {save_dir}")
    shutil.copy2(chosen, out_ckpt)

    metrics = {}
    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_yaml": str(data_yaml),
        "model_source": model_source,
        "resolved_device": device,
        "train_args": {
            "epochs": int(args.epochs),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "workers": workers,
        },
        "run_dir": str(save_dir),
        "checkpoint_copied_from": str(chosen),
        "checkpoint_output": str(out_ckpt),
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved checkpoint: {out_ckpt}")


if __name__ == "__main__":
    main()
