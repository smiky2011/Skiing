"""
Train YOLOv8 gate detection model for ski racing.

Usage:
    python scripts/train_detector.py --data data/annotations/data.yaml
    python scripts/train_detector.py --data data/annotations/final_combined_1class_20260213/data.yaml --model yolov8n.pt --epochs 150 --imgsz 960 --batch 8 --freeze 10 --cos-lr
    python scripts/train_detector.py --resume runs/detect/gate_detector_xxx/weights/last.pt
    python scripts/train_detector.py --eval-only models/gate_detector_best.pt --data data/annotations/data.yaml --save-metrics
"""
from ultralytics import YOLO
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch

# ---------------------------------------------------------------------------
# Monkey-patch: raise the default NMS per-image time limit from 0.05s to 0.5s.
# The Ultralytics default (0.05s) is too tight for 960px images on MPS/CPU and
# triggers "WARNING ⚠️ NMS time limit exceeded", which silently drops detections.
# Searches multiple module paths to handle different Ultralytics versions.
# ---------------------------------------------------------------------------
def _patch_nms_time_limit():
    import importlib
    for mod_path in [
        "ultralytics.utils.nms",          # v8.4+ (split into dedicated nms module)
        "ultralytics.utils.ops",           # v8.0–v8.3
        "ultralytics.yolo.utils.ops",      # pre-v8.0
    ]:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        orig = getattr(mod, "non_max_suppression", None)
        if orig is None:
            continue

        def _make_patched(original):
            def _patched(*args, **kwargs):
                kwargs.setdefault("max_time_img", 0.5)
                return original(*args, **kwargs)
            return _patched

        mod.non_max_suppression = _make_patched(orig)
        print(f"✓ Patched NMS time limit (max_time_img=0.5s) in {mod_path}")
        return
    print("⚠️  Could not locate non_max_suppression to patch — NMS time limit unchanged")

_patch_nms_time_limit()


def _resolve_device(device: str, *, training: bool = True) -> str:
    """Resolve device string to an available compute device.

    Priority when training: CUDA > CPU (MPS is skipped — PyTorch MPS backend
    has known shape-mismatch bugs in the Task-Aligned Assigner during training).
    Priority for inference: MPS > CUDA > CPU.
    Raises RuntimeError if a specific device is requested but unavailable.
    """
    if device == "auto":
        if not training and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if device == "mps":
        if not torch.backends.mps.is_available():
            built = torch.backends.mps.is_built()
            raise RuntimeError(
                "MPS requested but not available. "
                f"mps_built={built}. "
                "Ensure you're using arm64 Python (not Rosetta) and a PyTorch build with MPS support."
            )
        return "mps"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Check GPU drivers and CUDA PyTorch build.")
        return "cuda"

    return device


def train_gate_detector(
    data_yaml,
    model_name="yolov8n.pt",
    epochs=150,
    imgsz=960,
    batch=8,
    device="auto",
    workers=8,
    cos_lr=False,
    freeze=10,
    resume=None,
    amp=None,
):
    """
    Train YOLOv8 model for gate detection.

    Args:
        data_yaml: Path to YOLO data.yaml (from Roboflow export).
        model_name: Pre-trained model to fine-tune (yolov8n.pt, yolov8m.pt, etc.).
        epochs: Number of training epochs.
        imgsz: Image size for training.
        batch: Batch size.
        device: Compute device — 'auto', 'mps', 'cuda', or 'cpu'.
        workers: Number of dataloader workers.
        cos_lr: Use cosine learning rate schedule (helps on small datasets).
        freeze: Number of backbone layers to freeze (0 = train all).
        resume: Path to last.pt to resume an interrupted run.
        amp: Mixed precision. Defaults to False on MPS, True elsewhere.

    Returns:
        Training results and metrics.
    """
    device = _resolve_device(device, training=True)
    if amp is None:
        amp = device != "mps"
    if device == "mps":
        workers = 0

    # Resume from checkpoint or load base model
    if resume:
        model = YOLO(resume)
        print(f"Resuming training from {resume}")
        results = model.train(resume=True, device=device, workers=workers, amp=amp)
        return results

    model = YOLO(model_name)

    # Build training kwargs
    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        name=f"gate_detector_{datetime.now().strftime('%Y%m%d_%H%M')}",
        patience=30,       # Early stopping
        save=True,
        plots=True,
        cos_lr=cos_lr,
        amp=amp,
        max_det=30,        # Gate images have <30 objects; avoids NMS time limit exceeded
        # Augmentation config for ski racing
        flipud=0.0,        # Don't flip upside down (sky/snow orientation matters)
        fliplr=0.5,        # Horizontal flip is fine
        mosaic=0.5,        # Balanced mosaic augmentation
        close_mosaic=25,   # Disable mosaic for final fine-tuning epochs
        scale=0.5,         # Keep default random scale range explicit for reproducibility
        mixup=0.0,         # Keep label assignment stable on small datasets
        copy_paste=0.0,    # Keep label assignment stable on small datasets
        hsv_h=0.015,       # Slight hue variation (weather/lighting)
        hsv_s=0.4,         # Saturation variation
        hsv_v=0.3,         # Brightness variation
    )

    if freeze > 0:
        train_kwargs["freeze"] = freeze

    results = model.train(**train_kwargs)
    return results


def evaluate_model(model_path, data_yaml=None, device="auto", workers=8, save_metrics=None):
    """
    Evaluate trained model and print metrics.

    Args:
        model_path: Path to trained .pt weights.
        data_yaml: Path to data.yaml (uses model's default split if None).
        device: Compute device.
        workers: Number of dataloader workers.
        save_metrics: If set, save metrics JSON to this path.

    Returns:
        Ultralytics metrics object.
    """
    model = YOLO(model_path)
    device = _resolve_device(device, training=False)

    if data_yaml:
        metrics = model.val(data=data_yaml, device=device, workers=workers)
    else:
        metrics = model.val(device=device, workers=workers)

    print("\n=== Model Evaluation ===")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")

    # Per-class breakdown when available
    if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > 1:
        print("\n--- Per-class mAP@0.5 ---")
        for i, ap in enumerate(metrics.box.ap50):
            print(f"  Class {i}: {ap:.4f}")

    # Check if we meet minimum thresholds
    if metrics.box.map50 < 0.60:
        print("\n⚠️  mAP@0.5 < 0.60 — Consider:")
        print("   - Annotating 50+ more images")
        print("   - Using a smaller model (yolov8n.pt) with --freeze 10")
        print("   - Checking annotation quality")
        print("   - Reducing mosaic to 0.5 and increasing close_mosaic to 25")
        print("   - Training longer (150+ epochs with --cos-lr)")
    elif metrics.box.map50 < 0.85:
        print("\n📊 mAP@0.5 is decent but can improve.")
        print("   Target: >0.85 F1 by Week 6.")
    else:
        print("\n✅ Strong detection performance!")

    # Save metrics to JSON
    if save_metrics:
        metrics_dict = {
            "model": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        out_path = Path(save_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"\n💾 Metrics saved to {out_path}")

    return metrics


def test_on_video(model_path, video_path, conf=0.25, device="auto", vid_stride=1):
    """
    Run detection on a video and save annotated output.

    Args:
        model_path: Path to trained .pt weights.
        video_path: Path to input video file.
        conf: Confidence threshold (default 0.25, tuned to F1-optimal point).
        device: Compute device.
        vid_stride: Process every Nth frame (higher = faster).
    """
    model = YOLO(model_path)
    device = _resolve_device(device, training=False)

    results = model.predict(
        source=video_path,
        save=True,
        conf=conf,
        device=device,
        vid_stride=vid_stride,
    )

    print(f"✓ Annotated video saved to runs/detect/predict/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train gate detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python scripts/train_detector.py --data data/annotations/final_combined_1class_20260213/data.yaml --model yolov8n.pt --epochs 150 --imgsz 960 --batch 8 --freeze 10 --cos-lr

  # Resume interrupted training
  python scripts/train_detector.py --data data/annotations/data.yaml --resume runs/detect/gate_detector_xxx/weights/last.pt

  # Evaluate and save metrics
  python scripts/train_detector.py --data data/annotations/data.yaml --eval-only models/gate_detector_best.pt --save-metrics

  # Test on video with custom confidence
  python scripts/train_detector.py --data data/annotations/data.yaml --eval-only models/gate_detector_best.pt --test-video video.mp4 --conf 0.3
        """,
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, cpu")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mixed precision. Default: auto (disabled on MPS for stability).",
    )
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning rate schedule")
    parser.add_argument("--freeze", type=int, default=10, help="Freeze first N backbone layers (e.g. 10)")
    parser.add_argument("--resume", type=str, default=None, help="Path to last.pt to resume training")
    parser.add_argument("--eval-only", type=str, default=None, help="Path to trained model for eval only")
    parser.add_argument("--test-video", type=str, default=None, help="Video path for inference test")
    parser.add_argument("--vid-stride", type=int, default=1, help="Video frame stride for inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference (default: 0.25)")
    parser.add_argument("--save-metrics", action="store_true", help="Save evaluation metrics to JSON")
    args = parser.parse_args()

    # Determine metrics output path
    metrics_path = None
    if args.save_metrics:
        metrics_path = f"runs/metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    if args.resume:
        print("Resuming training...")
        results = train_gate_detector(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            cos_lr=args.cos_lr,
            freeze=args.freeze,
            resume=args.resume,
            amp=args.amp,
        )
        print("\nTraining complete!")

    elif args.test_video:
        if args.eval_only:
            test_on_video(
                args.eval_only,
                args.test_video,
                conf=args.conf,
                device=args.device,
                vid_stride=args.vid_stride,
            )
        else:
            print("Provide --eval-only with model path to test on video")

    elif args.eval_only:
        evaluate_model(
            args.eval_only,
            args.data,
            device=args.device,
            workers=args.workers,
            save_metrics=metrics_path,
        )

    else:
        resolved_device = _resolve_device(args.device)
        effective_amp = args.amp if args.amp is not None else (resolved_device != "mps")
        print("Starting training...")
        print(f"  Device:    {resolved_device}")
        print(f"  Model:     {args.model}")
        print(f"  Epochs:    {args.epochs}")
        print(f"  Cosine LR: {args.cos_lr}")
        print(f"  AMP:       {effective_amp}")
        print(f"  Freeze:    {args.freeze} layers")
        results = train_gate_detector(
            args.data,
            args.model,
            args.epochs,
            args.imgsz,
            args.batch,
            device=args.device,
            workers=args.workers,
            cos_lr=args.cos_lr,
            freeze=args.freeze,
            amp=args.amp,
        )
        print("\nTraining complete!")
        print(f"Best model saved to: runs/detect/*/weights/best.pt")
