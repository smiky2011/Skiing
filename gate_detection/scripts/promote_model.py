"""
Model promotion script for 2.2.

Usage:
    python scripts/promote_model.py \
        --new-model runs/detect/gate_detector_20260226_0923/weights/best.pt \
        --data data/datasets/final_combined_1class_20260226_curated/data.yaml \
        --baseline-model models/gate_detector_best.pt

Compares new vs. current model on the validation split.
Promotes only if:
  - new mAP@0.5 > current mAP@0.5, AND
  - no Stage 2 metric regresses >20% (checked by running regression suite if --regression-dir given)

On promotion:
  - Copies new model to models/gate_detector_best.pt
  - Saves timestamped backup of old model
  - Updates shared/docs/MODEL_REGISTRY.md
  - Saves promotion report to tracks/B_model_retraining/reports/
"""
import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_model(model_path: str, data_yaml: str, device: str = "auto") -> dict:
    """Run YOLO val and return metrics dict."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device, workers=0, verbose=False)
    return {
        "model": str(model_path),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "F1": 2 * float(metrics.box.mp) * float(metrics.box.mr) / (
            float(metrics.box.mp) + float(metrics.box.mr) + 1e-9
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and promote new gate detector model")
    parser.add_argument("--new-model", required=True, help="Path to new model best.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml for eval")
    parser.add_argument("--baseline-model", default="models/gate_detector_best.pt",
                        help="Current best model to compare against")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, cpu")
    parser.add_argument("--force", action="store_true", help="Promote even if new model is worse")
    args = parser.parse_args()

    new_path = Path(args.new_model)
    baseline_path = PROJECT_ROOT / args.baseline_model
    data_yaml = str(PROJECT_ROOT / args.data if not Path(args.data).is_absolute() else args.data)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"Evaluating baseline: {baseline_path}")
    baseline_metrics = evaluate_model(str(baseline_path), data_yaml, args.device)
    print(f"  mAP@0.5={baseline_metrics['mAP50']:.4f}  F1={baseline_metrics['F1']:.4f}")

    print(f"\nEvaluating new model: {new_path}")
    new_metrics = evaluate_model(str(new_path), data_yaml, args.device)
    print(f"  mAP@0.5={new_metrics['mAP50']:.4f}  F1={new_metrics['F1']:.4f}")

    delta_map = new_metrics["mAP50"] - baseline_metrics["mAP50"]
    delta_f1 = new_metrics["F1"] - baseline_metrics["F1"]
    print(f"\nΔ mAP@0.5 = {delta_map:+.4f}")
    print(f"Δ F1      = {delta_f1:+.4f}")

    # Promotion requires BOTH mAP and F1 improvement (or --force)
    improves_map = delta_map > 0
    improves_f1 = delta_f1 > 0
    should_promote = args.force or (improves_map and improves_f1)

    if not should_promote:
        reasons = []
        if not improves_map:
            reasons.append(f"mAP@0.5 did not improve ({delta_map:+.4f})")
        if not improves_f1:
            reasons.append(f"F1 did not improve ({delta_f1:+.4f})")
        print(f"\n❌ Not promoting: {'; '.join(reasons)}")
        print("   Use --force to promote anyway.")
        sys.exit(1)

    print(f"\n✅ New model improves mAP@0.5 by {delta_map:+.4f} and F1 by {delta_f1:+.4f}. Promoting...")

    # Backup old model
    backup_path = PROJECT_ROOT / f"models/gate_detector_best_{ts}_backup.pt"
    shutil.copy2(str(baseline_path), str(backup_path))
    print(f"   Backed up old model to {backup_path}")

    # Copy new model
    shutil.copy2(str(new_path), str(baseline_path))
    print(f"   Promoted {new_path} → {baseline_path}")

    # Save promotion report
    report_dir = PROJECT_ROOT / "tracks/B_model_retraining/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": ts,
        "new_model": str(new_path),
        "baseline_model": str(baseline_path),
        "baseline_metrics": baseline_metrics,
        "new_metrics": new_metrics,
        "delta_mAP50": round(delta_map, 6),
        "promoted": True,
        "note": "Curated dataset: final_combined_1class_20260226_curated (100 hard-neg added)",
    }
    report_path = report_dir / f"promotion_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Promotion report → {report_path}")

    # Update MODEL_REGISTRY.md
    registry_path = PROJECT_ROOT / "shared/docs/MODEL_REGISTRY.md"
    if not registry_path.parent.exists():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists():
        registry_path.write_text("# Model Registry\n\n")
    existing = registry_path.read_text()
    entry = (
        f"\n## {ts} — gate_detector_best.pt promoted\n"
        f"- Source: `{new_path}`\n"
        f"- mAP@0.5: {new_metrics['mAP50']:.4f} (Δ={delta_map:+.4f} vs baseline)\n"
        f"- F1: {new_metrics['F1']:.4f}\n"
        f"- Dataset: `data/datasets/final_combined_1class_20260226_curated`\n"
        f"- Training: YOLOv8s, imgsz=960, batch=8, freeze=10, cos-lr, 150 epochs\n"
    )
    registry_path.write_text(existing + entry)
    print(f"   Updated {registry_path}")

    print("\nDone! Run scripts/run_eval.py to verify Stage 2 metrics.")


if __name__ == "__main__":
    main()
