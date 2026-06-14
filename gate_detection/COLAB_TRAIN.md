# Re-train the gate detector on Google Colab

Re-trains the 1-class (`gate`) detector from the recovered labeled dataset, so we can probe it on our ski clips (issue #20). The original trained weights weren't recoverable; the **dataset is**, so we re-train.

- **Base model:** `yolo11s.pt` (stock **detection** model — NOT a pose model / NOT v1). Newer than the original's `yolov8n`, `s` tier for better small/far-gate recall and cross-domain generalization.
- **Recipe** (from the recovered `scripts/train_detector.py`): imgsz 960, freeze 10, cos-lr, `patience=30` early-stopping, `close_mosaic=25`, ski-safe augmentation (`flipud=0`). **Epochs capped at 100** (early-stopping is the real control; 150 was the original cap).
- **Runtime:** Colab GPU (T4 is fine). Set **Runtime → Change runtime type → GPU**.

> The dataset lives **only on branch `recover-gate-detection`** (and this probe branch off it) — it is NOT on `main`. Clone + checkout that branch, or upload the dataset zip.

---

## Cell 1 — install
```python
!pip -q install "ultralytics>=8.0.0"
import ultralytics; ultralytics.checks()   # confirms GPU
```

## Cell 2 — get the dataset (pick ONE)

**Option A — clone the repo branch** (if the repo is accessible to you; for a private repo, paste a GitHub token):
```python
# public:
!git clone --branch recover-gate-detection --single-branch https://github.com/smiky2011/Skiing.git /content/Skiing
# private (uncomment, set TOKEN):
# !git clone --branch recover-gate-detection --single-branch https://<TOKEN>@github.com/smiky2011/Skiing.git /content/Skiing
DATA_ROOT = "/content/Skiing/gate_detection/data/datasets/final_combined_1class_20260215"
```

**Option B — upload `final_combined_1class_20260215.zip`** (zip the dataset dir locally, drag into Colab Files):
```python
import zipfile, os
zipfile.ZipFile("/content/final_combined_1class_20260215.zip").extractall("/content/ds")
DATA_ROOT = "/content/ds/final_combined_1class_20260215"
```

## Cell 3 — fix the data.yaml path
The recovered `data.yaml` has only relative paths (`train: train/images`, …) and no `path:` key, so Ultralytics can't resolve it from Colab's cwd. Inject an absolute `path:`:
```python
import yaml, os
yml = os.path.join(DATA_ROOT, "data.yaml")
d = yaml.safe_load(open(yml))
d["path"] = DATA_ROOT                       # absolute root
d["train"], d["val"], d["test"] = "train/images", "valid/images", "test/images"
yaml.safe_dump(d, open(yml, "w"))
print(d)                                     # expect nc:1, names:['gate']
```

## Cell 4 — train (`yolo11s`, cap 100 epochs, early-stops via patience=30)
This inlines the recovered recipe (so it works whether you cloned or uploaded):
```python
from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.train(
    data=yml, epochs=100, imgsz=960, batch=16, freeze=10, cos_lr=True,
    patience=30, close_mosaic=25,
    flipud=0.0, fliplr=0.5, mosaic=0.5, mixup=0.0, copy_paste=0.0,
    hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, scale=0.5,
    project="/content/runs_gate", name="gate_yolo11s", device=0, workers=8,
)
# OOM? lower batch to 8. Underfitting at cap? raise epochs.
BEST = "/content/runs_gate/gate_yolo11s/weights/best.pt"
```

## Cell 5 — report BOTH val and test mAP
(`train` already validates on `val`; we add an explicit `test` split — the recovered `train_detector.py --eval-only` only does val.)
```python
print("VAL :", YOLO(BEST).val(data=yml, split="val",  device=0).results_dict)
print("TEST:", YOLO(BEST).val(data=yml, split="test", device=0).results_dict)
```

## Cell 6 — download the weights
```python
from google.colab import files
files.download(BEST)        # -> save locally as gate_yolo11s.pt for the probe
```

---

## After download
Locally, run the generalization probe on our clips (model kept out of git):
```bash
python gate_detection/probe_gates.py --model gate_yolo11s.pt
```
Then review the overlays in `outputs/gate_probe/` against the go/no-go (qiaobo gates detected + barriers rejected; outdoor no hallucination).

### Optional: fallback / attribution arms
- If `yolo11s` over-fires on barriers → re-train mixing in the `ablation_neg0` hard-negatives (Cell 4 with a combined data.yaml).
- If `yolo11s` underperforms → also train `yolov8n.pt` (original recipe) and compare on the probe.
