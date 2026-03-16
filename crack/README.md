# Crack Detection Module
### SID — Structural Inspection Drone · IDEATOR GECT

This module handles automated crack detection on concrete surfaces using a YOLOv11s segmentation model. It covers training, inference, real-world crack analysis, and a REST API for integration with the broader SID system.

---

## Files in This Folder

```
crack/
│
├── crack.ipynb                 ← MAIN FILE — full training pipeline (run on Google Colab)
├── crack_dataset.yaml          ← dataset config: classes, train/val split paths
├── convert_to_yolo.py          ← converts raw annotated data to YOLO format
├── Colab_Training_Guide.md     ← step-by-step guide to reproduce training on Colab
├── documentation.md            ← project notes, dataset sources, references
│
├── yolo11n-seg.pt              ← base nano model (kept for reference only)
│                                  training uses yolo11s-seg.pt (small)
│
├── best.pt                     ← TRAINED MODEL WEIGHTS — use this for all inference
├── results.png                 ← training metrics graph (loss, mAP across epochs)
│
├── crack_dataset/              ← raw dataset
│   ├── images/
│   └── labels/
│
├── analysis/                   ← crack geometry + severity logic
│   ├── geometry.py             ← extracts pixel-level crack dimensions from bbox
│   ├── severity.py             ← classifies severity + generates JSON output
│   └── output_schema.json      ← authoritative output format (reference this)
│
├── pipeline/
│   └── realtime.py             ← runs inference on live video / USB camera
│
└── api/
    └── app.py                  ← Flask/FastAPI endpoint for image upload + detection
```

---

## Dataset

The YOLO-formatted dataset is not stored in this repository. It was generated from the raw annotated data using `convert_to_yolo.py` and is shared as a ZIP file on Google Drive.

**Download:** `[ INSERT GOOGLE DRIVE LINK ]` — `crack_yolo_data.zip`

Place the ZIP in your Google Drive at:
```
MyDrive/crack_yolo_data.zip
```
The training notebook (`crack.ipynb`) expects it at this exact path.

**Dataset details**
- Format: YOLO segmentation (images + polygon `.txt` label files)
- Classes: `crack` (single class, index 0)
- Split: defined in `crack_dataset.yaml` — do not modify the split

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLOv11s-seg (small segmentation) |
| Base weights | `yolo11s-seg.pt` (downloaded by Colab automatically) |
| Trained weights | `best.pt` — highest validation mAP checkpoint |
| Input resolution | 640 × 640 |
| Epochs | 100 |
| Batch size | 16 |
| Device | GPU (Google Colab T4) |

> **Always use `best.pt` for inference — not `last.pt`.**  
> If a retraining produces better results, the new file will be named `best_v2.pt` and this README will be updated.

---

## Training

Training is done on Google Colab using `crack.ipynb`. To reproduce:

**1. Open the notebook in Colab**

Upload `crack.ipynb` directly or open it from GitHub via `File → Open notebook → GitHub`.

**2. Make sure the dataset ZIP is in your Drive**

```
MyDrive/crack_yolo_data.zip
```

**3. Run all cells in order**

| Cell | What it does |
|---|---|
| 1 | Mounts Google Drive |
| 2 | Copies and extracts the dataset ZIP, installs `ultralytics` |
| 3 | Updates `crack_dataset.yaml` with the correct Colab paths |
| 4 | Loads `yolo11s-seg.pt` and runs training for 100 epochs |
| 5 | Runs inference on a test image to verify the trained model |
| 6 | Copies `best.pt` and `results.png` back to Google Drive |

**4. After training**

Download `best.pt` and `results.png` from Drive and replace the files in this folder. Commit both to the repository.

For a more detailed walkthrough, see `Colab_Training_Guide.md`.

---

## Training Metrics

> Metrics will be updated here after the next training run. Fill in the values from the final epoch output.

| Metric | Value |
|---|---|
| Epoch reached | — / 100 |
| GPU memory | — |
| Box Precision | — |
| Box Recall | — |
| Box mAP@50 | — |
| Box mAP@50-95 | — |
| Mask mAP@50 | — |
| box_loss | — |
| cls_loss | — |
| dfl_loss | — |

---

## Inference — Quick Start

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('crack/best.pt')

# Run on a single image
results = model.predict('test_image.jpg', conf=0.45, imgsz=640)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        print(f'Crack at [{x1},{y1},{x2},{y2}]  conf={confidence:.2f}')
```

---

## Analysis Layer — Crack Dimensions and Severity

Once a crack is detected, the analysis layer computes pixel-level dimensions and a severity classification.

**geometry.py** — called by Viswajith's severity module and the pipeline

```python
def extract_geometry(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    width_px  = abs(x2 - x1)
    height_px = abs(y2 - y1)
    length_px = (width_px**2 + height_px**2) ** 0.5   # diagonal
    crack_width_px = min(width_px, height_px)           # short axis
    relative_width = crack_width_px / image_width

    return {
        "length_px":      round(length_px, 1),
        "width_px":       crack_width_px,
        "relative_width": round(relative_width, 4)
    }
```

**severity.py** — thresholds are starting values, calibrate against real images

```python
def classify_severity(relative_width):
    if relative_width < 0.01:
        return 'Minor'
    elif relative_width < 0.03:
        return 'Moderate'
    else:
        return 'Severe'
```

> **Why relative_width and not mm?**  
> Converting pixels to mm requires knowing the distance from the camera to the surface (depth). This will be available once the ZED2 stereo camera is integrated in Month 4. Until then, `relative_width` is the most reliable scale-agnostic measure. The `geometry_mm` fields in the JSON output are intentionally `null` for now.

---

## JSON Output Format

Every detection produces a structured JSON object. The full schema is in `analysis/output_schema.json`. A simplified example:

```json
{
  "session_id": "sid_20250316_143022",
  "timestamp": "2025-03-16T14:30:22Z",
  "source": "usb_camera",
  "frame_id": 142,
  "image_width_px": 1280,
  "image_height_px": 720,
  "location": {
    "label": "bridge_deck_faceA",
    "latitude": null,
    "longitude": null
  },
  "model": {
    "name": "yolov11s-seg",
    "weights": "best.pt"
  },
  "detections": [
    {
      "detection_id": "det_142_001",
      "bbox_px": [100, 120, 250, 160],
      "confidence": 0.94,
      "geometry_px": {
        "length_px": 158,
        "width_px": 40,
        "relative_width": 0.031
      },
      "geometry_mm": {
        "width_mm": null,
        "stereo_available": false
      },
      "severity": {
        "level": "Moderate",
        "basis": "relative_width"
      }
    }
  ],
  "summary": {
    "total_detections": 1,
    "highest_severity": "Moderate",
    "flagged": false
  }
}
```

The full field reference is in `SID_Software_Plan_Month1.docx` — Section 7.

---

## API

Sreeda's Flask/FastAPI app in `api/app.py` exposes two endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/detect` | POST | Upload an image, returns JSON detection |
| `/session/{session_id}` | GET | Returns all detections for a session |

Session logs are saved to `logs/{session_id}.json` and `logs/{session_id}.csv`.

---

## Installation

```bash
# From the repo root
python -m venv venv
source venv/bin/activate      # Mac / Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt

# requirements.txt:
# ultralytics>=8.0.0
# opencv-python>=4.8
# flask>=3.0
# numpy>=1.24
```

---

## Evaluation Guidance

When testing `best.pt` on unseen images, use these thresholds to decide if the model is ready:

| Metric | Minimum target | Action if below |
|---|---|---|
| Box mAP@50 (unseen) | 0.70 | Flag — retrain decision needed |
| Box Precision | 0.80 | Check false positive rate |
| Box Recall | 0.60 | Check dataset class balance |
| Confidence threshold | 0.45 | Adjust based on FP/FN |

---

## IS 456:2000 Alignment (Month 4)

Once stereo depth is available, severity will be reclassified against Indian Standards permissible crack widths:

| Exposure class | Permissible crack width |
|---|---|
| Mild | 0.30 mm |
| Moderate | 0.20 mm |
| Severe | 0.10 mm |

The `is456_compliance` block in the JSON schema is reserved for this.

---

*IDEATOR GECT · Centre for Innovation · Government Engineering College Thrissur*