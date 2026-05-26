# SID — Structural Inspection Drone
### 🔍 Crack Detection, Sub-Pixel Width Estimation & BIM Integration
**IDEATOR GECT — Government Engineering College Thrissur**

---

Welcome to the core software repository for the **Structural Inspection Drone (SID)**. This system provides an end-to-end, high-precision pipeline designed for structural health monitoring of bridges. It detects concrete surface cracks using deep learning, estimates their true physical widths with sub-pixel accuracy, classifies structural severity based on Indian Standards (**IS 456:2000**), maps the damage onto 3D BIM (**IFC**) models, and outputs interactive inspection reports.

---

## 🏗️ Project Architecture

The codebase is organized into modular, decoupled packages:

```
SID-Software-Team-Ideator/
├── crack_detection/              # Core Detection & Analysis Engine
│   ├── schemas.py                # Data schemas (Pydantic v2 validation)
│   ├── detector.py               # YOLOv11s-seg model wrapper
│   ├── segmentation.py           # Morphology & PCA orientation extraction
│   ├── severity.py               # IS 456:2000 exposure-based classifier
│   └── width_estimation/
│       ├── common.py             # Scale factors and GSD helpers
│       ├── calibration.py        # Altitude, ArUco reference, & checkerboard calibration
│       ├── monocular.py          # State-of-the-art skeleton & perpendicular width estimation
│       └── stereo.py             # ZED2 stereo camera depth-based 3D width estimation
│
├── bim_mapping/                  # Building Information Modeling (BIM) Integration
│   ├── coordinate_transform.py   # Image ➔ Camera ➔ World ➔ IFC coordinate transformation
│   ├── mapper.py                 # Spatial queries matching cracks to IFC structural elements
│   ├── ifc_annotator.py          # IfcSurfaceFeature annotation and Property Set writer
│   └── visualizer.py             # Interactive 3D Plotly/Matplotlib visualizer
│
├── pipeline/                     # Workflow Orchestration
│   ├── inspector.py              # End-to-end pipeline (Image, Directory, Video streams)
│   └── report_generator.py       # Exporters for interactive HTML, JSON, and CSV reports
│
├── tests/                        # Comprehensive Pytest Suite
│   ├── conftest.py               # Pytest environment fixtures
│   ├── fixtures/                 # Synthetic crack generators for math verification
│   ├── test_detector.py          # YOLO wrapper mocks
│   ├── test_width_monocular.py   # 19 tests validating sub-pixel math and outlier filtering
│   ├── test_width_stereo.py      # ZED2 projection verification tests
│   ├── test_bim_mapping.py       # IFC reader and spatial transform checks
│   └── test_pipeline.py          # End-to-end integration tests
│
├── docs/                         # Theoretical documentation
│   └── width_estimation_algorithms.md
│
├── best.pt                       # Trained YOLOv11s-seg model weights (20.5 MB)
├── run_demo.py                   # CLI demo interface (Synthetic benchmark & batch processing)
├── run_webcam.py                 # Interactive real-time webcam testing tool (with ArUco calibration)
├── requirements.txt              # Project dependencies
└── setup.py                      # Package installation script
```

---

## 🚀 Quick Start (Dev Environment Setup)

### 1. Prerequisites
Ensure you have **Python 3.9+** and `pip` installed.

### 2. Set Up Virtual Environment
Run the following commands in your terminal from the project directory:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate       # Mac / Linux
# OR
venv\Scripts\activate          # Windows

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Testing the Modules

We provide three simple entry points to verify and test the software suite:

### 1. Synthetic Data Benchmark (Algorithm Verification)
Verify the mathematical accuracy of the sub-pixel width estimation algorithm using synthetic cracks with known ground-truth dimensions (from 2px to 20px). It doesn't require a camera or GPU.
```bash
python run_demo.py --mode demo
```
*Outputs detailed statistics and an interactive HTML report to `inspection_output/demo/`.*

### 2. Real-Time Webcam Stream (Interactive Test)
Test the crack detection and width-measurement pipeline live using your computer's webcam.
```bash
python run_webcam.py --model best.pt --confidence 0.40
```
* **Milimeter Calibration:** Hold a printed **ArUco marker** (DICT_4X4_50, e.g. ID 0) at a known size (default 50.0mm) next to a crack. The webcam will automatically detect it and instantly calibrate measurements from pixels to millimeters.
* **Controls:** Press `q` to quit, `c` to reset calibration, `s` to save a screenshot, and `+`/`-` to modify the model's confidence threshold.

### 3. Run the Automated Test Suite
Run 60 comprehensive unit and integration tests using pytest:
```bash
pytest tests/ -v
```
*(All 55 non-BIM tests pass. BIM mapping tests are automatically skipped if the optional `ifcopenshell` package is not installed).*

---

## 📖 In-Depth Guides

*   Read [WALKTHROUGH.md](./WALKTHROUGH.md) for a beginner-friendly tutorial on setting up, testing with your webcam, and generating reports.
*   Read [BEGINNERS_GUIDE.md](./BEGINNERS_GUIDE.md) for the absolute beginner's introduction to SID.
*   Read [docs/width_estimation_algorithms.md](./docs/width_estimation_algorithms.md) for a detailed explanation of the mathematics behind the skeletonization and perpendicular ray casting width estimation algorithm.

---

## 👥 Team

| Name | Batch | Role |
|---|---|---|
| Gabriel | S4 CSE | Software Lead |
| Devika | S2 CSE | Crack Analysis — Geometry |
| Viswajith M P | S2 CSE | Crack Analysis — Severity & JSON |
| Aswin | S6 ECE | Model Evaluation |
| Hridhya | S6 ECE | Real-Time Pipeline |
| Sreeda | S6 CSE | API Development |
| Maria | S6 CSE | Corrosion Dataset |
| Sreehari | S4 ECE | Corrosion Dataset |

---

## 📅 Month-by-Month Roadmap

| Month | Module | Status |
|---|---|---|
| 1 | Crack Detection (YOLOv11s-seg, analysis layer, API) | ✅ Complete |
| 2 | Corrosion Detection (RGB + IR fusion) | ⏳ Dataset prep underway |
| 3 | Spalling Detection (visual + acoustic fusion) | ⏳ Planned |
| 4 | Stereo integration — real-world mm measurements | ⏳ Planned |
| 5 | GPS geotagging + BIM integration | ⏳ Planned |
| 6–8 | Full system integration, Jetson deployment, field testing | ⏳ Planned |

---

## 📄 Quick Links

- 📄 [Team Execution Plan](./docs/planning/SID_Plan_final.pdf)
- 🧠 [Width Estimation Algorithms](./docs/width_estimation_algorithms.md)
- 📦 [Dataset ZIP (Google Drive)](https://drive.google.com/file/d/1tmmljYEAbh6Ng9Bt0vUibe6p62VnvyAv/view?usp=sharing)

---

## 🤝 Contributing

1. Clone the repo and create a branch: `yourname/task` (e.g. `devika/geometry`)
2. Commit your work with a clear message: `feat: add geometry.py crack width extractor`
3. Open a pull request — do not push directly to `main`
4. Do not share code over WhatsApp — always commit to GitHub

---

*IDEATOR GECT · Centre for Innovation · Government Engineering College Thrissur*
