# SID — Structural Inspection Drone
### Software Repository · IDEATOR GECT

An autonomous drone-based system for structural inspection of bridges and civil infrastructure. SID uses computer vision and AI to detect cracks, corrosion, and spalling — replacing manual, hazardous inspection methods with fast, repeatable, data-driven assessments.

> Built by the software team at IDEATOR GECT, Government Engineering College Thrissur, under the guidance of **Prof. Jikhil Joseph, M.Tech** (Asst. Professor, Civil Engineering Dept).

---

## What This System Does

SID equips a hexacopter drone with RGB, infrared, and stereo cameras. Onboard AI models process the video feed in real time to detect and classify structural defects. Each detection is geotagged, logged as structured JSON, and fed into a dashboard for field engineers.

**Defects targeted:** cracks · corrosion · spalling  
**Hardware:** Tarot 680 Pro hexacopter · ZED2 stereo camera · FLIR IR camera · Pixhawk Cube Orange FC  
**Edge compute:** Jetson Nano (TensorRT optimised inference)

---

## Repository Structure

```
SID-Software/
│
├── SID_Software_Plan_Month1.docx   ← full team execution plan (Month 1)
│
├── crack/                          ← Crack Detection Module (Month 1)
│   └── README.md                   ← see this for training + inference details
│
└── corrosion_dataset/              ← Corrosion dataset (Month 2 prep)
    ├── images/
    └── annotations/
```

Each module has its own README with setup and usage instructions. Start with the one relevant to your task.

---

## Team

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

## Month-by-Month Roadmap

| Month | Module | Status |
|---|---|---|
| 1 | Crack Detection (YOLOv11s-seg, analysis layer, API) | 🔄 In progress |
| 2 | Corrosion Detection (RGB + IR fusion) | ⏳ Dataset prep underway |
| 3 | Spalling Detection (visual + acoustic fusion) | ⏳ Planned |
| 4 | Stereo integration — real-world mm measurements | ⏳ Planned |
| 5 | GPS geotagging + BIM integration | ⏳ Planned |
| 6–8 | Full system integration, Jetson deployment, field testing | ⏳ Planned |

---

## Quick Links

- 📄 [Team Execution Plan](./SID_Software_Plan_Month1.docx)
- 🧠 [Crack Module](./crack/README.md)
- 📦 Dataset ZIP (Google Drive) — `[ INSERT DRIVE LINK ]`
- 🔬 Project Proposal — `[ INSERT LINK IF AVAILABLE ]`

---

## Contributing

1. Clone the repo and create a branch: `yourname/task` (e.g. `devika/geometry`)
2. Commit your work with a clear message: `feat: add geometry.py crack width extractor`
3. Open a pull request — do not push directly to `main`
4. Do not share code over WhatsApp — always commit to GitHub

---

*IDEATOR GECT · Centre for Innovation · Government Engineering College Thrissur*